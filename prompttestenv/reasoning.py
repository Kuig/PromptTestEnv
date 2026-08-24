"""Reasoning trace analysis for thinking-enabled candidates.

This module provides functions to segment a candidate's reasoning trace into
4 cognitive categories (Interpretation, Planning, Pure Reasoning, Output
Formulation) using an LLM judge, extract numeric quality metrics, and
aggregate results across multiple repetitions.

Usage:
    from prompttestenv.reasoning import analyze_reasoning, aggregate_reasoning_stats
"""
from __future__ import annotations

import json
import statistics

import prompttestenv.logger as logger
from prompttestenv.api import get_llm_response
from prompttestenv.models import JudgeConfig, ReasoningStats


def compute_reasoning_percentages(stats: ReasoningStats) -> None:
    """Calculate character-count proportions for each segmentation category.

    Mutates the ``stats`` object in-place by populating the ``*_pct`` fields.
    The denominator is the total character count across all 4 categories.
    If total is zero, all percentages remain 0.0.

    Args:
        stats: A ReasoningStats instance whose text fields are already populated.
    """
    total = (
        len(stats.interpretation)
        + len(stats.planning)
        + len(stats.pure_reasoning)
        + len(stats.output_formulation)
    )
    if total == 0:
        return
    stats.interpretation_pct = round(len(stats.interpretation) / total * 100, 2)
    stats.planning_pct = round(len(stats.planning) / total * 100, 2)
    stats.pure_reasoning_pct = round(len(stats.pure_reasoning) / total * 100, 2)
    stats.output_formulation_pct = round(len(stats.output_formulation) / total * 100, 2)


def analyze_reasoning(
    reasoning_text: str,
    judge_config: JudgeConfig,
    candidate_response: str = "",
) -> ReasoningStats | None:
    """Analyze a candidate's reasoning trace using a dedicated LLM judge.

    Performs two sequential LLM calls:
    1. Segmentation: divides the reasoning trace into 4 cognitive categories.
    2. Metrics: extracts alt_path, autocorrect and alignment_score counts.
       The candidate's final response is also passed so that alignment_score
       can be evaluated against the actual output, not just the trace.

    Returns None if reasoning_text is empty, reasoning_analysis is disabled,
    or if the required templates are missing.

    Args:
        reasoning_text: The raw thinking transcript from the candidate model.
        judge_config: JudgeConfig instance containing reasoning_judge settings.
        candidate_response: The final response produced by the candidate model.
            Used by the metrics_template to evaluate alignment_score.

    Returns:
        A populated ReasoningStats instance, or None if analysis was skipped.
    """
    if not judge_config.reasoning_analysis:
        return None
    if not reasoning_text.strip():
        return None

    rj = judge_config.reasoning_judge
    if not rj.segmentation_template or not rj.metrics_template:
        logger.log_warning(
            "Reasoning analysis enabled but segmentation_template or "
            "metrics_template is missing in reasoning_judge config. Skipping."
        )
        return None

    stats = ReasoningStats()

    # --- Call 1: Segmentation ---
    try:
        seg_prompt = rj.segmentation_template.format(reasoning_text=reasoning_text)
        seg_resp, _, _, _ = get_llm_response(
            provider=rj.provider,
            model_name=rj.model,
            system_instruction=rj.reasoning_system_prompt or None,
            user_prompt=seg_prompt,
            temp=rj.temperature,
            thinking=rj.thinking,
            response_mime_type="application/json",
        )
        seg_data = json.loads(seg_resp)
        if isinstance(seg_data, list) and seg_data:
            seg_data = seg_data[0]
        stats.interpretation = seg_data.get("interpretation", "")
        stats.planning = seg_data.get("planning", "")
        stats.pure_reasoning = seg_data.get("pure_reasoning", "")
        stats.output_formulation = seg_data.get("output_formulation", "")
    except Exception as exc:
        logger.log_error(f"Reasoning segmentation failed: {exc}")
        return None

    compute_reasoning_percentages(stats)

    # --- Call 2: Metrics ---
    try:
        met_prompt = rj.metrics_template.format(
            reasoning_text=reasoning_text,
            candidate_response=candidate_response,
        )
        met_resp, _, _, _ = get_llm_response(
            provider=rj.provider,
            model_name=rj.model,
            system_instruction=rj.reasoning_system_prompt or None,
            user_prompt=met_prompt,
            temp=rj.temperature,
            thinking=rj.thinking,
            response_mime_type="application/json",
        )
        met_data = json.loads(met_resp)
        if isinstance(met_data, list) and met_data:
            met_data = met_data[0]
        stats.alt_path = int(met_data.get("alt_path", 0))
        stats.autocorrect = int(met_data.get("autocorrect", 0))
        stats.alignment_score = max(1, min(10, int(met_data.get("alignment_score", 5))))
    except Exception as exc:
        logger.log_error(f"Reasoning metrics extraction failed: {exc}")
        # Segmentation succeeded — return partial result without metrics
        return stats

    return stats


def aggregate_reasoning_stats(analyses: list[dict]) -> dict:
    """Aggregate per-repetition reasoning analysis results into mean ± std.

    Accepts the raw list of dicts stored in CandidatePerformance.reasoning_analyses
    and returns a plain dict suitable for report rendering and verdict generation.

    Args:
        analyses: List of per-repetition analysis dicts, each being the output
            of ReasoningStats.to_dict().

    Returns:
        Dict with keys:
            avg_interpretation_pct, std_interpretation_pct,
            avg_planning_pct, std_planning_pct,
            avg_pure_reasoning_pct, std_pure_reasoning_pct,
            avg_output_formulation_pct, std_output_formulation_pct,
            avg_alt_path, std_alt_path,
            avg_autocorrect, std_autocorrect,
            avg_alignment_score, std_alignment_score.
        Returns an empty dict if analyses is empty.
    """
    if not analyses:
        return {}

    def _avg_std(key: str) -> tuple[float, float]:
        vals = [a[key] for a in analyses if key in a]
        if not vals:
            return 0.0, 0.0
        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return round(avg, 2), round(std, 2)

    result: dict = {}
    for field_name in (
        "interpretation_pct",
        "planning_pct",
        "pure_reasoning_pct",
        "output_formulation_pct",
        "alt_path",
        "autocorrect",
        "alignment_score",
    ):
        avg, std = _avg_std(field_name)
        result[f"avg_{field_name}"] = avg
        result[f"std_{field_name}"] = std

    return result
