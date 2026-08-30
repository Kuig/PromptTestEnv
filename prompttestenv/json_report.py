"""JSON export of a benchmark report.

The alternative renderer to reporting.py: same inputs, same content, structured
for a program instead of for a page. It is kept in its own module because it
shares no code at all with the HTML side — not one helper — so the two only ever
had their imports in common.

What it must keep doing, and why:

- Read pool_by_candidate and aggregate_reasoning_stats rather than pooling
  anything itself. They are what the Jinja template reads too, which is the only
  reason the JSON and the HTML cannot describe a candidate differently.
- Write the -1 "not measured" sentinel through unchanged instead of mapping it
  to null. progress.jsonl already writes -1, and a consumer reading both must
  not meet two conventions. report.schema.json documents it per field, including
  where 0.0 is a real measurement instead.
- Emit best.reasoning_text in full, because the analysis units are character
  offsets into it and are unresolvable without it.

report.schema.json (repo root, and a copy under templates/) is this module's
published contract; tests/test_report_schema.py fails if they drift apart.
"""
from __future__ import annotations

import datetime
import json
import os

from prompttestenv.config import get_app_config
from prompttestenv.models import (
    JUDGE_TYPE_ASSERT,
    JUDGE_TYPE_LLM,
    JUDGE_TYPE_SIMILARITY,
    Candidate,
    CandidatePerformance,
    calculate_stats,
    pool_by_candidate,
    JudgeConfig,
    TestCaseResult,
    GlobalCriteria,
)
from prompttestenv.reasoning import aggregate_reasoning_stats
from prompttestenv.verdict import parse_grouped_verdict


JSON_SCHEMA_VERSION = "1.0"


def _stat_block(values: list) -> dict:
    """Render one measured quantity as mean, std and every raw value.

    The HTML shows only mean and std; the JSON keeps the per-repetition list
    too, since it is already in memory on the CandidatePerformance and is the
    one thing a consumer cannot recompute from an aggregate.

    Args:
        values: The raw per-repetition values, in the order they were appended.

    Returns:
        Dict with ``mean``, ``std`` and ``values``. A -1 mean means "not
        measured", the sentinel used throughout the codebase; for coverages and
        rates 0.0 is instead a real measurement.
    """
    mean, std = calculate_stats(values, default_val=-1.0)
    return {"mean": mean, "std": std, "values": list(values)}


def _perf_block(perf: CandidatePerformance) -> dict:
    """Serialize the measured quantities shared by the pooled and per-test views.

    Args:
        perf: The record to describe, pooled or for a single test case.

    Returns:
        Dict of stat blocks plus both cost-per-point statistics.
    """
    return {
        "score": _stat_block(perf.scores),
        "global_score": _stat_block(perf.global_scores),
        "tokens": _stat_block(perf.tokens),
        "reasoning_tokens": _stat_block(perf.reasoning_tokens),
        "time": _stat_block(perf.times),
        # Two different statistics that need not agree: "pooled" is the ratio of
        # the means, "mean"/"std" the mean of each repetition's own ratio. The
        # HTML shows both, in two different places, so the JSON carries both.
        "cost_per_point": {
            "pooled": perf.reasoning_cost_per_point,
            "mean": perf.mean_cost_per_point,
            "std": perf.std_cost_per_point,
        },
    }


def _verdict_block(verdict_text: str) -> dict:
    """Split the verdict into its groups when it is a grouped one.

    Args:
        verdict_text: The verdict exactly as stored on the ``verdict`` event.

    Returns:
        Dict with ``grouped``, ``groups``, ``global_verdict`` and the raw
        ``text``. For an ungrouped verdict the whole text is the global one.
    """
    data = parse_grouped_verdict(verdict_text)
    if data is None:
        return {
            "grouped": False,
            "groups": [],
            "global_verdict": verdict_text,
            "text": verdict_text,
        }
    return {
        "grouped": True,
        "groups": [
            {"group_name": g.get("group_name", ""), "verdict": g.get("verdict", "")}
            for g in data.get("groups", [])
        ],
        "global_verdict": data.get("global_verdict", ""),
        "text": verdict_text,
    }


def _active_global_criteria(global_criteria: GlobalCriteria) -> str:
    """The criteria text the project's global mode actually uses.

    GlobalCriteria holds one field per mode and only one of them is live, which
    the HTML template selects inline. The JSON resolves it here so a consumer
    does not have to know the mode-to-field mapping.

    Args:
        global_criteria: The project's global criteria.

    Returns:
        The active criteria string, empty when the mode is "none".
    """
    return {
        JUDGE_TYPE_LLM: global_criteria.llm_judge_criteria,
        JUDGE_TYPE_SIMILARITY: global_criteria.similarity_criteria,
        JUDGE_TYPE_ASSERT: global_criteria.assert_criteria,
    }.get(global_criteria.mode, "")


def _configuration_block(
    candidates: list[Candidate],
    global_criteria: GlobalCriteria,
    judge_config: JudgeConfig,
) -> dict:
    """Describe what produced the numbers: candidates, judges and taxonomy.

    Mirrors the report's "Candidates Configuration" footer.
    ``resolved_system_instruction`` is deliberately not emitted: it is derived
    from system_prompt_file and can be very large, the same reason
    projectio.py keeps it out of candidates.json.

    Args:
        candidates: Resolved candidate configurations.
        global_criteria: The project's global criteria.
        judge_config: The project's judge configuration.

    Returns:
        Dict describing the run's configuration.
    """
    schema = get_app_config().reasoning_schema
    return {
        "candidates": [
            {
                "name": cand.name,
                "provider": cand.provider,
                "model": cand.model,
                "temperature": cand.temperature,
                "disable_safety": cand.disable_safety,
                "thinking": cand.thinking,
                "system_prompt_file": cand.system_prompt_file,
            }
            for cand in candidates
        ],
        "repetitions": judge_config.repetitions,
        "group_verdicts": judge_config.group_verdicts,
        "global_criteria": {
            "mode": global_criteria.mode,
            "criteria": _active_global_criteria(global_criteria),
        },
        "test_judge": {
            "provider": judge_config.test_judge.provider,
            "model": judge_config.test_judge.model,
            "temperature": judge_config.test_judge.temperature,
            "disable_safety": judge_config.test_judge.disable_safety,
            "thinking": judge_config.test_judge.thinking,
        },
        "verdict_judge": {
            "provider": judge_config.verdict_judge.provider,
            "model": judge_config.verdict_judge.model,
            "temperature": judge_config.verdict_judge.temperature,
            "disable_safety": judge_config.verdict_judge.disable_safety,
            "thinking": judge_config.verdict_judge.thinking,
        },
        "reasoning": {
            "enabled": judge_config.reasoning_enabled,
            "scope": judge_config.reasoning_analysis,
            "schema_stamp": schema.stamp,
            "intensity_scale": schema.intensity_scale,
            "dimensions": [
                {"name": d.name, "definition": d.definition, "color": d.color}
                for d in schema.dimensions
            ],
            "judge": {
                "provider": judge_config.reasoning_judge.provider,
                "model": judge_config.reasoning_judge.model,
                "temperature": judge_config.reasoning_judge.temperature,
                "thinking": judge_config.reasoning_judge.thinking,
                "dimension_mode": judge_config.reasoning_judge.dimension_mode,
            },
        },
    }


def generate_json_report(
    project_dir: str,
    results: list[TestCaseResult],
    candidates: list[Candidate],
    verdict_text: str,
    global_criteria: GlobalCriteria,
    judge_config: JudgeConfig,
    filename: str = "report_benchmark.json",
) -> str:
    """Export the report as structured JSON instead of a rendered page.

    Same inputs and same content as generate_html_report — an alternative
    renderer, not a second aggregation: the pooled figures come from
    pool_by_candidate and the reasoning profile from aggregate_reasoning_stats,
    exactly as the template does, so the two surfaces cannot disagree.

    What it adds over the HTML is the per-repetition raw values the page
    reduces to mean and std. What it leaves out is the responses of the
    repetitions that were not the best one: those live in progress.jsonl.

    The -1 "not measured" sentinel is written through unchanged rather than
    mapped to null, so this file and progress.jsonl share one convention. The
    schema (report.schema.json) documents it field by field, including where
    0.0 is a real measurement instead.

    Args:
        project_dir: Path to the benchmark project directory.
        results: Completed test case results, reasoning already attached.
        candidates: Resolved candidate configurations.
        verdict_text: The verdict as stored, grouped JSON or plain text.
        global_criteria: The project's global criteria.
        judge_config: The project's judge configuration.
        filename: Report file name, written under ``Report/``.

    Returns:
        Path to the written JSON file.
    """
    stats = pool_by_candidate(candidates, results)

    aggregate = {}
    for name, perf in stats.items():
        block = _perf_block(perf)
        block["combined_avg"] = perf.combined_avg
        block["reasoning_profile"] = aggregate_reasoning_stats(perf.reasoning_analyses)
        aggregate[name] = block

    test_cases = []
    for row in results:
        per_candidate = {}
        for cand in candidates:
            perf = row.candidates_perf.get(cand.name)
            if not perf:
                continue
            block = _perf_block(perf)
            block["best"] = {
                "output": perf.best_output,
                "reason": perf.best_reason,
                "global_reason": perf.best_global_reason,
                # Emitted in full because the analysis units below are character
                # offsets into it: without the text they cannot be resolved.
                "reasoning_text": perf.best_reasoning_text,
                "reasoning_analysis": (
                    perf.best_reasoning_analysis.to_dict()
                    if perf.best_reasoning_analysis is not None
                    else None
                ),
            }
            per_candidate[cand.name] = block
        test_cases.append({
            "test_id": row.test_id,
            "group": row.group,
            "judge_type": row.judge_type,
            "prompt": row.prompt,
            "criteria": row.criteria,
            "files_used": list(row.files_used),
            "candidates": per_candidate,
        })

    payload = {
        "schema_version": JSON_SCHEMA_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "project_name": os.path.basename(project_dir),
        "verdict": _verdict_block(verdict_text),
        "configuration": _configuration_block(candidates, global_criteria, judge_config),
        "aggregate": aggregate,
        "test_cases": test_cases,
    }

    report_dir = os.path.join(project_dir, "Report")
    os.makedirs(report_dir, exist_ok=True)
    json_file = os.path.join(report_dir, filename)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return json_file
