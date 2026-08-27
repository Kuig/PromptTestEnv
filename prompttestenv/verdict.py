from __future__ import annotations

import json
import os

import prompttestenv.logger as logger
from prompttestenv.api import get_llm_response
from prompttestenv.config import get_app_config
from prompttestenv.models import (
    TestCaseResult,
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    DEFAULT_GROUP,
    JUDGE_TYPE_ASSERT,
    JUDGE_TYPE_LLM,
    JUDGE_TYPE_SIMILARITY,
    REASONING_DIMENSIONS,
    REASONING_SCOPE_ALL,
    REASONING_SCOPE_BEST,
    pool_by_candidate,
)
from prompttestenv.reasoning import aggregate_reasoning_stats


def save_verdict_debug_file(
    project_dir: str,
    sys_prompt: str,
    prompt: str,
    provider: str,
    model: str,
    temp: float,
) -> None:
    """Save verdict judge payload and configuration to a debug file.

    Args:
        project_dir: Benchmark project directory path.
        sys_prompt: System instruction used for the verdict judge.
        prompt: Full user prompt sent to the verdict judge.
        provider: LLM provider name.
        model: Model identifier.
        temp: Sampling temperature.
    """
    debug_path = os.path.join(project_dir, "verdict_prompt_debug.txt")
    debug_content = (
        "=" * 72 + "\nVERDICT JUDGE DEBUG INFO\n" + "=" * 72 + "\n\n"
        f"PROVIDER: {provider}\nMODEL:    {model}\nTEMP:     {temp}\n\n"
        + "-" * 72 + "\nSYSTEM INSTRUCTION:\n" + "-" * 72 + f"\n\n{sys_prompt}\n\n"
        + "-" * 72 + "\nUSER PROMPT (PAYLOAD):\n" + "-" * 72 + f"\n\n{prompt}\n"
    )
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(debug_content)
    except Exception as exc:
        logger.log_warning(f"Failed to write verdict debug file: {exc}")


def parse_grouped_verdict(verdict_text: str) -> dict | None:
    """Parse verdict_text as grouped-verdict JSON, if it is one.

    Args:
        verdict_text: Raw verdict text — either the JSON object produced by
            generate_verdict()'s grouped-verdict path, or plain/Markdown text.

    Returns:
        The parsed dict when verdict_text is valid JSON with "is_grouped": true,
        otherwise None.
    """
    stripped = verdict_text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except Exception:
        return None
    return data if data.get("is_grouped") else None


def _strip_code_fence(text: str) -> str:
    """Strip a wrapping Markdown code fence from LLM output, if present.

    Args:
        text: Response text, already .strip()-ed.

    Returns:
        text with the first/last fence lines removed if text starts with
        ``` and ends with a lone ``` line; otherwise text unchanged.
    """
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _format_score(mean: float, std: float, with_std: bool) -> str:
    """Render a mean±std score, or "N/A" when it was not measured (-1).

    Args:
        mean: The score's mean, or -1 if not measured.
        std: The score's standard deviation.
        with_std: Whether to append the standard deviation.

    Returns:
        Formatted score string.
    """
    if mean < 0:
        return "N/A"
    if with_std:
        return f"{mean:.2f} ± {std:.2f}/10"
    return f"{mean:.2f}/10"


def _format_global_score(perf: CandidatePerformance, with_std: bool) -> str:
    """Render a global score, or "N/A" when global scoring produced nothing.

    Args:
        perf: Performance record to read the global score from.
        with_std: Whether to append the standard deviation.

    Returns:
        Formatted score string.
    """
    return _format_score(perf.global_score_mean, perf.global_score_std, with_std)


def _format_task_score(perf: CandidatePerformance, with_std: bool) -> str:
    """Render a task score, or "N/A" when every judge call for it failed.

    Args:
        perf: Performance record to read the task score from.
        with_std: Whether to append the standard deviation.

    Returns:
        Formatted score string.
    """
    return _format_score(perf.score_mean, perf.score_std, with_std)


def _format_metric(agg: dict, key: str, suffix: str = "", scale: float = 1.0) -> str:
    """Format one aggregated figure, or "not measured" when it has no value.

    Args:
        agg: Aggregated reasoning stats.
        key: Base field name, without the avg_/std_ prefix.
        suffix: Unit to append, such as "%".
        scale: Multiplier applied before formatting.

    Returns:
        A "mean ± sd" string, or "not measured".
    """
    mean = agg.get(f"avg_{key}", -1)
    if mean is None or mean < 0:
        return "not measured"
    std = agg.get(f"std_{key}", 0) or 0
    return f"{mean * scale:.1f}{suffix} ± {std * scale:.1f}{suffix}"


def _format_reasoning_profile(analyses: list[dict], scope: str = REASONING_SCOPE_ALL) -> str:
    """Render the aggregated reasoning trace profile for one candidate.

    The three dimensions are scored independently, so their coverages do not sum
    to 100%: a sentence can belong to more than one at once, and the sum of the
    coverages is reported separately as density. Saying so in the payload matters,
    because a judge shown three percentages will otherwise read them as shares of
    a whole and "explain" the missing remainder.

    Args:
        analyses: Per-repetition reasoning analysis dicts.
        scope: The reasoning_analysis scope the figures were produced under.

    Returns:
        Indented multi-line block describing the trace's composition and metrics.
    """
    agg = aggregate_reasoning_stats(analyses)
    source = "provider SUMMARY of the thinking" if agg.get("is_summary") else "raw thinking trace"
    sampled = (
        "the single highest-scoring repetition of each test, so this profile describes "
        "how the model reasons WHEN IT SUCCEEDS and is not a sample of its typical run"
        if scope == REASONING_SCOPE_BEST
        else "every repetition of each test"
    )
    lines = [
        f"    Reasoning profile over the {source}, measured on {sampled} "
        f"(independent coverages, NOT shares of a whole, so they need not sum to 100%):",
        f"      framing {_format_metric(agg, 'coverage_framing', '%', 100)}, "
        f"solving {_format_metric(agg, 'coverage_solving', '%', 100)}, "
        f"presentation {_format_metric(agg, 'coverage_presentation', '%', 100)}.",
        f"      Density (sum of the coverages; above 1.0 means the dimensions overlap): "
        f"{_format_metric(agg, 'density')}.",
        f"      Alternatives explored: {_format_metric(agg, 'alt_path')}. "
        f"Self-corrections: {_format_metric(agg, 'autocorrect')}. "
        f"Both are counts of cited sentences, so 0 means none were found, not none occurred.",
        f"      Response/reasoning alignment: {_format_metric(agg, 'alignment_score')} out of 10. "
        f"Trace/response similarity: {_format_metric(agg, 'trace_response_drift')}.",
        f"      Repeated-trigram share of the trace: {_format_metric(agg, 'repetition_rate', '%', 100)}.",
    ]
    return "\n".join(lines) + "\n"


class MetadataError(Exception):
    """A metadata section the payload needs is missing or will not format.

    Raised rather than degraded: an absent section does not remove a figure from
    the payload, only the caveat that stops the judge over-claiming about it.
    """


def _section(metadata, name: str, **values) -> str:
    """Return one metadata section, formatted.

    Args:
        metadata: The VerdictMetadata carrying the section texts.
        name: Field name on metadata.
        **values: Placeholder values for str.format.

    Returns:
        The formatted section text.

    Raises:
        MetadataError: If the section is empty or its placeholders do not resolve.
    """
    template = getattr(metadata, name, "")
    if not template.strip():
        raise MetadataError(
            f"verdict_metadata.{name} is empty in config.json, but this payload needs it."
        )
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise MetadataError(
            f"verdict_metadata.{name} in config.json could not be formatted ({exc})."
        ) from exc


def _build_metadata(
    rows: list[TestCaseResult],
    pooled: dict[str, CandidatePerformance],
    judge_config: JudgeConfig,
    profile_shown: bool,
) -> str:
    """Assemble the payload header from the sections this payload actually needs.

    Emitting the whole thing unconditionally meant a project with no reasoning
    analysis read several paragraphs on how to interpret a reasoning profile it
    would never see, and a project using a single judge type was told its scores
    came from different judges.

    Args:
        rows: Test case results, read for the judge types in use.
        pooled: Per-candidate pooled records, read for whether any cost exists.
        judge_config: JudgeConfig, for the repetition count.
        profile_shown: Whether the reasoning profile table is being emitted.

    Returns:
        The metadata block, sections separated by blank lines.

    Raises:
        MetadataError: If a needed section is missing or malformed.
    """
    metadata = get_app_config().verdict_metadata
    parts = [
        _section(metadata, "header"),
        _section(metadata, "figures", repetitions=judge_config.repetitions),
        _section(metadata, "score_scales"),
    ]

    if any(perf.reasoning_cost_per_point >= 0 for perf in pooled.values()):
        parts.append(_section(metadata, "cost_per_point"))

    if profile_shown:
        parts.append(_section(metadata, "reasoning_profile"))

    judge_types = {row.judge_type for row in rows}
    legend = [_section(metadata, "judge_types_intro")]
    for judge_type, field_name in (
        (JUDGE_TYPE_LLM, "judge_llm"),
        (JUDGE_TYPE_SIMILARITY, "judge_similarity"),
        (JUDGE_TYPE_ASSERT, "judge_assert"),
    ):
        if judge_type in judge_types:
            legend.append(_section(metadata, field_name))
    if len(judge_types) > 1:
        legend.append(_section(metadata, "judge_types_mixed"))
    parts.append("\n".join(legend))

    parts.append(_build_global_criteria_legend(metadata, judge_config.global_criteria))

    return "\n\n".join(parts) + "\n"


_GLOBAL_CRITERIA_FIELDS = {
    JUDGE_TYPE_LLM: ("global_criteria_llm", "llm_judge_criteria"),
    JUDGE_TYPE_SIMILARITY: ("global_criteria_similarity", "similarity_criteria"),
    JUDGE_TYPE_ASSERT: ("global_criteria_assert", "assert_criteria"),
}


def _build_global_criteria_legend(metadata, gc: GlobalCriteria) -> str:
    """Render how the global score was produced, plus the criteria text itself.

    Styled after the per-test judge_type legend just above it (judge_types_intro
    / judge_llm / ...): a code-owned "how this figure is produced" line from
    config.json, immediately followed by the project's own rubric text via
    {criteria_text}. It is the last thing _build_metadata emits, so it lands
    right before OVERALL AGGREGATE rather than in the verdict_template's own
    preamble ahead of the whole payload — which is what let it read as an
    instruction that outranks the data, when it is exactly as much "the rubric"
    as the per-test-case EVALUATION CRITERIA already is.

    Args:
        metadata: VerdictMetadata carrying the section texts.
        gc: The project's resolved GlobalCriteria.

    Returns:
        The intro line plus the mode-specific body.

    Raises:
        MetadataError: If the needed section is missing or malformed.
    """
    if gc.mode not in _GLOBAL_CRITERIA_FIELDS:
        body = _section(metadata, "global_criteria_none")
    else:
        field_name, criteria_attr = _GLOBAL_CRITERIA_FIELDS[gc.mode]
        criteria_text = getattr(gc, criteria_attr).strip() or "(none set)"
        # Every global_criteria_* section puts "  " ahead of {criteria_text}, so a
        # multi-line criteria (e.g. a numbered list) needs its own continuation
        # lines re-indented to match — otherwise only the first line lines up.
        criteria_text = criteria_text.replace("\n", "\n  ")
        body = _section(metadata, field_name, criteria_text=criteria_text)
    return _section(metadata, "global_criteria_intro") + "\n" + body


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a padded Markdown table.

    Columns are padded to a common width purely so the payload stays readable
    when a human opens verdict_prompt_debug.txt; the judge reads it either way.

    Args:
        headers: Column headings.
        rows: One list of pre-formatted cells per row, same length as headers.

    Returns:
        The table as text, ending in a newline.
    """
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines += [
        "| " + " | ".join(c.rjust(w) for c, w in zip(row, widths)) + " |" for row in rows
    ]
    return "\n".join(lines) + "\n"


def _metric_cell(
    agg: dict,
    key: str,
    suffix: str = "",
    scale: float = 1.0,
    decimals: int = 2,
) -> str:
    """Format one aggregated figure as a table cell, mean only.

    The pooled standard deviation is left out on purpose: across test cases it
    is mostly between-task variance, not the run-to-run stability a reader
    assumes a "±" describes. The per-test-case sections still carry both.

    Args:
        agg: Aggregated reasoning stats.
        key: Base field name, without the avg_ prefix.
        suffix: Unit to append, such as "%".
        scale: Multiplier applied before formatting.
        decimals: Digits after the decimal point.

    Returns:
        The formatted value, or "n/a" when it was not measured.
    """
    mean = agg.get(f"avg_{key}", -1)
    if mean is None or mean < 0:
        return "n/a"
    return f"{mean * scale:.{decimals}f}{suffix}"


def _format_cost_cell(perf: CandidatePerformance) -> str:
    """Format reasoning cost per point, or "n/a" when it was not measurable.

    Args:
        perf: Pooled performance record.

    Returns:
        The rounded ratio, or "n/a".
    """
    cost = perf.reasoning_cost_per_point
    return "n/a" if cost < 0 else f"{cost:.0f}"


def _format_cost_suffix(perf: CandidatePerformance) -> str:
    """Format the per-test-case mean-of-ratios cost, or "" when not measurable.

    Needs only reasoning_tokens and scores, both always recorded regardless of
    whether reasoning_analysis is enabled, so unlike _format_cost_cell's
    sibling this never depends on the reasoning-analysis phase having run. An
    empty string rather than "not measured" because the line it appends to has
    never needed a placeholder for an absent figure.

    Args:
        perf: The row's own performance record (repetitions of one test case),
            or a pooled one; the statistic is defined the same way for both.

    Returns:
        " | Cost: X.X ± Y.Y/point" or "".
    """
    mean = perf.mean_cost_per_point
    if mean < 0:
        return ""
    return f" | Cost: {mean:.1f} ± {perf.std_cost_per_point:.1f}/point"


def _source_cell(analyses: list[dict]) -> str:
    """Say whether the traces are raw chains of thought, summaries, or unknown.

    "raw" is a positive claim, so it must not be the fallback for a missing
    flag: traces recorded before the provider reported it carry None, and
    labelling those raw is exactly the mistake that leads a judge to compare a
    Google summary against a real transcript on length and composition.

    Args:
        analyses: Per-repetition analysis dicts for one candidate.

    Returns:
        "summary", "raw", "unknown", or "mixed".
    """
    flags = {a.get("reasoning_is_summary") for a in analyses}
    if flags == {None}:
        return "unknown"
    if len(flags - {None}) > 1:
        return "mixed"
    return "summary" if True in flags else "raw"


def _build_overall_aggregate(
    pooled: dict[str, CandidatePerformance],
    judge_config: JudgeConfig,
) -> tuple[str, bool]:
    """Render the pooled per-candidate view as one or two tables.

    The reasoning table is emitted only when there is a profile to show, so a
    project with the analysis switched off gets no empty columns to explain
    away. Before this existed the judge saw no reasoning figures at all in the
    pooled view, only fragmented per-test-case ones.

    Args:
        pooled: Per-candidate pooled records, from models.pool_by_candidate.
        judge_config: JudgeConfig, for the reasoning scope.

    Returns:
        The section text and whether the reasoning profile table was emitted,
        which decides whether the metadata explains how to read one.
    """
    parts = ["\n# OVERALL AGGREGATE (pooled across all test cases and repetitions)\n\n"]
    parts.append(_render_table(
        ["Candidate", "Task", "Global", "Time", "Out tok", "Think tok", "Think/point"],
        [
            [
                name,
                "n/a" if perf.score_mean < 0 else f"{perf.score_mean:.2f}",
                _format_global_score(perf, with_std=False),
                f"{perf.time_mean:.2f}s",
                f"{perf.tokens_mean:.0f}",
                f"{perf.reasoning_tokens_mean:.0f}",
                _format_cost_cell(perf),
            ]
            for name, perf in pooled.items()
        ],
    ))

    if not judge_config.reasoning_enabled:
        return "".join(parts), False

    profiles = {
        name: (aggregate_reasoning_stats(perf.reasoning_analyses), perf.reasoning_analyses)
        for name, perf in pooled.items()
        if perf.reasoning_analyses
    }
    if not profiles:
        return "".join(parts), False

    measured_on = (
        "the highest-scoring repetition of each test case"
        if judge_config.reasoning_analysis == REASONING_SCOPE_BEST
        else "every repetition"
    )
    parts.append(
        f"\n## REASONING PROFILE (measured on {measured_on}; coverages are independent "
        "axes, NOT shares of a whole, so they need not sum to 100%)\n\n"
    )
    parts.append(_render_table(
        ["Candidate"] + list(REASONING_DIMENSIONS)
        + ["density", "alt", "corr", "align", "drift", "repet", "n", "source"],
        [
            [name]
            + [_metric_cell(agg, f"coverage_{d}", "%", 100, 0) for d in REASONING_DIMENSIONS]
            + [
                _metric_cell(agg, "density"),
                _metric_cell(agg, "alt_path", decimals=1),
                _metric_cell(agg, "autocorrect", decimals=1),
                _metric_cell(agg, "alignment_score", decimals=1),
                _metric_cell(agg, "trace_response_drift"),
                _metric_cell(agg, "repetition_rate", "%", 100, 0),
                str(agg.get("n", 0)),
                _source_cell(analyses),
            ]
            for name, (agg, analyses) in profiles.items()
        ],
    ))
    parts.append(
        "Legend: alt = alternative approaches explored, corr = self-corrections, "
        "align = how faithfully the response followed the trace (1-10), "
        "drift = trace/response embedding similarity, repet = repeated-trigram share, "
        "n = traces analysed, source = raw chain of thought, provider summary, or "
        "unknown when the provider did not report it. "
        "A summarised trace is not the model's own wording: never rank it against a raw "
        "one on length, composition or self-correction counts.\n"
    )

    stamps = sorted({
        stamp for agg, _ in profiles.values() for stamp in agg.get("schema_stamps", [])
    })
    if len(stamps) > 1:
        parts.append(
            "WARNING: these profiles were produced by DIFFERENT analysis schemas "
            f"({', '.join(stamps)}), so their coverages are not comparable with "
            "each other. Re-run `prompttestenv analyze --force-reanalyze` to align them.\n"
        )

    return "".join(parts), True


def _build_summary_data(
    rows: list[TestCaseResult],
    candidates: list[Candidate],
    judge_config: JudgeConfig,
) -> str:
    """Build the self-describing results payload for a verdict prompt.

    The payload carries its own metadata header (repetition count, metric
    semantics, judge-type legend) so that whoever authors a verdict template
    only needs to interpolate {summary_data} to get data the judge can read
    without knowing anything about this codebase.

    Args:
        rows: Test case results to summarize (one group's rows, or all rows).
        candidates: Resolved candidate configurations.
        judge_config: JudgeConfig instance (repetition count and the
            reasoning_analysis scope).

    Returns:
        Formatted summary text ready to interpolate into a verdict template.
    """
    pooled = pool_by_candidate(candidates, rows)
    aggregate, profile_shown = _build_overall_aggregate(pooled, judge_config)
    parts = [_build_metadata(rows, pooled, judge_config, profile_shown), aggregate]

    parts.append("\n# TEST RESULTS\n")
    for row in rows:
        parts.append(
            f"\n## TEST ID: {row.test_id}\n"
            f"JUDGE TYPE: {row.judge_type}\n"
            f"ATTACHMENT: {row.file_used or 'none'}\n"
            f"PROMPT:\n{row.prompt}\n"
            f"EVALUATION CRITERIA (rubric, NOT shown to the candidate):\n{row.criteria}\n"
        )
        for cand in candidates:
            perf = row.candidates_perf.get(cand.name)
            parts.append(f"\n  > CANDIDATE: {cand.name}\n")
            if not perf:
                parts.append("    N/A (no completed repetitions)\n")
                continue
            parts.append(
                f"    Task Score: {_format_task_score(perf, with_std=True)} | "
                f"Global Score: {_format_global_score(perf, with_std=True)} | "
                f"Time: {perf.time_mean:.2f}s ± {perf.time_std:.2f}s\n"
                f"    Output Tokens: {perf.tokens_mean:.0f} ± {perf.tokens_std:.0f} | "
                f"Reasoning Tokens: {perf.reasoning_tokens_mean:.0f} ± {perf.reasoning_tokens_std:.0f}"
                f"{_format_cost_suffix(perf)}\n"
            )
            if judge_config.reasoning_enabled and perf.reasoning_analyses:
                parts.append(
                    _format_reasoning_profile(
                        perf.reasoning_analyses, judge_config.reasoning_analysis
                    )
                )
            # Notes go last: best_reason is free-form judge prose that may span
            # several lines, so indenting its continuations keeps them visibly
            # subordinate and leaves the blank line before the next candidate as
            # the one unambiguous record boundary.
            notes = perf.best_reason.replace("\n", "\n      ")
            parts.append(f"    Notes: {notes}\n")

    # No trailing blank line: the surrounding verdict template owns the spacing
    # between {summary_data} and whatever follows it.
    return "".join(parts).rstrip()


def generate_verdict(
    candidates: list[Candidate],
    results: list[TestCaseResult],
    project_dir: str,
    judge_config: JudgeConfig,
) -> str | None:
    """Generate a final comparative verdict across all candidates.

    Args:
        candidates: Resolved candidate configurations.
        results: Completed test case results with scores.
        project_dir: Benchmark project directory path.
        judge_config: JudgeConfig instance.

    Returns:
        Verdict text (Markdown, or JSON if grouped) from the verdict judge, or
        None if it could not be produced. None must never be written to the
        progress log: a persisted failure is resumed forever instead of retried.
    """
    logger.log_separator()
    logger.log_ai("Generating final verdict...")

    v_provider = judge_config.verdict_judge.provider
    v_model = judge_config.verdict_judge.model
    v_judge_temp = judge_config.verdict_judge.temperature
    thinking = judge_config.verdict_judge.thinking
    sys_prompt = judge_config.verdict_judge.verdict_system_prompt
    disable_safety = judge_config.verdict_judge.disable_safety

    try:
        verdict_template = judge_config.verdict_judge.verdict_template
        if not verdict_template:
            logger.log_error("No verdict template provided in judge_config.json.")
            return None
            
        if judge_config.group_verdicts:
            groups = {}
            for row in results:
                g = row.group or DEFAULT_GROUP
                groups.setdefault(g, []).append(row)
                
            group_verdicts_list = []
            
            for group_name, group_results in groups.items():
                logger.log_ai(f"Generating verdict for group: {group_name}...")
                summary_data = _build_summary_data(group_results, candidates, judge_config)

                prompt = verdict_template.format(summary_data=summary_data)

                save_verdict_debug_file(project_dir, sys_prompt, prompt, v_provider, v_model, v_judge_temp)

                result = get_llm_response(
                    provider=v_provider,
                    model_name=v_model,
                    system_instruction=sys_prompt,
                    user_prompt=prompt,
                    temp=v_judge_temp,
                    thinking=thinking,
                    disable_safety=disable_safety,
                )
                response_text = _strip_code_fence(result.text.strip())

                group_verdicts_list.append({"group_name": group_name, "verdict": response_text})
                
            logger.log_ai("Generating global verdict...")
            global_verdict_template = judge_config.verdict_judge.global_verdict_template
            if not global_verdict_template:
                logger.log_error(
                    "No global verdict template provided in judge_config.json."
                )
                return None
                
            group_verdicts_data = ""
            for g in group_verdicts_list:
                group_verdicts_data += f"## GROUP: {g['group_name']}\n{g['verdict']}\n\n"
                
            global_prompt = global_verdict_template.format(
                group_verdicts_data=group_verdicts_data
            )
            
            save_verdict_debug_file(project_dir, sys_prompt, global_prompt, v_provider, v_model, v_judge_temp)
            
            global_result = get_llm_response(
                provider=v_provider,
                model_name=v_model,
                system_instruction=sys_prompt,
                user_prompt=global_prompt,
                temp=v_judge_temp,
                thinking=thinking,
                disable_safety=disable_safety,
            )
            global_response_text = _strip_code_fence(global_result.text.strip())

            final_json = {
                "is_grouped": True,
                "groups": group_verdicts_list,
                "global_verdict": global_response_text
            }
            return json.dumps(final_json)

        else:
            summary_data = _build_summary_data(results, candidates, judge_config)

            prompt = verdict_template.format(summary_data=summary_data)
            save_verdict_debug_file(project_dir, sys_prompt, prompt, v_provider, v_model, v_judge_temp)

            result = get_llm_response(
                provider=v_provider,
                model_name=v_model,
                system_instruction=sys_prompt,
                user_prompt=prompt,
                temp=v_judge_temp,
                thinking=thinking,
                disable_safety=disable_safety,
            )
            response_text = _strip_code_fence(result.text.strip())
            return response_text
    except Exception as exc:
        logger.log_error(f"Verdict generation failed: {exc}")
        return None


def evaluate_best_candidate_fast(
    candidates: list[Candidate],
    results: list[TestCaseResult],
) -> str:
    """Determine the winner by average task score without generating a full verdict.

    Args:
        candidates: Resolved candidate configurations.
        results: Completed test case results.

    Returns:
        String naming the winning candidate and their average score.
    """
    stats = {
        name: agg.score_mean
        for name, agg in pool_by_candidate(candidates, results).items()
    }
    best_cand = max(stats, key=lambda k: stats[k])
    return f"Winner (by Average Task Score): {best_cand} with a score of {stats[best_cand]:.2f}/10"
