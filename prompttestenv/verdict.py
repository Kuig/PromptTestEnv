from __future__ import annotations

import json
import os

import prompttestenv.logger as logger
from prompttestenv.api import get_llm_response
from prompttestenv.models import (
    TestCaseResult,
    Candidate,
    CandidatePerformance,
    JudgeConfig,
    DEFAULT_GROUP,
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
        "=" * 72 + "\nVERDICT JUDGE DEBUG INFO\n" + "=" * 72 + "\n"
        f"PROVIDER: {provider}\nMODEL:    {model}\nTEMP:     {temp}\n"
        + "-" * 72 + "\nSYSTEM INSTRUCTION:\n" + "-" * 72 + f"\n{sys_prompt}\n"
        + "=" * 72 + "\nUSER PROMPT (PAYLOAD):\n" + "=" * 72 + f"\n{prompt}\n"
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


METADATA_TEMPLATE = """# BENCHMARK METADATA
Each candidate saw only the PROMPT, the ATTACHMENT, and its own system prompt (not shown
here, and it may differ per candidate). The EVALUATION CRITERIA are the judges' rubric,
NOT shown to the candidate: missing one is a difference in behaviour, not disobedience.

Repetitions per candidate x test case: {repetitions}. Figures are mean ± SD across them;
at 1 repetition every ± is 0.00 by construction and says nothing about stability.
'Notes' come from the best-scoring repetition, not an average one.
Token counts are OUTPUT only — reasoning tokens are separate and excluded, input tokens
are not tracked.
Both scores run 1-10, but what they measure is set by this project's criteria and may be
redefined by the instructions below; 'N/A' means not computed.

Task scores come from different judges and are NOT comparable across test cases:
- llm-judge: graded 1-10 by an LLM against the test's criteria.
- similarity: cosine similarity to a target text, rescaled to 1-10.
- assert: a user-authored Python expression; it may use the full 1-10 range or just a
  pass/fail pair, and which one is never declared.
Each test case declares its judge; the aggregate below pools all of them, so read it as an
indication, not a measurement.
"""


def _aggregate_by_candidate(
    candidates: list[Candidate],
    rows: list[TestCaseResult],
) -> dict[str, CandidatePerformance]:
    """Pool each candidate's per-test-case performance into a single record.

    Args:
        candidates: Resolved candidate configurations.
        rows: Test case results to pool over.

    Returns:
        Mapping of candidate name to a CandidatePerformance holding every
        repetition of every test case, so its mean/std properties describe the
        candidate's run as a whole. Candidates with no data map to an empty
        record (whose means fall back to the usual defaults).
    """
    pooled: dict[str, CandidatePerformance] = {}
    for cand in candidates:
        agg = CandidatePerformance()
        for row in rows:
            perf = row.candidates_perf.get(cand.name)
            if not perf:
                continue
            agg.scores.extend(perf.scores)
            agg.global_scores.extend(perf.global_scores)
            agg.times.extend(perf.times)
            agg.tokens.extend(perf.tokens)
            agg.reasoning_tokens.extend(perf.reasoning_tokens)
        pooled[cand.name] = agg
    return pooled


def _format_global_score(perf: CandidatePerformance, with_std: bool) -> str:
    """Render a global score, or "N/A" when global scoring produced nothing.

    Args:
        perf: Performance record to read the global score from.
        with_std: Whether to append the standard deviation.

    Returns:
        Formatted score string.
    """
    if perf.global_score_mean < 0:
        return "N/A"
    if with_std:
        return f"{perf.global_score_mean:.2f} ± {perf.global_score_std:.2f}/10"
    return f"{perf.global_score_mean:.2f}/10"


def _format_reasoning_profile(analyses: list[dict]) -> str:
    """Render the aggregated reasoning trace profile for one candidate.

    Args:
        analyses: Per-repetition reasoning analysis dicts.

    Returns:
        Indented multi-line block describing the trace's composition and metrics.
    """
    agg = aggregate_reasoning_stats(analyses)
    return (
        f"    Reasoning trace profile (share of the trace's characters): "
        f"interpretation {agg.get('avg_interpretation_pct', 0):.1f}% ± {agg.get('std_interpretation_pct', 0):.1f}%, "
        f"planning {agg.get('avg_planning_pct', 0):.1f}% ± {agg.get('std_planning_pct', 0):.1f}%,\n"
        f"      problem-solving {agg.get('avg_pure_reasoning_pct', 0):.1f}% ± {agg.get('std_pure_reasoning_pct', 0):.1f}%, "
        f"output formulation {agg.get('avg_output_formulation_pct', 0):.1f}% ± {agg.get('std_output_formulation_pct', 0):.1f}%.\n"
        f"      Alternatives explored: {agg.get('avg_alt_path', 0):.1f} ± {agg.get('std_alt_path', 0):.1f}. "
        f"Self-corrections: {agg.get('avg_autocorrect', 0):.1f} ± {agg.get('std_autocorrect', 0):.1f}.\n"
        f"      Response/reasoning alignment: {agg.get('avg_alignment_score', 0):.1f} ± {agg.get('std_alignment_score', 0):.1f} out of 10.\n"
    )


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
            reasoning_analysis flag).

    Returns:
        Formatted summary text ready to interpolate into a verdict template.
    """
    parts = [METADATA_TEMPLATE.format(repetitions=judge_config.repetitions)]

    parts.append("\n# OVERALL AGGREGATE (pooled across all test cases and repetitions)\n")
    for name, agg in _aggregate_by_candidate(candidates, rows).items():
        parts.append(
            f"  > CANDIDATE: {name} | "
            f"Task Score: {agg.score_mean:.2f}/10 | "
            f"Global Score: {_format_global_score(agg, with_std=False)} | "
            f"Time: {agg.time_mean:.2f}s | "
            f"Output Tokens: {agg.tokens_mean:.0f} | "
            f"Reasoning Tokens: {agg.reasoning_tokens_mean:.0f}\n"
        )

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
                f"    Task Score: {perf.score_mean:.2f} ± {perf.score_std:.2f}/10 | "
                f"Global Score: {_format_global_score(perf, with_std=True)} | "
                f"Time: {perf.time_mean:.2f}s ± {perf.time_std:.2f}s\n"
                f"    Output Tokens: {perf.tokens_mean:.0f} ± {perf.tokens_std:.0f} | "
                f"Reasoning Tokens: {perf.reasoning_tokens_mean:.0f} ± {perf.reasoning_tokens_std:.0f}\n"
            )
            if judge_config.reasoning_analysis and perf.reasoning_analyses:
                parts.append(_format_reasoning_profile(perf.reasoning_analyses))
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
) -> str:
    """Generate a final comparative verdict across all candidates.

    Args:
        candidates: Resolved candidate configurations.
        results: Completed test case results with scores.
        project_dir: Benchmark project directory path.
        judge_config: JudgeConfig instance.

    Returns:
        Verdict text (Markdown or JSON if grouped) from the verdict judge, or an error string.
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
            return "No verdict template provided."
            
        global_criteria = judge_config.global_criteria.to_verdict_string()

        if judge_config.group_verdicts:
            groups = {}
            for row in results:
                g = row.group or DEFAULT_GROUP
                groups.setdefault(g, []).append(row)
                
            group_verdicts_list = []
            
            for group_name, group_results in groups.items():
                logger.log_ai(f"Generating verdict for group: {group_name}...")
                summary_data = _build_summary_data(group_results, candidates, judge_config)

                prompt = verdict_template.format(
                    summary_data=summary_data,
                    global_criteria=global_criteria
                )
                
                save_verdict_debug_file(project_dir, sys_prompt, prompt, v_provider, v_model, v_judge_temp)
                
                response_text, _, _, _ = get_llm_response(
                    provider=v_provider,
                    model_name=v_model,
                    system_instruction=sys_prompt,
                    user_prompt=prompt,
                    temp=v_judge_temp,
                    thinking=thinking,
                    disable_safety=disable_safety,
                )
                response_text = _strip_code_fence(response_text.strip())

                group_verdicts_list.append({"group_name": group_name, "verdict": response_text})
                
            logger.log_ai("Generating global verdict...")
            global_verdict_template = judge_config.verdict_judge.global_verdict_template
            if not global_verdict_template:
                return "No global verdict template provided."
                
            group_verdicts_data = ""
            for g in group_verdicts_list:
                group_verdicts_data += f"## GROUP: {g['group_name']}\n{g['verdict']}\n\n"
                
            global_prompt = global_verdict_template.format(
                group_verdicts_data=group_verdicts_data,
                global_criteria=global_criteria
            )
            
            save_verdict_debug_file(project_dir, sys_prompt, global_prompt, v_provider, v_model, v_judge_temp)
            
            global_response_text, _, _, _ = get_llm_response(
                provider=v_provider,
                model_name=v_model,
                system_instruction=sys_prompt,
                user_prompt=global_prompt,
                temp=v_judge_temp,
                thinking=thinking,
                disable_safety=disable_safety,
            )
            global_response_text = _strip_code_fence(global_response_text.strip())

            final_json = {
                "is_grouped": True,
                "groups": group_verdicts_list,
                "global_verdict": global_response_text
            }
            return json.dumps(final_json)

        else:
            summary_data = _build_summary_data(results, candidates, judge_config)

            prompt = verdict_template.format(
                summary_data=summary_data,
                global_criteria=global_criteria
            )
            save_verdict_debug_file(project_dir, sys_prompt, prompt, v_provider, v_model, v_judge_temp)

            response_text, _, _, _ = get_llm_response(
                provider=v_provider,
                model_name=v_model,
                system_instruction=sys_prompt,
                user_prompt=prompt,
                temp=v_judge_temp,
                thinking=thinking,
                disable_safety=disable_safety,
            )
            response_text = _strip_code_fence(response_text.strip())
            return response_text
    except Exception as exc:
        return f"Error generating verdict: {str(exc)}"


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
        for name, agg in _aggregate_by_candidate(candidates, results).items()
    }
    best_cand = max(stats, key=lambda k: stats[k])
    return f"Winner (by Average Task Score): {best_cand} with a score of {stats[best_cand]:.2f}/10"
