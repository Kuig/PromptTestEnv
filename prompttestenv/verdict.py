from __future__ import annotations

import json
import os

import prompttestenv.logger as logger
from prompttestenv.api import get_llm_response
from prompttestenv.models import TestCaseResult, Candidate, JudgeConfig, calculate_stats, DEFAULT_GROUP
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


def _build_summary_data(
    rows: list[TestCaseResult],
    candidates: list[Candidate],
    judge_config: JudgeConfig,
) -> str:
    """Build the per-test-case candidate performance summary for a verdict prompt.

    Args:
        rows: Test case results to summarize (one group's rows, or all rows).
        candidates: Resolved candidate configurations.
        judge_config: JudgeConfig instance (used for the reasoning_analysis flag).

    Returns:
        Formatted summary text ready to interpolate into a verdict template.
    """
    summary_data = ""
    for row in rows:
        summary_data += f"## TEST ID: {row.test_id}\n"
        for cand in candidates:
            cand_id = cand.name
            perf = row.candidates_perf.get(cand_id)
            if perf:
                g_score_str = f"{perf.global_score_mean:.2f} ± {perf.global_score_std:.2f}/10" if perf.global_score_mean >= 0 else "N/A"
                summary_data += (
                    f"  > SYSTEM: {cand_id} | "
                    f"Task Score: {perf.score_mean:.2f} ± {perf.score_std:.2f}/10 | "
                    f"Global Score: {g_score_str} | "
                    f"Time: {perf.time_mean:.2f}s ± {perf.time_std:.2f}s | "
                    f"Tokens: {perf.tokens_mean:.0f} ± {perf.tokens_std:.0f} (Reasoning: {perf.reasoning_tokens_mean:.0f} ± {perf.reasoning_tokens_std:.0f}) | "
                    f"Notes: {perf.best_reason}\n"
                )
                if judge_config.reasoning_analysis and perf.reasoning_analyses:
                    r_agg = aggregate_reasoning_stats(perf.reasoning_analyses)
                    summary_data += (
                        f"    Cognitive Resource Analysis: On average, the candidate utilized "
                        f"{r_agg.get('avg_pure_reasoning_pct', 0):.1f}% ± {r_agg.get('std_pure_reasoning_pct', 0):.1f}% "
                        f"of their cognitive resources to solve the problem, with the remainder "
                        f"dedicated to interacting with the user. "
                        f"The model explored {r_agg.get('avg_alt_path', 0):.1f} ± {r_agg.get('std_alt_path', 0):.1f} alternatives "
                        f"and self-corrected {r_agg.get('avg_autocorrect', 0):.1f} ± {r_agg.get('std_autocorrect', 0):.1f} times. "
                        f"The alignment between the response and the reasoning process scored "
                        f"{r_agg.get('avg_alignment_score', 0):.1f} ± {r_agg.get('std_alignment_score', 0):.1f} out of 10.\n"
                    )
            else:
                summary_data += f"  > SYSTEM: {cand_id} | N/A\n"
        summary_data += "\n"
    return summary_data


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
            
        global_criteria_obj = getattr(judge_config, "global_criteria", None)
        if global_criteria_obj and hasattr(global_criteria_obj, "to_verdict_string"):
            global_criteria = global_criteria_obj.to_verdict_string()
        else:
            global_criteria = "None"
            
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
    stats: dict[str, float] = {}
    for cand in candidates:
        cand_id = cand.name
        task_scores = []
        for row in results:
            perf = row.candidates_perf.get(cand_id)
            if perf:
                task_scores.extend(perf.scores)
        t_mean, _ = calculate_stats(task_scores)
        stats[cand_id] = t_mean

    best_cand = max(stats, key=lambda k: stats[k])
    return f"Winner (by Average Task Score): {best_cand} with a score of {stats[best_cand]:.2f}/10"
