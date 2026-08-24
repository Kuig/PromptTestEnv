from __future__ import annotations

import time

import prompttestenv.logger as logger
from prompttestenv.api import call_with_timeout, preload_model_for_run
from prompttestenv.progress import append_event
from prompttestenv.models import JudgeConfig, ProgressState
from prompttestenv.test_judge import evaluate_with_judge
from prompttestenv.reasoning import analyze_reasoning


def run_evaluation_phase(
    pending_evals: list[dict],
    judge_config: JudgeConfig,
    project_dir: str,
    progress_state: ProgressState,
) -> None:
    """Execute Phase 2: judge evaluation of all generated responses.

    Args:
        pending_evals: List of generation task results to evaluate.
        judge_config: JudgeConfig instance.
        project_dir: Benchmark project directory path.
        progress_state: Current progress state (for resume support).
    """
    logger.log_separator()
    logger.log_info("PHASE 2: Judge Evaluation")

    timeout_val = judge_config.max_response_timeout_seconds
    pass_media = judge_config.pass_media_to_judge
    eval_delay = judge_config.evaluation_delay_seconds

    j_provider = judge_config.test_judge.provider
    j_model = judge_config.test_judge.model

    preload_model_for_run(j_provider, j_model)

    best_scores: dict = {}
    current_cand = None
    current_test = None

    for eval_task in pending_evals:
        cand_id = eval_task["cand_id"]
        test_result = eval_task["test_result"]
        cand_perf = eval_task["cand_perf"]
        output = eval_task["output"]
        rep = eval_task["rep"]
        repetitions = eval_task["repetitions"]

        if current_cand != cand_id:
            logger.log_ai(f"Evaluation: {cand_id} [{eval_task['cand_model']}]")
            current_cand = cand_id
            current_test = None

        if current_test != test_result.test_id:
            logger.log_action(f"Test: {test_result.test_id}...")
            current_test = test_result.test_id

        prefix = f"[Rep {rep + 1}/{repetitions}] " if repetitions > 1 else ""
        m_path = test_result._media_file_path if pass_media else None

        # Resume logic
        key = (cand_id, test_result.test_id, rep)
        if key in progress_state.completed_eval:
            event = progress_state.eval_events[key]
            score = event["score"]
            g_score = event["global_score"]
            reason = event["reason"]
            g_reason = event["g_reason"]

            cand_perf.scores.append(score)
            cand_perf.global_scores.append(g_score)

            # Restore reasoning analysis from progress if available
            r_analysis = event.get("reasoning_analysis")
            if r_analysis:
                cand_perf.reasoning_analyses.append(r_analysis)

            score_key = (cand_id, test_result.test_id)
            if score > best_scores.get(score_key, -1):
                best_scores[score_key] = score
                cand_perf.best_output = output
                cand_perf.best_reason = reason
                cand_perf.best_global_reason = g_reason

            logger.log_metric(f"{prefix}Score: {score}, Global: {g_score} (Resumed)")
            continue

        eval_result, timed_out = call_with_timeout(
            evaluate_with_judge,
            test_result,
            output,
            judge_config,
            fn_kwargs={"local_media_path": m_path},
            timeout=timeout_val,
            provider=j_provider,
            model=j_model,
        )
        if timed_out:
            logger.log_warning(f"{prefix}Judge timeout ({timeout_val}s).")
            eval_result = {
                "score": 0,
                "reasoning": "⛔ [JUDGE TIMEOUT EXCEEDED]",
                "global_score": 0,
                "global_reasoning": "⛔ [JUDGE TIMEOUT EXCEEDED]",
            }

        score = eval_result.get("score", 0)
        g_score = eval_result.get("global_score", 0)
        reason = eval_result.get("reasoning", "")
        g_reason = eval_result.get("global_reasoning", "")

        logger.log_metric(f"{prefix}Score: {score}, Global: {g_score}")

        cand_perf.scores.append(score)
        cand_perf.global_scores.append(g_score)

        # --- Reasoning analysis (optional) ---
        reasoning_analysis_dict: dict | None = None
        reasoning_text = eval_task.get("reasoning_text", "")
        if judge_config.reasoning_analysis and reasoning_text:
            logger.log_action(f"{prefix}Analyzing reasoning trace...")
            r_stats = analyze_reasoning(reasoning_text, judge_config, candidate_response=output)
            if r_stats is not None:
                reasoning_analysis_dict = r_stats.to_dict()
                cand_perf.reasoning_analyses.append(reasoning_analysis_dict)
                logger.log_metric(
                    f"{prefix}Reasoning: {r_stats.pure_reasoning_pct:.1f}% pure | "
                    f"align={r_stats.alignment_score}/10"
                )

        score_key = (cand_id, test_result.test_id)
        if score > best_scores.get(score_key, -1):
            best_scores[score_key] = score
            cand_perf.best_output = output
            cand_perf.best_reason = reason
            cand_perf.best_global_reason = g_reason

        append_event(project_dir, {
            "type": "eval",
            "cand_id": cand_id,
            "test_id": test_result.test_id,
            "rep": rep,
            "score": score,
            "global_score": g_score,
            "reason": reason,
            "g_reason": g_reason,
            "reasoning_analysis": reasoning_analysis_dict,
        })

        if eval_delay > 0 and eval_task is not pending_evals[-1]:
            time.sleep(eval_delay)
