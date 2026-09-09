from __future__ import annotations

import time

import prompttestenv.logger as logger
from prompttestenv.api import call_with_timeout, preload_model_for_run
from prompttestenv.progress import JUDGE_TIMEOUT_TEXT, append_event
from prompttestenv.models import (
    JUDGE_TYPE_LLM, GenerationOutcome, JudgeConfig, ProgressState, RunKey, apply_eval_event,
)
from prompttestenv.test_judge import evaluate_with_judge


def _eval_key(task: GenerationOutcome) -> RunKey:
    """The (candidate, test, repetition) key of one pending evaluation."""
    return RunKey(task.cand_id, task.test_result.test_id, task.rep)


def _test_judge_will_be_called(
    pending_evals: list[GenerationOutcome],
    judge_config: JudgeConfig,
    progress_state: ProgressState,
    redo_keys: frozenset[RunKey],
) -> bool:
    """Whether any evaluation this phase runs actually reaches the test_judge.

    Only the "llm-judge" evaluators do: "similarity" embeds through
    judge_config.similarity_judge and "assert" is a lambda over the response
    text. The resume state counts too, since a key already scored and not being
    redone makes no call at all, so a fully resumed phase preloads nothing.

    Args:
        pending_evals: The evaluations this phase was handed.
        judge_config: JudgeConfig instance.
        progress_state: Current progress state.
        redo_keys: Keys to be judged again despite being in the log.

    Returns:
        True if at least one call will go to the test_judge.
    """
    global_is_llm = getattr(judge_config.global_criteria, "mode", None) == JUDGE_TYPE_LLM
    for task in pending_evals:
        key = _eval_key(task)
        if key in progress_state.completed_eval and key not in redo_keys:
            continue
        if global_is_llm or task.test_result.judge_type == JUDGE_TYPE_LLM:
            return True
    return False


def run_evaluation_phase(
    pending_evals: list[GenerationOutcome],
    judge_config: JudgeConfig,
    project_dir: str,
    progress_state: ProgressState,
    redo_keys: frozenset[RunKey] = frozenset(),
) -> None:
    """Execute Phase 2: judge evaluation of all generated responses.

    Args:
        pending_evals: List of generation task results to evaluate.
        judge_config: JudgeConfig instance.
        project_dir: Benchmark project directory path.
        progress_state: Current progress state (for resume support).
        redo_keys: Keys to judge again even though the log already holds a
            score for them, either because that score is a framework failure
            or because the response it was computed on has been regenerated.
    """
    logger.log_separator()
    logger.log_info("PHASE 2: Judge Evaluation")

    timeout_val = judge_config.max_response_timeout_seconds
    pass_media = judge_config.pass_media_to_judge
    eval_delay = judge_config.evaluation_delay_seconds

    # Conditional: only the llm-judge evaluators reach test_judge, so a project
    # scoring everything with similarity or assert used to load a model into
    # VRAM that no call would ever use, beside the one that would.
    if _test_judge_will_be_called(pending_evals, judge_config, progress_state, redo_keys):
        preload_model_for_run(
            judge_config.test_judge.provider, judge_config.test_judge.model
        )

    current_cand = None
    current_test = None

    for eval_task in pending_evals:
        cand_id = eval_task.cand_id
        test_result = eval_task.test_result
        cand_perf = eval_task.cand_perf
        output = eval_task.output
        rep = eval_task.rep
        repetitions = eval_task.repetitions

        if current_cand != cand_id:
            logger.log_ai(f"Evaluation: {cand_id} [{eval_task.cand_model}]")
            current_cand = cand_id
            current_test = None

        if current_test != test_result.test_id:
            logger.log_action(f"Test: {test_result.test_id}...")
            current_test = test_result.test_id

        prefix = f"[Rep {rep + 1}/{repetitions}] " if repetitions > 1 else ""
        m_paths = test_result.media_file_paths if pass_media else None

        # Resume logic
        key = _eval_key(eval_task)
        if key in progress_state.completed_eval and key not in redo_keys:
            event = progress_state.eval_events[key]
            score = event["score"]
            g_score = event["global_score"]

            apply_eval_event(cand_perf, event)

            logger.log_metric(f"{prefix}Score: {score}, Global: {g_score} (Resumed)")
            continue

        eval_result, timed_out = call_with_timeout(
            evaluate_with_judge,
            test_result,
            output,
            judge_config,
            fn_kwargs={"local_media_paths": m_paths},
            timeout=timeout_val,
        )
        if timed_out:
            logger.log_warning(f"{prefix}Judge timeout ({timeout_val}s).")
            eval_result = {
                "score": -1,
                "reasoning": JUDGE_TIMEOUT_TEXT,
                "global_score": -1,
                "global_reasoning": JUDGE_TIMEOUT_TEXT,
            }

        score = eval_result.get("score", -1)
        g_score = eval_result.get("global_score", -1)
        reason = eval_result.get("reasoning", "")
        g_reason = eval_result.get("global_reasoning", "")

        logger.log_metric(f"{prefix}Score: {score}, Global: {g_score}")

        apply_eval_event(cand_perf, eval_result)

        append_event(project_dir, {
            "type": "eval",
            "cand_id": cand_id,
            "test_id": test_result.test_id,
            "rep": rep,
            "score": score,
            "global_score": g_score,
            "reason": reason,
            "g_reason": g_reason,
        })

        if eval_delay > 0 and eval_task is not pending_evals[-1]:
            time.sleep(eval_delay)
