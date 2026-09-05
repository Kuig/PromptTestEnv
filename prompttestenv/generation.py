from __future__ import annotations

import time

import prompttestenv.logger as logger
from prompttestenv.api import call_with_timeout, get_llm_response, warm_up_for_run
from prompttestenv.config import get_app_config
from prompttestenv.progress import append_event
from prompttestenv.models import (
    Candidate, TestCaseResult, CandidatePerformance, JudgeConfig, LlmResult, ProgressState,
)



def _pending_tests(
    cand_id: str,
    results: list[TestCaseResult],
    repetitions: int,
    progress_state: ProgressState,
) -> list[TestCaseResult]:
    """The test cases this candidate still has generation work left for.

    A resumed run must not pay to warm up a candidate whose every repetition is
    already in the log, and must not upload the attachments of a test case it
    will never call again.

    Args:
        cand_id: Candidate name.
        results: All test case results, in declaration order.
        repetitions: Repetitions configured for this run.
        progress_state: Resume state built from progress.jsonl.

    Returns:
        The results holding at least one (cand_id, test_id, rep) key that is
        not yet completed, in declaration order.
    """
    return [
        test_result
        for test_result in results
        if any(
            (cand_id, test_result.test_id, rep) not in progress_state.completed_gen
            for rep in range(repetitions)
        )
    ]


def _warm_up_candidate(
    cand: Candidate,
    results: list[TestCaseResult],
    repetitions: int,
    progress_state: ProgressState,
) -> None:
    """Pay this candidate's one-off provider costs before the clock starts.

    Runs outside the timed block so the SDK import, the client construction,
    the TLS handshake and every attachment upload are not charged to whichever
    candidate happens to go first. The upload cache lives on the provider
    instance shared by every candidate, so handing each one its full list of
    attachments is idempotent: the first candidate uploads, the rest get the
    cached reference without a round-trip.

    Args:
        cand: The candidate about to be timed.
        results: All test case results, in declaration order.
        repetitions: Repetitions configured for this run.
        progress_state: Resume state built from progress.jsonl.
    """
    if not get_app_config().warmup.enabled:
        return

    pending = _pending_tests(cand.name, results, repetitions, progress_state)
    if not pending:
        return

    # dict.fromkeys: de-duplicated, declaration order preserved. A file shared
    # by several test cases is uploaded once anyway, but sending it once is
    # also what the log should show.
    media = list(dict.fromkeys(
        path for test_result in pending for path in test_result.media_file_paths
    ))

    t_start = time.time()
    if warm_up_for_run(cand.provider, cand.model, media or None):
        detail = f", {len(media)} attachment(s)" if media else ""
        logger.log_metric(f"Warmed up in {time.time() - t_start:.2f}s{detail}.")


def run_generation_phase(
    candidates: list[Candidate],
    results: list[TestCaseResult],
    judge_config: JudgeConfig,
    project_dir: str,
    progress_state: ProgressState,
) -> list[dict]:
    """Execute Phase 1: generate LLM responses for all candidates x test cases.

    Args:
        candidates: Resolved candidate configurations.
        results: Initialized test case result objects.
        judge_config: JudgeConfig instance.
        project_dir: Benchmark project directory path.
        progress_state: Current progress state (for resume support).

    Returns:
        List of pending evaluation task dicts.
    """
    logger.log_separator()
    logger.log_info("PHASE 1: Response Generation")

    for test_result in results:
        for cand in candidates:
            if cand.name not in test_result.candidates_perf:
                test_result.candidates_perf[cand.name] = CandidatePerformance()

    pending_evals = []

    repetitions = judge_config.repetitions
    timeout_val = judge_config.max_response_timeout_seconds
    rep_delay = judge_config.repetition_delay_seconds

    for cand in candidates:
        cand_id = cand.name
        provider = cand.provider
        logger.log_ai(
            f"Generation with: {cand_id} [{cand.model}, {provider}, T={cand.temperature}]"
        )

        _warm_up_candidate(cand, results, repetitions, progress_state)

        sys_instr = cand.resolved_system_instruction
        thinking = cand.thinking

        for test_result in results:
            logger.log_action(f"Test: {test_result.test_id}...")
            cand_perf = test_result.candidates_perf[cand_id]

            for rep in range(repetitions):
                prefix = f"[Rep {rep + 1}/{repetitions}] " if repetitions > 1 else ""

                # Resume logic
                key = (cand_id, test_result.test_id, rep)
                if key in progress_state.completed_gen:
                    event = progress_state.gen_events[key]
                    cand_perf.tokens.append(event["tokens"])
                    cand_perf.reasoning_tokens.append(event["reasoning_tokens"])
                    cand_perf.times.append(event["elapsed"])
                    pending_evals.append({
                        "cand_id": cand_id,
                        "cand_model": cand.model,
                        "test_result": test_result,
                        "cand_perf": cand_perf,
                        "output": event["output"],
                        "reasoning_text": event.get("reasoning_text", ""),
                        "reasoning_is_summary": event.get("reasoning_is_summary", False),
                        "rep": rep,
                        "repetitions": repetitions,
                        "elapsed": event["elapsed"],
                    })
                    logger.log_info(f"{prefix}Resumed from log.")
                    continue

                t_start = time.time()
                result, timed_out = call_with_timeout(
                    get_llm_response,
                    fn_kwargs=dict(
                        provider=provider,
                        model_name=cand.model,
                        system_instruction=sys_instr,
                        user_prompt=test_result.prompt,
                        local_media_paths=test_result.media_file_paths or None,
                        temp=cand.temperature,
                        thinking=thinking,
                        disable_safety=cand.disable_safety,
                        max_response_timeout_seconds=timeout_val,
                    ),
                    timeout=timeout_val,
                    provider=provider,
                    model=cand.model,
                )
                if timed_out:
                    result = LlmResult(text="⛔ [TIMEOUT EXCEEDED]")
                    elapsed = timeout_val
                    logger.log_warning(f"{prefix}Timeout hit ({timeout_val}s).")
                else:
                    elapsed = time.time() - t_start
                output = result.text
                tokens = result.output_tokens
                reasoning_tokens = result.reasoning_tokens

                logger.log_metric(f"{prefix}Generated ({elapsed:.2f}s) — tokens: {tokens}")
                cand_perf.tokens.append(tokens)
                cand_perf.reasoning_tokens.append(reasoning_tokens)
                cand_perf.times.append(elapsed)

                pending_evals.append({
                    "cand_id": cand_id,
                    "cand_model": cand.model,
                    "test_result": test_result,
                    "cand_perf": cand_perf,
                    "output": output,
                    "reasoning_text": result.reasoning_text,
                    "reasoning_is_summary": result.reasoning_is_summary,
                    "rep": rep,
                    "repetitions": repetitions,
                    "elapsed": elapsed,
                })

                append_event(project_dir, {
                    "type": "gen",
                    "cand_id": cand_id,
                    "test_id": test_result.test_id,
                    "rep": rep,
                    "output": output,
                    "tokens": tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "reasoning_text": result.reasoning_text,
                    "reasoning_is_summary": result.reasoning_is_summary,
                    "elapsed": elapsed,
                })

                if rep_delay > 0 and rep < repetitions - 1:
                    time.sleep(rep_delay)

    return pending_evals
