from __future__ import annotations

import time

import prompttestenv.logger as logger
from prompttestenv.api import call_with_timeout, get_llm_response, preload_model_for_run
from prompttestenv.progress import append_event
from prompttestenv.models import Candidate, TestCaseResult, CandidatePerformance, JudgeConfig, ProgressState


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

        preload_model_for_run(provider, cand.model)

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
                        local_media_path=test_result._media_file_path,
                        temp=cand.temperature,
                        thinking=thinking,
                        disable_safety=cand.disable_safety,
                    ),
                    timeout=timeout_val,
                    provider=provider,
                    model=cand.model,
                )
                if timed_out:
                    output, tokens, reasoning_tokens, reasoning_text = "⛔ [TIMEOUT EXCEEDED]", 0, 0, ""
                    elapsed = timeout_val
                    logger.log_warning(f"{prefix}Timeout hit ({timeout_val}s).")
                else:
                    output, tokens, reasoning_tokens, reasoning_text = result
                    elapsed = time.time() - t_start

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
                    "reasoning_text": reasoning_text,
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
                    "reasoning_text": reasoning_text,
                    "elapsed": elapsed,
                })

                if rep_delay > 0 and rep < repetitions - 1:
                    time.sleep(rep_delay)

    return pending_evals
