"""Phase 3: reasoning-trace analysis over an existing progress log.

This phase reads the traces already stored in ``progress.jsonl`` by the
generation phase, so it makes no generation and no judging calls of its own and
can be re-run as often as the schema is retuned. ``calculate_config_hash``
deliberately ignores the reasoning settings for the same reason: iterating on
how traces are measured must not cost a re-run of every candidate.

Results are appended as their own ``"reasoning"`` events rather than nested
inside the ``"eval"`` records, which is what makes them independently resumable.
"""
from __future__ import annotations

import time

import prompttestenv.logger as logger
from prompttestenv.api import is_local_provider, preload_model_for_run
from prompttestenv.models import (
    REASONING_SCOPE_BEST,
    JudgeConfig,
    ReasoningStats,
    TestCaseResult,
)
from prompttestenv.progress import append_event
from prompttestenv.reasoning import analyze_reasoning


def _best_key(
    keys: list[tuple[str, str, int]],
    eval_events: dict[tuple[str, str, int], dict],
) -> tuple[str, str, int]:
    """Pick the highest-scoring repetition out of one candidate x test group.

    Sole definition of "best" in the codebase: the analysis phase uses it to
    decide what to spend calls on, and the report uses it to decide which trace
    to draw, so the two must not be able to disagree.

    A repetition whose evaluation is missing or failed scores -1 and therefore
    loses to any evaluated one; ties go to the earliest repetition, so the
    choice stays deterministic when scores are missing or identical.

    Args:
        keys: Non-empty list of (candidate, test_id, repetition) keys.
        eval_events: Eval events keyed the same way.

    Returns:
        The winning key.
    """
    return max(keys, key=lambda k: (eval_events.get(k, {}).get("score", -1), -k[2]))


def keys_to_analyze(
    progress_state,
    scope: str,
    force_reanalyze: bool,
) -> list[tuple[str, str, int]]:
    """List the traces this run should spend judge calls on.

    Args:
        progress_state: ProgressState snapshot holding gen, eval and reasoning events.
        scope: REASONING_SCOPE_BEST or REASONING_SCOPE_ALL.
        force_reanalyze: If True, include traces that already have an analysis.

    Returns:
        Keys into progress_state.gen_events, in a stable order.
    """
    with_trace = [
        key
        for key, event in progress_state.gen_events.items()
        if (event.get("reasoning_text") or "").strip()
    ]

    if scope == REASONING_SCOPE_BEST:
        by_group: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
        for key in with_trace:
            by_group.setdefault((key[0], key[1]), []).append(key)
        with_trace = [
            _best_key(group, progress_state.eval_events) for group in by_group.values()
        ]

    return [
        key
        for key in sorted(with_trace)
        if force_reanalyze or key not in progress_state.reasoning_events
    ]


def run_analysis_phase(
    results: list[TestCaseResult],
    judge_config: JudgeConfig,
    project_dir: str,
    progress_state,
    force_reanalyze: bool = False,
) -> dict[tuple[str, str, int], dict]:
    """Analyze the stored reasoning traces that have no analysis yet.

    Which traces those are depends on judge_config.reasoning_analysis: "all"
    covers every repetition, "best" only the highest-scoring one per test, which
    costs `repetitions` times less and is the repetition the report draws the
    trace for anyway. Analyses already in the log are kept and reused either
    way, so narrowing the scope never discards work already paid for, and
    widening it later fills in only what is missing.

    Args:
        results: Test case results, used to recover each test's prompt and
            criteria. The judge needs them to tell a model restating the
            request apart from one reasoning about it.
        judge_config: JudgeConfig carrying the reasoning_judge settings.
        project_dir: Benchmark project directory path.
        progress_state: ProgressState snapshot holding the gen and reasoning events.
        force_reanalyze: If True, recompute analyses that already exist.

    Returns:
        Mapping of (candidate, test_id, repetition) to reasoning event dicts,
        covering both pre-existing and newly computed analyses.
    """
    logger.log_separator()
    logger.log_info("PHASE 3: Reasoning Analysis")

    tests_by_id = {row.test_id: row for row in results}
    analyzed: dict[tuple[str, str, int], dict] = dict(progress_state.reasoning_events)

    scope = judge_config.reasoning_analysis
    pending = [
        (key, progress_state.gen_events[key])
        for key in keys_to_analyze(progress_state, scope, force_reanalyze)
    ]
    if scope == REASONING_SCOPE_BEST:
        logger.log_info(
            "Scope 'best': analyzing only the highest-scoring repetition of each test."
        )
    if not pending:
        logger.log_info("No reasoning traces to analyze.")
        return analyzed

    rj = judge_config.reasoning_judge
    if is_local_provider(rj.provider):
        logger.log_info(
            f"Reasoning judge '{rj.model}' is local: dimension calls run sequentially."
        )
        preload_model_for_run(rj.provider, rj.model, context_size=rj.context_size)

    delay = judge_config.evaluation_delay_seconds
    for position, (key, event) in enumerate(pending):
        cand_id, test_id, rep = key
        test_result = tests_by_id.get(test_id)
        if test_result is None:
            logger.log_warning(f"Test '{test_id}' is no longer configured. Skipping its analysis.")
            continue

        logger.log_action(f"Analyzing trace: {cand_id} / {test_id} [rep {rep + 1}]...")
        stats = analyze_reasoning(
            event["reasoning_text"],
            judge_config,
            candidate_response=event.get("output", ""),
            user_prompt=test_result.prompt,
            criteria=test_result.criteria,
            reasoning_is_summary=event.get("reasoning_is_summary"),
        )
        if stats is None:
            continue

        record = {
            "type": "reasoning",
            "cand_id": cand_id,
            "test_id": test_id,
            "rep": rep,
            **stats.to_dict(),
        }
        append_event(project_dir, record)
        analyzed[key] = record

        logger.log_metric(
            "  " + " | ".join(
                f"{dim} {stats.coverage(dim) * 100:.0f}%"
                for dim in ("framing", "solving", "presentation")
                if stats.coverage(dim) >= 0
            )
            + f" | density {stats.density:.2f} | align {stats.alignment_score}/10"
        )

        if delay > 0 and position < len(pending) - 1:
            time.sleep(delay)

    return analyzed


def attach_reasoning(
    results: list[TestCaseResult],
    reasoning_events: dict[tuple[str, str, int], dict],
    eval_events: dict[tuple[str, str, int], dict],
    gen_events: dict[tuple[str, str, int], dict],
) -> None:
    """Populate the reasoning fields of every CandidatePerformance in place.

    The per-repetition analyses feed the aggregates, while ``best_reasoning_*``
    captures the repetition the report already shows the response for, so the
    trace on screen belongs to the response next to it.

    Args:
        results: Test case results to populate.
        reasoning_events: Reasoning events keyed by (candidate, test_id, repetition).
        eval_events: Eval events, used to identify each candidate's best repetition.
        gen_events: Generation events, used to recover the trace text itself.
    """
    for row in results:
        for cand_id, perf in row.candidates_perf.items():
            perf.reasoning_analyses = []
            perf.best_reasoning_text = ""
            perf.best_reasoning_analysis = None

            keys = sorted(
                (k for k in reasoning_events if k[0] == cand_id and k[1] == row.test_id),
                key=lambda k: k[2],
            )
            if not keys:
                continue
            perf.reasoning_analyses = [reasoning_events[k] for k in keys]

            best_key = _best_key(keys, eval_events)
            perf.best_reasoning_analysis = ReasoningStats.from_dict(reasoning_events[best_key])
            perf.best_reasoning_text = gen_events.get(best_key, {}).get("reasoning_text", "")
