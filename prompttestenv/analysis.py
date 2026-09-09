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
from prompttestenv.progress import ProgressState, RunKey, append_event
from prompttestenv.reasoning import analyze_reasoning


def _best_key(
    keys: list[RunKey],
    eval_events: dict[RunKey, dict],
) -> RunKey:
    """Pick the highest-scoring repetition out of one candidate x test group.

    Sole definition of "best" in the codebase. Three different questions get
    asked of it, each over its own key set: keys_to_analyze asks it over
    traced keys only, to decide what is worth spending a judge call on;
    attach_best asks it over every repetition, to decide which response the
    report shows; attach_reasoning asks it twice, once unrestricted to find
    the same winner attach_best found and once over traced keys, so it can
    tell whether the winner it needs to show a trace for actually has one.
    The function itself never changes: only the key set each caller narrows
    it to differs, and that is where the three used to disagree before they
    all routed through here.

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


def attach_best(
    results: list[TestCaseResult],
    eval_events: dict[RunKey, dict],
    gen_events: dict[RunKey, dict],
) -> None:
    """Populate best_output/best_reason/best_global_reason in place.

    The single computation of "which repetition wins" that touches the
    response: generation.py and evaluator.py used to each track it
    incrementally, with their own strict `>` comparison, and runner.py's
    render_from_progress carried a third, textually duplicated copy. All
    three are gone; this is the only one left, applied in one pass after
    every event for the run is loaded rather than chased one event at a time.

    Runs over every repetition gen_events holds for a (candidate, test) pair,
    never a subset: restricting the candidate set is exactly the bug this
    replaces (see attach_reasoning, which restricts on purpose for a
    different reason and has to cross-check its own pick against this one).

    A repetition whose evaluation failed scores -1 like any other missing
    score, so when every repetition in a group failed, best_output still
    becomes that group's actual (deterministically earliest-rep) response
    text rather than staying blank: the -1/"not measured" score already
    rendered beside it says the evaluation failed, so the text is never
    mistaken for a real result, and there is no second, silent way for a
    failure to go undisplayed.

    Args:
        results: Test case results to populate.
        eval_events: Eval events keyed by (candidate, test_id, repetition).
        gen_events: Generation events keyed the same way.
    """
    for row in results:
        for cand_id, perf in row.candidates_perf.items():
            keys = [k for k in gen_events if k[0] == cand_id and k[1] == row.test_id]
            if not keys:
                continue
            best_key = _best_key(keys, eval_events)
            gen_event = gen_events[best_key]
            eval_event = eval_events.get(best_key, {})
            perf.best_output = gen_event.get("output", "")
            perf.best_reason = eval_event.get("reason", "")
            perf.best_global_reason = eval_event.get("g_reason", "")


def keys_to_analyze(
    progress_state: ProgressState,
    scope: str,
    force_reanalyze: bool,
    force_keys: frozenset[RunKey] = frozenset(),
) -> list[RunKey]:
    """List the traces this run should spend judge calls on.

    Args:
        progress_state: ProgressState snapshot holding gen, eval and reasoning events.
        scope: REASONING_SCOPE_BEST or REASONING_SCOPE_ALL.
        force_reanalyze: If True, include traces that already have an analysis.
        force_keys: Keys whose trace was regenerated in this run, so any
            analysis already stored for them describes text that no longer
            exists (its unit offsets would resolve into the wrong string).
            Belt and braces: a regenerated key is a generation placeholder,
            and those carry an empty trace, so today no analysis can exist for
            one. This drops the dependency on that invariant, which lives in
            generation.py rather than here.

    Returns:
        Keys into progress_state.gen_events, in a stable order.
    """
    with_trace = [
        key
        for key, event in progress_state.gen_events.items()
        if (event.get("reasoning_text") or "").strip()
    ]

    if scope == REASONING_SCOPE_BEST:
        by_group: dict[tuple[str, str], list[RunKey]] = {}
        for key in with_trace:
            by_group.setdefault((key[0], key[1]), []).append(key)
        with_trace = [
            _best_key(group, progress_state.eval_events) for group in by_group.values()
        ]

    return [
        key
        for key in sorted(with_trace)
        if force_reanalyze
        or key in force_keys
        or key not in progress_state.reasoning_events
    ]


def run_analysis_phase(
    results: list[TestCaseResult],
    judge_config: JudgeConfig,
    project_dir: str,
    progress_state: ProgressState,
    force_reanalyze: bool = False,
    force_keys: frozenset[RunKey] = frozenset(),
) -> dict[RunKey, dict]:
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
        force_keys: Keys whose trace was regenerated in this run, recomputed
            even when an analysis exists. See keys_to_analyze.

    Returns:
        Mapping of (candidate, test_id, repetition) to reasoning event dicts,
        covering both pre-existing and newly computed analyses.
    """
    logger.log_separator()
    logger.log_info("PHASE 3: Reasoning Analysis")

    tests_by_id = {row.test_id: row for row in results}
    analyzed: dict[RunKey, dict] = dict(progress_state.reasoning_events)

    scope = judge_config.reasoning_analysis
    pending = [
        (key, progress_state.gen_events[key])
        for key in keys_to_analyze(progress_state, scope, force_reanalyze, force_keys)
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
    reasoning_events: dict[RunKey, dict],
    eval_events: dict[RunKey, dict],
    gen_events: dict[RunKey, dict],
) -> None:
    """Populate the reasoning fields of every CandidatePerformance in place.

    The per-repetition analyses feed the aggregates, while ``best_reasoning_*``
    captures the repetition the report already shows the response for, so the
    trace on screen belongs to the response next to it — never a different,
    lower-scoring repetition's trace borrowed because it happened to have one.

    Call attach_best first: it decides which repetition wins over the full,
    unrestricted set of what was generated, and this function must agree with
    that same winner rather than re-deriving its own answer over a narrower
    set. The two used to disagree, silently, whenever the true winner had no
    trace at all.

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

            traced_keys = sorted(
                (k for k in reasoning_events if k[0] == cand_id and k[1] == row.test_id),
                key=lambda k: k[2],
            )
            if not traced_keys:
                continue
            perf.reasoning_analyses = [reasoning_events[k] for k in traced_keys]

            # The winner over EVERY repetition, traced or not — the same set
            # attach_best resolves its own pick from. Restricting to
            # traced_keys here (as this used to do) picks a different,
            # lower-scoring repetition whenever the actual winner has no
            # trace, and shows its analysis next to a response it does not
            # belong to.
            all_keys = [k for k in gen_events if k[0] == cand_id and k[1] == row.test_id]
            overall_best_key = _best_key(all_keys, eval_events) if all_keys else None
            if overall_best_key not in reasoning_events:
                # The winning repetition has no trace. No trace shown beats a
                # trace shown for the wrong repetition.
                continue

            perf.best_reasoning_analysis = ReasoningStats.from_dict(reasoning_events[overall_best_key])
            perf.best_reasoning_text = gen_events.get(overall_best_key, {}).get("reasoning_text", "")
