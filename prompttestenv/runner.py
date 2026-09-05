from __future__ import annotations

import os
import datetime

import prompttestenv.logger as logger
from prompttestenv.analysis import attach_reasoning, keys_to_analyze, run_analysis_phase
from prompttestenv.api import teardown
from prompttestenv.evaluator import run_evaluation_phase
from prompttestenv.verdict import generate_verdict, evaluate_best_candidate_fast, parse_grouped_verdict
from prompttestenv.generation import run_generation_phase
from prompttestenv.reporting import generate_html_report
from prompttestenv.json_report import generate_json_report
from prompttestenv.models import TestCaseResult, CandidatePerformance, Candidate, TestCase, JudgeConfig, GlobalCriteria, ProgressState
from prompttestenv.progress import append_event, failed_eval_keys, failed_gen_keys


def _initialize_test_results(test_cases: list[TestCase], project_dir: str) -> list[TestCaseResult]:
    """Create TestCaseResult objects and attach media file paths.

    For each test case that references one or more files, stores the local
    paths. The actual file upload (for Google) or inline encoding (for Ollama)
    is handled by get_llm_response() / UnifiedAiClient at call time.

    Args:
        test_cases: List of TestCase instances.
        project_dir: Path to the benchmark project directory.

    Returns:
        List of initialized TestCaseResult objects.
    """
    results = []
    logger.log_info("Initializing test structure...")
    for test in test_cases:
        test_result = TestCaseResult(
            test_id=test.id,
            prompt=test.prompt,
            criteria=test.criteria,
            group=test.group,
            judge_type=test.judge_type,
        )
        # Normalised paths, not the raw `file` value: they are what the report
        # links to and what the verdict payload names.
        attachments = test.attachments()
        test_result.files_used = attachments
        test_result.media_file_paths = [
            os.path.join(project_dir, path) for path in attachments
        ]
        results.append(test_result)
    return results


def _missing_attachments(
    test_cases: list[TestCase], project_dir: str
) -> list[tuple[str, str]]:
    """Every declared attachment that is not a file on disk.

    A typo'd path is otherwise invisible: UnifiedAiClient inlines an unreadable
    text file as the literal "[File could not be read as text]" and the
    benchmark runs on a phantom attachment, while a binary one raises halfway
    through generation, after the API calls have already been paid for.

    Args:
        test_cases: List of TestCase instances.
        project_dir: Path to the benchmark project directory.

    Returns:
        (test_id, path) pairs, in declaration order. Empty when all exist.
    """
    return [
        (test.id, path)
        for test in test_cases
        for path in test.attachments()
        if not os.path.isfile(os.path.join(project_dir, path))
    ]


def _generate_output(
    output_mode: str,
    candidates: list[Candidate],
    results: list[TestCaseResult],
    project_dir: str,
    judge_config: JudgeConfig,
    global_criteria: GlobalCriteria,
    progress_state: ProgressState,
    verdict_is_stale: bool = False,
) -> str:
    """Generate the final benchmark report.

    Args:
        output_mode: Report format — 'html', 'md', 'json' or 'winner_only'.
        candidates: Resolved candidate configurations.
        results: Completed test case results.
        project_dir: Benchmark project directory path.
        judge_config: JudgeConfig instance.
        global_criteria: GlobalCriteria instance.
        progress_state: Current progress state.
        verdict_is_stale: If True, a stored verdict describes results this run
            has since replaced, so it must be written again rather than
            resumed. Without it a --retry-errors run would repair the
            responses and then publish the verdict written about the
            placeholders. The new verdict event supersedes the old one, since
            ProgressState.load keeps the last line for a key.

    Returns:
        Path to the report or a summary string.
    """
    if output_mode == "winner_only":
        return evaluate_best_candidate_fast(candidates, results)

    if progress_state.verdict and not verdict_is_stale:
        verdict = progress_state.verdict
        logger.log_info("Verdict resumed from log.")
    else:
        verdict = generate_verdict(candidates, results, project_dir, judge_config)
        if verdict is None:
            # Deliberately not logged as a verdict event: generation and
            # evaluation are already saved, so the run is resumable and the
            # judge can be retried once the cause is fixed.
            return (
                "Error: the verdict judge produced nothing. The run is intact, so fix "
                "the cause and re-run 'prompttestenv render' to retry just the verdict."
            )
        append_event(project_dir, {"type": "verdict", "content": verdict})

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{now_str}_{len(candidates)}C_{len(results)}T"

    verdict_for_md = verdict
    grouped = parse_grouped_verdict(verdict)
    if grouped is not None:
        md_text = ""
        for g in grouped["groups"]:
            md_text += f"# VERDICT FOR GROUP: {g['group_name']}\n{g['verdict']}\n\n"
        md_text += f"# GLOBAL VERDICT\n{grouped['global_verdict']}\n"
        verdict_for_md = md_text

    if output_mode == "json":
        json_file = generate_json_report(
            project_dir, results, candidates, verdict, global_criteria, judge_config,
            filename=f"{base_filename}.json",
        )
        logger.log_save(f"JSON report saved: {json_file}")
        return f"JSON report: {json_file}"

    if output_mode == "md":
        report_dir = os.path.join(project_dir, "Report")
        os.makedirs(report_dir, exist_ok=True)
        md_file = os.path.join(report_dir, f"{base_filename}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(verdict_for_md)
        logger.log_save(f"Markdown report: {md_file}")
        return f"Markdown generated: {md_file}"

    html_file = generate_html_report(
        project_dir, results, candidates, verdict, global_criteria, judge_config,
        filename=f"{base_filename}.html",
    )
    logger.log_success(f"HTML report saved: {html_file}")
    return f"Full process complete. HTML Report: {html_file}"


def run_project(
    project_dir: str,
    output_mode: str = "html",
    force_restart: bool = False,
    retry_errors: bool = False,
) -> str:
    """Execute the benchmark on an existing project directory.

    Args:
        project_dir: Path to the project directory (must contain config files).
        output_mode: Report format — 'html', 'md', 'json' or 'winner_only'.
        force_restart: If True, deletes progress.jsonl and restarts from scratch.
        retry_errors: If True, also redo the steps whose stored result is a
            failure placeholder the framework wrote rather than a real model
            answer: a '[TIMEOUT EXCEEDED]' response, or a -1 score whose
            reasoning is a framework error. Missing steps behave exactly as in
            a normal resume. Everything else in the log is left alone, so a
            deliberate -1 from an assert lambda and the permanent 'N/A' of
            global mode 'none' are never retried.

    Returns:
        Path to the generated report or an error description string.
    """
    if not os.path.exists(project_dir):
        return f"ERROR: Project folder '{project_dir}' does not exist. Use 'init' first."

    try:
        candidates = Candidate.load_all(project_dir)
        judge_config = JudgeConfig.load(project_dir)
        test_cases = TestCase.load_all(project_dir)
    except (FileNotFoundError, ValueError) as exc:
        return f"Error: {exc}"

    if not candidates:
        return f"Error: No candidates configured in {project_dir}"

    if not test_cases:
        return f"Error: No test cases found in {project_dir}"

    # Before ProgressState.load, so a typo'd path does not even create an empty
    # progress.jsonl, and before any LLM call is made. analyze_project and
    # render_from_progress deliberately skip this: they only consume results
    # already stored, and must keep working if an attachment moved since.
    missing = _missing_attachments(test_cases, project_dir)
    if missing:
        listed = "; ".join(f"{test_id} -> {path}" for test_id, path in missing)
        return f"Error: missing attachment(s): {listed}"

    global_criteria = GlobalCriteria.load(project_dir)
    judge_config.global_criteria = global_criteria

    logger.log_separator()
    logger.log_info(
        f"Starting benchmark in [{project_dir}] "
        f"({len(test_cases)} tests x {judge_config.repetitions} runs x {len(candidates)} candidates)"
    )

    results = _initialize_test_results(test_cases, project_dir)
    progress_state = ProgressState.load(project_dir, force_restart)

    if not progress_state.hash_match:
        return (
            "ERROR: Project configuration has changed since last execution. "
            "Use --force-restart to discard past progress and restart."
        )

    # A regenerated response invalidates the score computed on the placeholder
    # it replaces, so redo_gen cascades into redo_eval. Both stay empty on a
    # plain resume, which is what keeps this a no-op for every existing caller.
    redo_gen = frozenset(failed_gen_keys(progress_state.gen_events)) if retry_errors else frozenset()
    redo_eval = (
        frozenset(failed_eval_keys(progress_state.eval_events)) | redo_gen
        if retry_errors else frozenset()
    )
    if retry_errors:
        if redo_gen or redo_eval:
            logger.log_info(
                f"Retrying {len(redo_gen)} failed generation(s) and "
                f"{len(redo_eval)} evaluation(s)."
            )
        else:
            logger.log_info("Nothing to retry: no failed step found in the log.")

    # Stale steps are repaired on EVERY run, not just under retry_errors, and
    # the difference is deliberate: a failed step is a result you may choose to
    # live with, while a stale one is a log contradicting itself, whose report
    # would show a response next to a score computed on different text. This is
    # what an interrupted retry leaves behind, and only the line order in the
    # log records it (see ProgressState).
    if progress_state.stale_eval_keys:
        logger.log_warning(
            f"{len(progress_state.stale_eval_keys)} evaluation(s) score a response that "
            "was regenerated afterwards. Judging them again."
        )
    redo_eval |= progress_state.stale_eval_keys

    try:
        pending_evals = run_generation_phase(
            candidates, results, judge_config, project_dir, progress_state, redo_gen
        )

        if not pending_evals:
            return "ERROR: No results produced."

        run_evaluation_phase(
            pending_evals, judge_config, project_dir, progress_state, redo_eval
        )

        # Re-read the log: the snapshot taken before the phases predates every
        # event they just appended, and the analysis phase reads its input
        # (the stored traces) from exactly those events.
        progress_state = ProgressState.load(project_dir, force_restart=False)

        # Reasoning analysis reads the traces the generation phase already
        # stored, so it is a separate resumable pass rather than a step nested
        # inside evaluation. `prompttestenv analyze` runs exactly this phase.
        reasoning_events = progress_state.reasoning_events
        if judge_config.reasoning_enabled:
            # progress_state was just reloaded, so stale_reasoning_keys already
            # covers anything phase 1 regenerated; redo_gen stays as the belt
            # to that braces, since it is computed before the phases run.
            reasoning_events = run_analysis_phase(
                results, judge_config, project_dir, progress_state,
                force_keys=progress_state.stale_reasoning_keys | redo_gen,
            )
        attach_reasoning(
            results, reasoning_events, progress_state.eval_events, progress_state.gen_events
        )

    finally:
        teardown()

    # stale_verdict is the durable half and would suffice on its own, since the
    # reload above already sees whatever the phases appended after the stored
    # verdict. The in-process half stays as a safety net for the day that
    # reload moves or goes away: relying on process-local state alone is
    # exactly how this bug got here.
    return _generate_output(
        output_mode, candidates, results, project_dir, judge_config, global_criteria,
        progress_state,
        verdict_is_stale=progress_state.stale_verdict or bool(redo_gen or redo_eval),
    )


def analyze_project(project_dir: str, force_reanalyze: bool = False) -> str:
    """Run only the reasoning-analysis phase on an existing progress.jsonl.

    Makes no generation and no judging calls: it reads the traces the run
    already stored. This is what makes retuning the reasoning schema cheap, and
    why the reasoning settings are excluded from the config hash.

    Args:
        project_dir: Path to the benchmark project directory.
        force_reanalyze: If True, recompute analyses that already exist.

    Returns:
        A summary string, or a descriptive error. Never raises.
    """
    if not os.path.exists(project_dir):
        return f"ERROR: Project folder '{project_dir}' does not exist. Use 'init' first."

    try:
        judge_config = JudgeConfig.load(project_dir)
        test_cases = TestCase.load_all(project_dir)
    except (FileNotFoundError, ValueError) as exc:
        return f"Error: {exc}"

    if not test_cases:
        return f"Error: No test cases found in {project_dir}"

    if not judge_config.reasoning_enabled:
        return (
            "Error: reasoning_analysis is \"none\" in judge_config.json. "
            "Set it to \"best\" (the highest-scoring repetition of each test) "
            "or \"all\" (every repetition) first."
        )

    judge_config.global_criteria = GlobalCriteria.load(project_dir)
    # Read-only: analysis consumes stored traces and does not care whether the
    # config still matches, so it must never rename the log out from under itself.
    progress_state = ProgressState.load(project_dir, readonly=True)
    if not progress_state.gen_events:
        return f"No generated responses found in {project_dir}. Run the benchmark first."

    results = _initialize_test_results(test_cases, project_dir)
    try:
        analyzed = run_analysis_phase(
            results, judge_config, project_dir, progress_state, force_reanalyze,
            force_keys=progress_state.stale_reasoning_keys,
        )
    finally:
        teardown()

    # Count against the traces this scope actually targets, not every stored
    # trace: under "best" the denominator is one repetition per test, and
    # `analyzed` may also carry analyses left over from a wider earlier scope.
    scope = judge_config.reasoning_analysis
    in_scope = keys_to_analyze(progress_state, scope, force_reanalyze=True)
    done = sum(1 for key in in_scope if key in analyzed)
    return (
        f"Reasoning analysis complete (scope '{scope}'): "
        f"{done}/{len(in_scope)} traces analyzed."
    )

def render_from_progress(project_dir: str, output_mode: str = "html") -> str:
    """Regenerate the report from an existing progress.jsonl without re-running.

    Args:
        project_dir: Path to the benchmark project directory.
        output_mode: Report format — 'html', 'md', 'json' or 'winner_only'.

    Returns:
        Report path string or partial progress summary.
    """
    progress_state = ProgressState.load(project_dir, readonly=True)
    if not progress_state.events:
        return f"No progress found in {project_dir}"

    if not progress_state.hash_match:
        logger.log_warning(
            "Current configuration does not match the progress. "
            "Best-effort render will be performed."
        )

    try:
        candidates = Candidate.load_all(project_dir)
        judge_config = JudgeConfig.load(project_dir)
        test_cases = TestCase.load_all(project_dir)
    except (FileNotFoundError, ValueError) as exc:
        return f"Error: {exc}"

    if not test_cases:
        return f"Error: No test cases found in {project_dir}"

    global_criteria = GlobalCriteria.load(project_dir)
    judge_config.global_criteria = global_criteria
    results = _initialize_test_results(test_cases, project_dir)

    results_by_id = {r.test_id: r for r in results}
    best_scores: dict = {}

    # Read the deduplicated dicts, never progress_state.events: --retry-errors
    # appends a corrected event rather than rewriting the log, so the same key
    # can appear on several lines and iterating the raw list would count every
    # superseded attempt again. The dicts keep the first insertion position and
    # the last value, so the ordering here is the one the run itself produced.
    for (cand_id, test_id, _rep), event in progress_state.gen_events.items():
        test_result = results_by_id.get(test_id)
        if test_result:
            cand_perf = test_result.candidates_perf.setdefault(cand_id, CandidatePerformance())
            cand_perf.tokens.append(event["tokens"])
            cand_perf.reasoning_tokens.append(event["reasoning_tokens"])
            cand_perf.times.append(event["elapsed"])

    for (cand_id, test_id, rep), event in progress_state.eval_events.items():
        test_result = results_by_id.get(test_id)
        if test_result:
            # setdefault here too: an eval whose gen event is missing for this
            # test would otherwise raise KeyError on a lookup that has no
            # reason to assume the two logs cover the same keys.
            cand_perf = test_result.candidates_perf.setdefault(cand_id, CandidatePerformance())
            cand_perf.scores.append(event["score"])
            cand_perf.global_scores.append(event["global_score"])
            score_key = (cand_id, test_id)
            if event["score"] > best_scores.get(score_key, -1):
                best_scores[score_key] = event["score"]
                gen_event = progress_state.gen_events.get((cand_id, test_id, rep))
                if gen_event:
                    cand_perf.best_output = gen_event["output"]
                cand_perf.best_reason = event["reason"]
                cand_perf.best_global_reason = event["g_reason"]

    attach_reasoning(
        results,
        progress_state.reasoning_events,
        progress_state.eval_events,
        progress_state.gen_events,
    )

    # Warn only, and never pass verdict_is_stale on: repairing means judge and
    # verdict calls, and render is documented everywhere as costing nothing.
    # `run` is where a stale log gets fixed.
    if progress_state.stale_eval_keys:
        logger.log_warning(
            f"{len(progress_state.stale_eval_keys)} evaluation(s) in this log score a "
            "response that was regenerated afterwards. Run the project again to rejudge them."
        )
    if progress_state.stale_verdict:
        logger.log_warning(
            "The stored verdict predates the latest results and is rendered as it is. "
            "Run the project again to have it written from what the log holds now."
        )

    if progress_state.verdict:
        return _generate_output(
            output_mode, candidates, results, project_dir, judge_config, global_criteria, progress_state
        )

    return (
        f"Partial progress: "
        f"{len(progress_state.completed_gen)} generation(s) and "
        f"{len(progress_state.completed_eval)} evaluation(s) found."
    )