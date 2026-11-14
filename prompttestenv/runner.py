from __future__ import annotations

import os
import datetime

import prompttestenv.logger as logger
from prompttestenv.analysis import attach_best, attach_reasoning, keys_to_analyze, run_analysis_phase
from prompttestenv.api import teardown
from prompttestenv.catalog import offline_findings
from prompttestenv.evaluator import run_evaluation_phase
from prompttestenv.frontier import FrontierPanels, GroupClusters, build_panels
from prompttestenv.verdict import generate_verdict, evaluate_best_candidate_fast
from prompttestenv.generation import run_generation_phase
from prompttestenv.reporting import generate_html_report
from prompttestenv.json_report import generate_json_report
from prompttestenv.models import TestCaseResult, CandidatePerformance, Candidate, TestCase, JudgeConfig, GlobalCriteria, OUTPUT_MODES, apply_eval_event, apply_gen_event, parse_grouped_verdict
from prompttestenv.progress import ProgressState, append_event, failed_eval_keys, failed_gen_keys
import prompttestenv.projectio as pio


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
            json_output=test.json_output,
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


def _build_frontiers(
    project_dir: str, candidates: list[Candidate], results: list[TestCaseResult],
) -> FrontierPanels:
    """Read group_clusters.json and compute every frontier panel, once.

    Done here rather than in each renderer so that the HTML, the JSON and the
    verdict payload read the very same panels, and so that a name the file
    uses but the project lacks is announced once rather than per surface.
    """
    frontiers = build_panels(candidates, results, GroupClusters.load(project_dir))
    for message in frontiers.warnings:
        logger.log_warning(message)
    return frontiers


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

    frontiers = _build_frontiers(project_dir, candidates, results)

    if progress_state.verdict and not verdict_is_stale:
        verdict = progress_state.verdict
        logger.log_info("Verdict resumed from log.")
    else:
        verdict = generate_verdict(candidates, results, project_dir, judge_config, frontiers)
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
            filename=f"{base_filename}.json", frontiers=frontiers,
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

    if output_mode == "html":
        html_file = generate_html_report(
            project_dir, results, candidates, verdict, global_criteria, judge_config,
            filename=f"{base_filename}.html", frontiers=frontiers,
        )
        logger.log_success(f"HTML report saved: {html_file}")
        return f"Full process complete. HTML Report: {html_file}"

    # Explicit rather than a fall-through: HTML used to be whatever was left,
    # so an unlisted mode was silently rendered as a full report. Unreachable
    # while the entry points gate on OUTPUT_MODES, and here so that a mode added
    # to that tuple without a branch of its own fails loudly instead.
    return f"Error: no renderer for output_mode {output_mode!r}."


def _invalid_output_mode(output_mode: str) -> str | None:
    """The error string for an unknown output mode, or None when it is valid.

    Gated at the entry points rather than in _generate_output, which runs after
    the whole benchmark has been paid for. And in the core rather than only in
    argparse, because the CLI was the only caller that checked: the library, the
    MCP tools and both GUIs passed anything straight through.

    Args:
        output_mode: The requested report format.

    Returns:
        A descriptive error string, or None when output_mode is valid.
    """
    if output_mode in OUTPUT_MODES:
        return None
    return (
        f"Error: unknown output_mode {output_mode!r}. "
        f"Valid: {', '.join(OUTPUT_MODES)}."
    )


# Wider than the two the loaders document (FileNotFoundError only), because a
# malformed JSON shape can raise ValueError/TypeError/AttributeError/KeyError
# out of any of the three regardless of which entry point called them - they
# read the same files. Explicit tuple rather than `except Exception`, so a
# real bug still surfaces.
_LOADER_EXCEPTIONS = (FileNotFoundError, ValueError, TypeError, AttributeError, KeyError)


def _load_project_config(
    project_dir: str,
    *,
    load_candidates: bool = True,
) -> tuple[list[Candidate] | None, JudgeConfig, list[TestCase]] | str:
    """Directory, loader and non-empty checks shared by the three entry points.

    Every one of run_project/analyze_project/render_from_progress needs the
    project directory to exist and the config it loads to be non-degenerate
    before doing anything else; they differ only in whether candidates.json
    matters at all (analyze_project never reads it, since reasoning analysis
    re-scores traces already stored, not candidates) and in what each does
    next about the progress log, which stays each caller's own concern below
    this helper.

    Args:
        project_dir: Path to the benchmark project directory.
        load_candidates: Whether candidates.json is loaded and checked for
            emptiness. False for analyze_project.

    Returns:
        (candidates, judge_config, test_cases) on success - candidates is
        None when load_candidates is False - or an "Error: ..."/"ERROR: ..."
        string ready to be returned verbatim, matching every entry point's
        own documented "never raise, always return a descriptive string"
        contract.
    """
    if not os.path.exists(project_dir):
        return f"ERROR: Project folder '{project_dir}' does not exist. Use 'init' first."

    try:
        candidates = Candidate.load_all(project_dir) if load_candidates else None
        judge_config = JudgeConfig.load(project_dir)
        test_cases = TestCase.load_all(project_dir)
    except _LOADER_EXCEPTIONS as exc:
        return f"Error: {exc}"

    if load_candidates and not candidates:
        return f"Error: No candidates configured in {project_dir}"

    if not test_cases:
        return f"Error: No test cases found in {project_dir}"

    return candidates, judge_config, test_cases


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
        A one-line status string, never a bare path: `Error: ...` (or
        `ERROR: ...`) on failure, otherwise a confirmation naming the report
        file, e.g. `Full process complete. HTML Report: <path>`. Use
        gui/common.report_path to recover the path from it.
    """
    # An argument error, so it precedes every check about state.
    invalid = _invalid_output_mode(output_mode)
    if invalid:
        return invalid

    if not os.path.exists(project_dir):
        return f"ERROR: Project folder '{project_dir}' does not exist. Use 'init' first."

    # Pre-flight, before the loaders and before any LLM call. The same check
    # both editing surfaces run, so a project cannot be accepted by one and
    # silently mishandled by the other: a duplicate candidate name, say, keys
    # every progress.jsonl event, so two candidates would merge into one and
    # the report would describe results that never existed.
    #
    # Ahead of the loaders because projectio.load_project is total where they
    # are strict, so a malformed file is described here rather than raising an
    # AttributeError out of a dataclass constructor.
    #
    # Errors only. The warnings are informational and MUST stay that way: they
    # include "attachment outside test_files/", which is a lint on the editor's
    # convention, not on the run - a project may legitimately keep attachments
    # anywhere, and _missing_attachments below is what actually decides whether
    # a declared file exists. The two checks look alike and are not: do not
    # merge them.
    draft = pio.load_project(project_dir)
    config_errors, config_warnings = pio.validate(draft)
    for warning in config_warnings:
        logger.log_warning(f"Config check: {warning}")
    if config_errors:
        return "Error: invalid project configuration: " + "; ".join(config_errors)

    loaded = _load_project_config(project_dir)
    if isinstance(loaded, str):
        return loaded
    candidates, judge_config, test_cases = loaded

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

    # Capabilities, from the client's own registry and with no network: a judge
    # that cannot do its job would turn every score into -1 after the
    # candidates have been paid for, so that blocks; a candidate that cannot
    # take an attachment only skips that test, so that is announced.
    capability_errors, capability_warnings = offline_findings(
        candidates, judge_config, test_cases, project_dir,
        verdict=output_mode != "winner_only",
    )
    for warning in capability_warnings:
        logger.log_warning(f"Capability check: {warning}")
    if capability_errors:
        return "Error: a judge cannot run as configured: " + "; ".join(capability_errors)

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
        #
        # readonly, like analyze and render, and for a reason of its own: this
        # is a pure read, but a plain load renames the log to .bak on a hash
        # mismatch. A config saved from the editor while a benchmark runs - two
        # Streamlit apps meant to run side by side make that ordinary - would
        # therefore destroy the events phases 1 and 2 just wrote, and
        # append_event would then recreate the file with no meta line.
        #
        # A mismatch here still yields hash_match=False, but with the events
        # loaded, which is what this call needs. run_project does not recheck
        # hash_match after this line and must not start to: these events are
        # the ones this very run just appended.
        progress_state = ProgressState.load(project_dir, readonly=True)

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
        attach_best(results, progress_state.eval_events, progress_state.gen_events)
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
    loaded = _load_project_config(project_dir, load_candidates=False)
    if isinstance(loaded, str):
        return loaded
    _, judge_config, test_cases = loaded

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
        A one-line status string, never a bare path: a confirmation naming
        the report file (see run_project's own Returns) when the log holds a
        verdict, `Error: ...` on failure, or a partial-progress summary when
        the log has no verdict yet to render from.
    """
    invalid = _invalid_output_mode(output_mode)
    if invalid:
        return invalid

    loaded = _load_project_config(project_dir)
    if isinstance(loaded, str):
        return loaded
    candidates, judge_config, test_cases = loaded

    progress_state = ProgressState.load(project_dir, readonly=True)
    if not progress_state.events:
        return f"No progress found in {project_dir}"

    if not progress_state.hash_match:
        logger.log_warning(
            "Current configuration does not match the progress. "
            "Best-effort render will be performed."
        )

    global_criteria = GlobalCriteria.load(project_dir)
    judge_config.global_criteria = global_criteria
    results = _initialize_test_results(test_cases, project_dir)

    results_by_id = {r.test_id: r for r in results}

    # Read the deduplicated dicts, never progress_state.events: --retry-errors
    # appends a corrected event rather than rewriting the log, so the same key
    # can appear on several lines and iterating the raw list would count every
    # superseded attempt again. The dicts keep the first insertion position and
    # the last value, so the ordering here is the one the run itself produced.
    for (cand_id, test_id, _rep), event in progress_state.gen_events.items():
        test_result = results_by_id.get(test_id)
        if test_result:
            cand_perf = test_result.candidates_perf.setdefault(cand_id, CandidatePerformance())
            # Matches generation.py's resume branch exactly, defaults included:
            # a log written before a field existed must still render.
            apply_gen_event(cand_perf, event)

    for (cand_id, test_id, _rep), event in progress_state.eval_events.items():
        test_result = results_by_id.get(test_id)
        if test_result:
            # setdefault here too: an eval whose gen event is missing for this
            # test would otherwise raise KeyError on a lookup that has no
            # reason to assume the two logs cover the same keys.
            cand_perf = test_result.candidates_perf.setdefault(cand_id, CandidatePerformance())
            apply_eval_event(cand_perf, event)

    # One final pass over the whole log, rather than tracked incrementally
    # while the two loops above run: attach_best is the single place "best"
    # is decided, shared with run_project's own call site.
    attach_best(results, progress_state.eval_events, progress_state.gen_events)
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
    if progress_state.stale_reasoning_keys:
        # Pointed at `analyze`, not `run`, unlike the two above: analyze may
        # repair, since it already buys reasoning-judge calls. The stale ones
        # matter here because their unit offsets index a trace that has since
        # been replaced, so the report would colour the wrong words rather
        # than fail.
        logger.log_warning(
            f"{len(progress_state.stale_reasoning_keys)} reasoning analysis(es) in this log "
            "describe a trace that was regenerated afterwards. Run 'prompttestenv analyze' "
            "to redo them."
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