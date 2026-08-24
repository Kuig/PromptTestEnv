from __future__ import annotations

import os
import datetime

import prompttestenv.logger as logger
from prompttestenv.api import teardown
from prompttestenv.evaluator import run_evaluation_phase
from prompttestenv.verdict import generate_verdict, evaluate_best_candidate_fast
from prompttestenv.generation import run_generation_phase
from prompttestenv.reporting import generate_html_report
from prompttestenv.models import TestCaseResult, CandidatePerformance, Candidate, TestCase, JudgeConfig, GlobalCriteria, ProgressState
from prompttestenv.progress import append_event


def _initialize_test_results(test_cases: list[TestCase], project_dir: str) -> list[TestCaseResult]:
    """Create TestCaseResult objects and attach media file paths.

    For each test case that references a file, stores the local path. The
    actual file upload (for Google) or inline encoding (for Ollama) is handled
    by get_llm_response() / UnifiedAiClient at call time.

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
        if test.file:
            media_file_path = os.path.join(project_dir, test.file)
            test_result.file_used = test.file
            test_result._media_file_path = media_file_path
        results.append(test_result)
    return results


def _generate_output(
    output_mode: str,
    candidates: list[Candidate],
    results: list[TestCaseResult],
    project_dir: str,
    judge_config: JudgeConfig,
    global_criteria: GlobalCriteria,
    progress_state: ProgressState,
) -> str:
    """Generate the final benchmark report.

    Args:
        output_mode: Report format — 'html', 'md', or 'winner_only'.
        candidates: Resolved candidate configurations.
        results: Completed test case results.
        project_dir: Benchmark project directory path.
        judge_config: JudgeConfig instance.
        global_criteria: GlobalCriteria instance.
        progress_state: Current progress state.

    Returns:
        Path to the report or a summary string.
    """
    if output_mode == "winner_only":
        return evaluate_best_candidate_fast(candidates, results)

    if progress_state.verdict:
        verdict = progress_state.verdict
        logger.log_info("Verdict resumed from log.")
    else:
        verdict = generate_verdict(candidates, results, project_dir, judge_config)
        append_event(project_dir, {"type": "verdict", "content": verdict})

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{now_str}_{len(candidates)}C_{len(results)}T"

    verdict_for_md = verdict
    if verdict.strip().startswith("{"):
        try:
            import json
            verdict_data = json.loads(verdict)
            if verdict_data.get("is_grouped"):
                md_text = ""
                for g in verdict_data["groups"]:
                    md_text += f"# VERDICT FOR GROUP: {g['group_name']}\n{g['verdict']}\n\n"
                md_text += f"# GLOBAL VERDICT\n{verdict_data['global_verdict']}\n"
                verdict_for_md = md_text
        except Exception:
            pass

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
) -> str:
    """Execute the benchmark on an existing project directory.

    Args:
        project_dir: Path to the project directory (must contain config files).
        output_mode: Report format — 'html', 'md', or 'winner_only'.
        force_restart: If True, deletes progress.jsonl and restarts from scratch.

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

    try:
        pending_evals = run_generation_phase(
            candidates, results, judge_config, project_dir, progress_state
        )

        if not pending_evals:
            return "ERROR: No results produced."

        run_evaluation_phase(pending_evals, judge_config, project_dir, progress_state)

    finally:
        teardown()

    return _generate_output(
        output_mode, candidates, results, project_dir, judge_config, global_criteria, progress_state
    )


def render_from_progress(project_dir: str) -> str:
    """Regenerate the report from an existing progress.jsonl without re-running.

    Args:
        project_dir: Path to the benchmark project directory.

    Returns:
        Report path string or partial progress summary.
    """
    progress_state = ProgressState.load(project_dir, force_restart=False)
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

    best_scores: dict = {}
    for event in progress_state.events:
        if event["type"] == "gen":
            cand_id = event["cand_id"]
            test_result = next(
                (r for r in results if r.test_id == event["test_id"]), None
            )
            if test_result:
                if cand_id not in test_result.candidates_perf:
                    test_result.candidates_perf[cand_id] = CandidatePerformance()
                cand_perf = test_result.candidates_perf[cand_id]
                cand_perf.tokens.append(event["tokens"])
                cand_perf.reasoning_tokens.append(event["reasoning_tokens"])
                cand_perf.times.append(event["elapsed"])
        elif event["type"] == "eval":
            cand_id = event["cand_id"]
            test_result = next(
                (r for r in results if r.test_id == event["test_id"]), None
            )
            if test_result:
                cand_perf = test_result.candidates_perf[cand_id]
                cand_perf.scores.append(event["score"])
                cand_perf.global_scores.append(event["global_score"])
                # Restore reasoning analysis if present in the progress event
                r_analysis = event.get("reasoning_analysis")
                if r_analysis:
                    cand_perf.reasoning_analyses.append(r_analysis)
                score_key = (cand_id, test_result.test_id)
                if event["score"] > best_scores.get(score_key, -1):
                    best_scores[score_key] = event["score"]
                    gen_event = next(
                        (
                            e for e in progress_state.events
                            if e["type"] == "gen"
                            and e["cand_id"] == cand_id
                            and e["test_id"] == event["test_id"]
                            and e["rep"] == event["rep"]
                        ),
                        None,
                    )
                    if gen_event:
                        cand_perf.best_output = gen_event["output"]
                    cand_perf.best_reason = event["reason"]
                    cand_perf.best_global_reason = event["g_reason"]

    if progress_state.verdict:
        return _generate_output(
            "html", candidates, results, project_dir, judge_config, global_criteria, progress_state
        )

    return (
        f"Partial progress: "
        f"{len(progress_state.completed_gen)} generation(s) and "
        f"{len(progress_state.completed_eval)} evaluation(s) found."
    )