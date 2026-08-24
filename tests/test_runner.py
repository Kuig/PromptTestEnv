from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from prompttestenv.models import (
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    ProgressState,
    TestCase,
    TestCaseResult,
)
from prompttestenv.runner import (
    _generate_output,
    _initialize_test_results,
    render_from_progress,
    run_project,
)
from testutils import LoggerResetTestCase, make_temp_project


class TestInitializeTestResults(unittest.TestCase):
    def test_builds_results_with_media_path_when_file_set(self):
        test_cases = [
            TestCase(id="t1", prompt="p", criteria="c"),
            TestCase(id="t2", prompt="p2", criteria="c2", file="test_files/sample.txt"),
        ]
        results = _initialize_test_results(test_cases, "/fake/project")

        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0].file_used)
        self.assertEqual(results[1].file_used, "test_files/sample.txt")
        self.assertIn("test_files", results[1]._media_file_path)


class TestGenerateOutput(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="A", provider="google", model="m")]
        result = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        perf = CandidatePerformance()
        perf.scores.append(8.0)
        result.candidates_perf["A"] = perf
        self.results = [result]
        self.judge_config = JudgeConfig()
        self.global_criteria = GlobalCriteria(mode="none")

    def test_winner_only_mode_skips_verdict_generation(self):
        with patch("prompttestenv.runner.generate_verdict") as mock_verdict:
            result = _generate_output(
                "winner_only", self.candidates, self.results, self.project_dir,
                self.judge_config, self.global_criteria, ProgressState(),
            )
        self.assertIn("Winner", result)
        mock_verdict.assert_not_called()

    def test_resumed_verdict_is_not_regenerated(self):
        with patch("prompttestenv.runner.generate_verdict") as mock_verdict, \
             patch("prompttestenv.runner.generate_html_report", return_value="/fake/report.html"):
            progress = ProgressState(verdict="Already generated verdict")
            _generate_output(
                "html", self.candidates, self.results, self.project_dir,
                self.judge_config, self.global_criteria, progress,
            )
        mock_verdict.assert_not_called()

    def test_md_mode_writes_markdown_file(self):
        with patch("prompttestenv.runner.generate_verdict", return_value="Plain verdict."):
            result = _generate_output(
                "md", self.candidates, self.results, self.project_dir,
                self.judge_config, self.global_criteria, ProgressState(),
            )
        self.assertIn("Markdown generated", result)
        md_files = list((Path(self.project_dir) / "Report").glob("*.md"))
        self.assertEqual(len(md_files), 1)
        self.assertEqual(md_files[0].read_text(encoding="utf-8"), "Plain verdict.")

    def test_grouped_json_verdict_converted_to_markdown_sections(self):
        grouped = json.dumps({
            "is_grouped": True,
            "groups": [{"group_name": "G1", "verdict": "G1 text"}],
            "global_verdict": "Global text",
        })
        with patch("prompttestenv.runner.generate_verdict", return_value=grouped):
            _generate_output(
                "md", self.candidates, self.results, self.project_dir,
                self.judge_config, self.global_criteria, ProgressState(),
            )
        md_files = list((Path(self.project_dir) / "Report").glob("*.md"))
        content = md_files[0].read_text(encoding="utf-8")
        self.assertIn("VERDICT FOR GROUP: G1", content)
        self.assertIn("GLOBAL VERDICT", content)

    def test_html_mode_delegates_to_generate_html_report(self):
        with patch("prompttestenv.runner.generate_verdict", return_value="Plain verdict."), \
             patch("prompttestenv.runner.generate_html_report", return_value="/fake/report.html") as mock_html:
            result = _generate_output(
                "html", self.candidates, self.results, self.project_dir,
                self.judge_config, self.global_criteria, ProgressState(),
            )
        mock_html.assert_called_once()
        self.assertIn("/fake/report.html", result)


class TestRunProject(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_missing_project_dir_returns_error(self):
        result = run_project("/definitely/does/not/exist")
        self.assertIn("ERROR", result)

    def test_missing_config_file_returns_error_string(self):
        (Path(self.project_dir) / "candidates.json").unlink()
        result = run_project(self.project_dir)
        self.assertIn("Error", result)

    def test_empty_candidates_returns_error(self):
        (Path(self.project_dir) / "candidates.json").write_text("[]", encoding="utf-8")
        result = run_project(self.project_dir)
        self.assertIn("No candidates configured", result)

    def test_empty_test_cases_returns_error(self):
        (Path(self.project_dir) / "test_cases.json").write_text("[]", encoding="utf-8")
        result = run_project(self.project_dir)
        self.assertIn("No test cases found", result)

    def test_hash_mismatch_refuses_to_resume(self):
        # Seed progress.jsonl with a stored hash that cannot match the real
        # one, without going through run_project() (which would otherwise
        # make a real LLM call before we get the chance to corrupt the file).
        progress_file = Path(self.project_dir) / "progress.jsonl"
        progress_file.write_text(json.dumps({"type": "meta", "config_hash": "deadbeef"}) + "\n", encoding="utf-8")
        result = run_project(self.project_dir, output_mode="winner_only")
        self.assertIn("configuration has changed", result)

    def test_happy_path_runs_all_phases_and_calls_teardown(self):
        with patch("prompttestenv.runner.run_generation_phase") as mock_gen, \
             patch("prompttestenv.runner.run_evaluation_phase") as mock_eval, \
             patch("prompttestenv.runner.generate_verdict", return_value="Verdict text."), \
             patch("prompttestenv.runner.teardown") as mock_teardown:
            mock_gen.return_value = [{"fake": "pending_eval"}]
            result = run_project(self.project_dir, output_mode="md")

        mock_gen.assert_called_once()
        mock_eval.assert_called_once()
        mock_teardown.assert_called_once()
        self.assertIn("Markdown generated", result)

    def test_teardown_still_called_when_generation_phase_raises(self):
        with patch("prompttestenv.runner.run_generation_phase", side_effect=RuntimeError("boom")), \
             patch("prompttestenv.runner.teardown") as mock_teardown:
            with self.assertRaises(RuntimeError):
                run_project(self.project_dir, output_mode="winner_only")
        mock_teardown.assert_called_once()

    def test_no_pending_evals_returns_error(self):
        with patch("prompttestenv.runner.run_generation_phase", return_value=[]), \
             patch("prompttestenv.runner.teardown"):
            result = run_project(self.project_dir, output_mode="winner_only")
        self.assertIn("No results produced", result)


class TestRenderFromProgress(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_no_progress_file_reports_none_found(self):
        result = render_from_progress(self.project_dir)
        self.assertIn("No progress found", result)

    def test_rebuilds_results_from_events_and_tracks_best_score(self):
        from prompttestenv.progress import calculate_config_hash
        current_hash = calculate_config_hash(self.project_dir)
        lines = [
            json.dumps({"type": "meta", "config_hash": current_hash}),
            json.dumps({"type": "gen", "cand_id": "A", "test_id": "customer_email", "rep": 0, "output": "low quality", "tokens": 10, "reasoning_tokens": 0, "elapsed": 1.0}),
            json.dumps({"type": "eval", "cand_id": "A", "test_id": "customer_email", "rep": 0, "score": 4, "global_score": -1, "reason": "meh", "g_reason": "N/A"}),
            json.dumps({"type": "gen", "cand_id": "A", "test_id": "customer_email", "rep": 1, "output": "great quality", "tokens": 12, "reasoning_tokens": 0, "elapsed": 1.0}),
            json.dumps({"type": "eval", "cand_id": "A", "test_id": "customer_email", "rep": 1, "score": 9, "global_score": -1, "reason": "excellent", "g_reason": "N/A"}),
        ]
        (Path(self.project_dir) / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = render_from_progress(self.project_dir)
        self.assertIn("Partial progress", result)
        self.assertIn("2 generation(s)", result)
        self.assertIn("2 evaluation(s)", result)

    def test_verdict_present_generates_final_output(self):
        from prompttestenv.progress import calculate_config_hash
        current_hash = calculate_config_hash(self.project_dir)
        lines = [
            json.dumps({"type": "meta", "config_hash": current_hash}),
            json.dumps({"type": "gen", "cand_id": "A", "test_id": "customer_email", "rep": 0, "output": "x", "tokens": 1, "reasoning_tokens": 0, "elapsed": 0.1}),
            json.dumps({"type": "eval", "cand_id": "A", "test_id": "customer_email", "rep": 0, "score": 8, "global_score": -1, "reason": "ok", "g_reason": "N/A"}),
            json.dumps({"type": "verdict", "content": "Final verdict text."}),
        ]
        (Path(self.project_dir) / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

        with patch("prompttestenv.runner.generate_html_report", return_value="/fake/report.html") as mock_html:
            result = render_from_progress(self.project_dir)

        mock_html.assert_called_once()
        self.assertIn("/fake/report.html", result)


if __name__ == "__main__":
    unittest.main()
