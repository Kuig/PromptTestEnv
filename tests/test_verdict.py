from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompttestenv.models import (
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    TestCaseResult,
)
from prompttestenv.verdict import evaluate_best_candidate_fast, generate_verdict, save_verdict_debug_file
from testutils import LoggerResetTestCase


def _make_result(cand_name: str, score: float) -> TestCaseResult:
    result = TestCaseResult(test_id="t1", prompt="p", criteria="c")
    perf = CandidatePerformance()
    perf.scores.append(score)
    result.candidates_perf[cand_name] = perf
    return result


class TestEvaluateBestCandidateFast(unittest.TestCase):
    def test_picks_the_highest_average_scorer(self):
        candidates = [Candidate(name="A", provider="p", model="m"), Candidate(name="B", provider="p", model="m")]
        results = [_make_result("A", 6.0), _make_result("B", 9.0)]
        winner_line = evaluate_best_candidate_fast(candidates, results)
        self.assertIn("B", winner_line)
        self.assertIn("9.00", winner_line)


class TestSaveVerdictDebugFile(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_writes_expected_content(self):
        save_verdict_debug_file(self.project_dir, "sys prompt", "user prompt", "google", "gemini", 0.5)
        debug_file = Path(self.project_dir) / "verdict_prompt_debug.txt"
        content = debug_file.read_text(encoding="utf-8")
        self.assertIn("sys prompt", content)
        self.assertIn("user prompt", content)
        self.assertIn("gemini", content)

    def test_write_failure_logs_warning_instead_of_raising(self):
        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("prompttestenv.logger.log_warning") as mock_warn:
            save_verdict_debug_file(self.project_dir, "s", "u", "google", "gemini", 0.5)
        mock_warn.assert_called_once()


class TestGenerateVerdict(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="A", provider="google", model="gemini")]
        self.results = [_make_result("A", 8.0)]

    def _judge_config(self, **overrides):
        jc = JudgeConfig()
        jc.verdict_judge.verdict_template = "SUMMARY:\n{summary_data}\nCRITERIA:{global_criteria}"
        jc.verdict_judge.global_verdict_template = "GROUPS:\n{group_verdicts_data}\nCRITERIA:{global_criteria}"
        jc.global_criteria = GlobalCriteria(mode="none")
        for key, value in overrides.items():
            setattr(jc.verdict_judge, key, value)
        return jc

    def test_missing_verdict_template_short_circuits_without_llm_call(self):
        jc = self._judge_config(verdict_template="")
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertEqual(result, "No verdict template provided.")
        mock_llm.assert_not_called()

    def test_ungrouped_path_strips_code_fence(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = ("```\nFinal verdict text\n```", 0, 0, "")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertEqual(result, "Final verdict text")
        mock_llm.assert_called_once()

    def test_ungrouped_path_leaves_plain_text_untouched(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = ("Plain verdict, no fences.", 0, 0, "")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertEqual(result, "Plain verdict, no fences.")

    def test_grouped_path_calls_once_per_group_plus_global(self):
        jc = self._judge_config()
        jc.group_verdicts = True
        results = [_make_result("A", 8.0), _make_result("A", 4.0)]
        results[0].group = "G1"
        results[1].group = "G2"

        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.side_effect = [
                ("Group 1 verdict", 0, 0, ""),
                ("Group 2 verdict", 0, 0, ""),
                ("Global verdict text", 0, 0, ""),
            ]
            result = generate_verdict(self.candidates, results, self.project_dir, jc)

        self.assertEqual(mock_llm.call_count, 3)
        data = json.loads(result)
        self.assertTrue(data["is_grouped"])
        self.assertEqual(len(data["groups"]), 2)
        self.assertEqual(data["global_verdict"], "Global verdict text")

    def test_grouped_path_missing_global_template_short_circuits_after_group_calls(self):
        jc = self._judge_config(global_verdict_template="")
        jc.group_verdicts = True
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = ("Group verdict", 0, 0, "")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertEqual(result, "No global verdict template provided.")
        mock_llm.assert_called_once()  # only the per-group call, no global call

    def test_llm_exception_is_caught_and_returned_as_error_string(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.side_effect = RuntimeError("provider down")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertIn("Error generating verdict:", result)
        self.assertIn("provider down", result)


if __name__ == "__main__":
    unittest.main()
