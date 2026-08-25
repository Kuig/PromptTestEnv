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
from prompttestenv.verdict import (
    _build_summary_data,
    evaluate_best_candidate_fast,
    generate_verdict,
    save_verdict_debug_file,
)
from testutils import LoggerResetTestCase


def _make_result(cand_name: str, score: float) -> TestCaseResult:
    result = TestCaseResult(test_id="t1", prompt="p", criteria="c")
    perf = CandidatePerformance()
    perf.scores.append(score)
    result.candidates_perf[cand_name] = perf
    return result


def _reasoning_analysis() -> dict:
    """One repetition's reasoning analysis, as stored in progress.jsonl."""
    return {
        "interpretation_pct": 20.0,
        "planning_pct": 10.0,
        "pure_reasoning_pct": 40.0,
        "output_formulation_pct": 30.0,
        "alt_path": 2,
        "autocorrect": 1,
        "alignment_score": 9,
    }


class TestEvaluateBestCandidateFast(unittest.TestCase):
    def test_picks_the_highest_average_scorer(self):
        candidates = [Candidate(name="A", provider="p", model="m"), Candidate(name="B", provider="p", model="m")]
        results = [_make_result("A", 6.0), _make_result("B", 9.0)]
        winner_line = evaluate_best_candidate_fast(candidates, results)
        self.assertIn("B", winner_line)
        self.assertIn("9.00", winner_line)


class TestBuildSummaryData(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            Candidate(name="Alpha", provider="google", model="gemini"),
            Candidate(name="Beta", provider="google", model="gemini"),
        ]
        self.judge_config = JudgeConfig(repetitions=7)

    def _row(self, **overrides) -> TestCaseResult:
        row = TestCaseResult(
            test_id="t1",
            prompt="Summarize the attached report.",
            criteria="Must mention the revenue figure.",
        )
        for key, value in overrides.items():
            setattr(row, key, value)
        for cand in self.candidates:
            perf = CandidatePerformance()
            perf.scores.append(8.0)
            perf.global_scores.append(7.0)
            perf.times.append(1.5)
            perf.tokens.append(100)
            perf.reasoning_tokens.append(200)
            perf.best_reason = "Solid answer."
            row.candidates_perf[cand.name] = perf
        return row

    def test_metadata_header_declares_the_repetition_count(self):
        out = _build_summary_data([self._row()], self.candidates, self.judge_config)
        self.assertIn("# BENCHMARK METADATA", out)
        self.assertIn("Repetitions per candidate x test case: 7", out)

    def test_metadata_header_lists_every_judge_type(self):
        out = _build_summary_data([self._row()], self.candidates, self.judge_config)
        for judge_type in ("llm-judge", "similarity", "assert"):
            self.assertIn(judge_type, out)

    def test_test_case_block_declares_judge_type_prompt_and_criteria(self):
        row = self._row(judge_type="similarity")
        out = _build_summary_data([row], self.candidates, self.judge_config)
        self.assertIn("JUDGE TYPE: similarity", out)
        self.assertIn("PROMPT:\nSummarize the attached report.", out)
        self.assertIn(
            "EVALUATION CRITERIA (rubric, NOT shown to the candidate):\n"
            "Must mention the revenue figure.",
            out,
        )

    def test_metadata_header_states_what_the_candidate_could_see(self):
        out = _build_summary_data([self._row()], self.candidates, self.judge_config)
        # Scope to the header: the per-test-case CRITERIA label carries the same
        # wording, so asserting over the whole payload would pass without it.
        header = out.split("# OVERALL AGGREGATE")[0]
        self.assertIn("NOT shown to the candidate", header)
        self.assertIn("its own system prompt", header)

    def test_prompt_and_criteria_are_not_truncated(self):
        long_prompt = "x" * 5000
        out = _build_summary_data(
            [self._row(prompt=long_prompt)], self.candidates, self.judge_config
        )
        self.assertIn(long_prompt, out)

    def test_attachment_is_declared_when_present_and_when_absent(self):
        with_file = _build_summary_data(
            [self._row(file_used="paper.pdf")], self.candidates, self.judge_config
        )
        self.assertIn("ATTACHMENT: paper.pdf", with_file)

        without_file = _build_summary_data([self._row()], self.candidates, self.judge_config)
        self.assertIn("ATTACHMENT: none", without_file)

    def test_aggregate_block_lists_each_candidate_exactly_once(self):
        rows = [self._row(), self._row(test_id="t2")]
        out = _build_summary_data(rows, self.candidates, self.judge_config)
        aggregate = out.split("# TEST RESULTS")[0]
        self.assertIn("# OVERALL AGGREGATE", aggregate)
        for cand in self.candidates:
            self.assertEqual(aggregate.count(f"> CANDIDATE: {cand.name}"), 1)

    def test_candidate_label_replaces_the_ambiguous_system_label(self):
        out = _build_summary_data([self._row()], self.candidates, self.judge_config)
        self.assertIn("> CANDIDATE: Alpha", out)
        self.assertNotIn("> SYSTEM:", out)

    def test_missing_performance_renders_as_not_available(self):
        row = self._row()
        del row.candidates_perf["Beta"]
        out = _build_summary_data([row], self.candidates, self.judge_config)
        self.assertIn("> CANDIDATE: Beta\n    N/A (no completed repetitions)", out)

    def test_global_score_renders_as_not_available_when_never_scored(self):
        row = self._row()
        row.candidates_perf["Alpha"].global_scores.clear()
        out = _build_summary_data([row], self.candidates, self.judge_config)
        self.assertIn("Global Score: N/A", out)

    def test_reasoning_profile_reports_all_four_categories(self):
        row = self._row()
        row.candidates_perf["Alpha"].reasoning_analyses.append(_reasoning_analysis())
        self.judge_config.reasoning_analysis = True
        out = _build_summary_data([row], self.candidates, self.judge_config)

        self.assertIn("interpretation 20.0%", out)
        self.assertIn("planning 10.0%", out)
        self.assertIn("problem-solving 40.0%", out)
        self.assertIn("output formulation 30.0%", out)
        self.assertIn("Alternatives explored: 2.0", out)
        self.assertIn("Self-corrections: 1.0", out)
        self.assertIn("Response/reasoning alignment: 9.0", out)

    def test_reasoning_profile_drops_the_misleading_cognitive_framing(self):
        row = self._row()
        row.candidates_perf["Alpha"].reasoning_analyses.append(_reasoning_analysis())
        self.judge_config.reasoning_analysis = True
        out = _build_summary_data([row], self.candidates, self.judge_config)

        self.assertNotIn("cognitive resources", out)
        self.assertNotIn("interacting with the user", out)

    def test_reasoning_profile_omitted_when_analysis_is_disabled(self):
        row = self._row()
        row.candidates_perf["Alpha"].reasoning_analyses.append(_reasoning_analysis())
        self.judge_config.reasoning_analysis = False
        out = _build_summary_data([row], self.candidates, self.judge_config)
        self.assertNotIn("Reasoning trace profile", out)

    def test_reasoning_profile_omitted_when_no_trace_was_analysed(self):
        self.judge_config.reasoning_analysis = True
        out = _build_summary_data([self._row()], self.candidates, self.judge_config)
        self.assertNotIn("Reasoning trace profile", out)

    def test_multiline_notes_stay_indented_and_keep_record_boundaries_intact(self):
        row = self._row()
        row.candidates_perf["Alpha"].best_reason = "First line.\nSecond line."
        out = _build_summary_data([row], self.candidates, self.judge_config)

        # The continuation is indented under Notes, so it cannot be mistaken for
        # a new field, and the next candidate still opens its own record.
        self.assertIn("    Notes: First line.\n      Second line.\n", out)
        self.assertIn("\n  > CANDIDATE: Beta\n", out)

    def test_notes_are_the_last_field_of_a_candidate_record(self):
        row = self._row()
        row.candidates_perf["Alpha"].reasoning_analyses.append(_reasoning_analysis())
        self.judge_config.reasoning_analysis = True
        out = _build_summary_data([row], self.candidates, self.judge_config)

        results_section = out.split("# TEST RESULTS")[1]
        alpha_block = results_section.split("> CANDIDATE: Alpha")[1].split("> CANDIDATE: Beta")[0]
        self.assertLess(
            alpha_block.index("Reasoning trace profile"), alpha_block.index("Notes:")
        )


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
