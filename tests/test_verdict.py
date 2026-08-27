from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompttestenv.models import (
    LlmResult,
    ProgressState,
    REASONING_DIMENSIONS,
    REASONING_SCOPE_ALL,
    REASONING_SCOPE_BEST,
    REASONING_SCOPE_NONE,
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    ReasoningStats,
    TestCaseResult,
)
from prompttestenv.config import get_app_config
from prompttestenv.runner import _generate_output
from prompttestenv.verdict import (
    DEFAULT_DEBUG_FILENAME,
    MetadataError,
    _build_summary_data,
    _format_reasoning_profile,
    _sanitize_filename_part,
    evaluate_best_candidate_fast,
    generate_verdict,
    save_verdict_debug_file,
)
from testutils import LoggerResetTestCase, make_temp_project


def _make_result(cand_name: str, score: float) -> TestCaseResult:
    result = TestCaseResult(test_id="t1", prompt="p", criteria="c")
    perf = CandidatePerformance()
    perf.scores.append(score)
    result.candidates_perf[cand_name] = perf
    return result


def _reasoning_analysis(**overrides) -> dict:
    """One repetition's reasoning analysis, as stored in progress.jsonl."""
    analysis = ReasoningStats(
        coverage_framing=0.20,
        coverage_solving=0.40,
        coverage_presentation=0.30,
        density=0.90,
        alt_path=2,
        autocorrect=1,
        alignment_score=9,
        repetition_rate=0.05,
        trace_response_drift=0.82,
    ).to_dict()
    analysis.update(overrides)
    return analysis


class TestEvaluateBestCandidateFast(unittest.TestCase):
    def test_picks_the_highest_average_scorer(self):
        candidates = [Candidate(name="A", provider="p", model="m"), Candidate(name="B", provider="p", model="m")]
        results = [_make_result("A", 6.0), _make_result("B", 9.0)]
        winner_line = evaluate_best_candidate_fast(candidates, results)
        self.assertIn("B", winner_line)
        self.assertIn("9.00", winner_line)


class _SummaryFixture:
    """Shared setup for the payload tests. Not a TestCase: inheriting one would
    re-run every parent test in each subclass."""

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

    def _payload(self, rows=None, **config) -> tuple[str, str]:
        """Return (metadata_text, summary_data) for one call, applying overrides."""
        for key, value in config.items():
            setattr(self.judge_config, key, value)
        return _build_summary_data(
            rows if rows is not None else [self._row()], self.candidates, self.judge_config
        )

    def _aggregate(self, rows=None, **config) -> str:
        """Return just the OVERALL AGGREGATE section of the results half."""
        _metadata, out = self._payload(rows, **config)
        return out[out.index("# OVERALL AGGREGATE"):out.index("# TEST RESULTS")]

    def _table_rows(self, aggregate: str) -> list[str]:
        """The data rows of every table, without headers, separators or prose."""
        return [
            line for line in aggregate.splitlines()
            if line.startswith("|") and not line.startswith("|--") and "Candidate" not in line
        ]


class TestBuildSummaryData(_SummaryFixture, unittest.TestCase):
    """Tests below assert against whichever half of the payload the content in
    question now lives in: `metadata` (appended to the SYSTEM prompt) or `out`
    aka summary_data (interpolated into {summary_data} in the USER prompt)."""

    def test_metadata_header_declares_the_repetition_count(self):
        metadata, _out = self._payload()
        self.assertIn("# BENCHMARK METADATA", metadata)
        self.assertIn("Repetitions per candidate x test case: 7", metadata)

    def test_metadata_explains_only_the_judge_types_in_use(self):
        """7 of 9 real projects use llm-judge alone and used to read all three."""
        metadata, _out = self._payload()
        self.assertIn("llm-judge", metadata)
        self.assertNotIn("similarity: cosine similarity", metadata)
        self.assertNotIn("assert: a user-authored", metadata)

    def test_metadata_explains_every_judge_type_actually_present(self):
        rows = [
            self._row(),
            self._row(test_id="t2", judge_type="similarity"),
            self._row(test_id="t3", judge_type="assert"),
        ]
        metadata, _out = self._payload(rows)
        for judge_type in ("llm-judge", "similarity", "assert"):
            self.assertIn(judge_type, metadata)

    def test_scores_are_declared_incomparable_only_across_mixed_judges(self):
        single, _out = self._payload()
        self.assertNotIn("not comparable across test cases", single)

        mixed, _out = self._payload(
            [self._row(), self._row(test_id="t2", judge_type="assert")]
        )
        self.assertIn("not comparable across test cases", mixed)

    def test_score_scale_note_survives_with_the_reasoning_section_off(self):
        """It used to be glued to the end of the reasoning paragraph."""
        metadata, _out = self._payload()
        self.assertNotIn("SHAPE of a thinking trace", metadata)
        self.assertIn("'N/A' means not computed", metadata)

    def test_cost_section_appears_with_thinking_tokens_and_no_analysis(self):
        metadata, _out = self._payload()
        self.assertIn("'Think/point'", metadata)

    def test_cost_section_is_omitted_without_thinking_tokens(self):
        row = self._row()
        for perf in row.candidates_perf.values():
            perf.reasoning_tokens.clear()
        metadata, _out = self._payload([row])
        self.assertNotIn("'Think/point'", metadata)

    def test_an_empty_section_is_refused_rather_than_skipped(self):
        """Dropping a caveat while keeping its figures is the failure to avoid."""
        config = get_app_config()
        original = config.verdict_metadata.header
        config.verdict_metadata.header = ""
        self.addCleanup(setattr, config.verdict_metadata, "header", original)
        with self.assertRaises(MetadataError) as caught:
            _build_summary_data([self._row()], self.candidates, self.judge_config)
        self.assertIn("header", str(caught.exception))

    def test_an_unresolvable_placeholder_is_refused(self):
        config = get_app_config()
        original = config.verdict_metadata.figures
        config.verdict_metadata.figures = "Repetitions: {no_such_placeholder}"
        self.addCleanup(setattr, config.verdict_metadata, "figures", original)
        with self.assertRaises(MetadataError) as caught:
            _build_summary_data([self._row()], self.candidates, self.judge_config)
        self.assertIn("figures", str(caught.exception))

    def test_test_case_block_declares_judge_type_prompt_and_criteria(self):
        row = self._row(judge_type="similarity")
        _metadata, out = self._payload([row])
        self.assertIn("JUDGE TYPE: similarity", out)
        self.assertIn("PROMPT:\nSummarize the attached report.", out)
        self.assertIn(
            "EVALUATION CRITERIA (rubric, NOT shown to the candidate):\n"
            "Must mention the revenue figure.",
            out,
        )

    def test_metadata_header_states_what_the_candidate_could_see(self):
        metadata, _out = self._payload()
        self.assertIn("NOT shown to the candidate", metadata)
        self.assertIn("its own system prompt", metadata)

    def test_prompt_and_criteria_are_not_truncated(self):
        long_prompt = "x" * 5000
        _metadata, out = self._payload([self._row(prompt=long_prompt)])
        self.assertIn(long_prompt, out)

    def test_attachment_is_declared_when_present_and_when_absent(self):
        _metadata, with_file = self._payload([self._row(file_used="paper.pdf")])
        self.assertIn("ATTACHMENT: paper.pdf", with_file)

        _metadata, without_file = self._payload()
        self.assertIn("ATTACHMENT: none", without_file)

    def test_aggregate_block_lists_each_candidate_exactly_once(self):
        aggregate = self._aggregate([self._row(), self._row(test_id="t2")])
        self.assertIn("# OVERALL AGGREGATE", aggregate)
        rows = self._table_rows(aggregate)
        self.assertEqual(len(rows), len(self.candidates))
        for cand in self.candidates:
            self.assertEqual(sum(1 for line in rows if cand.name in line), 1)

    def test_aggregate_is_a_table_with_the_expected_columns(self):
        aggregate = self._aggregate()
        for column in ("Candidate", "Task", "Global", "Time", "Out tok", "Think tok", "Think/point"):
            self.assertIn(column, aggregate)
        self.assertIn("|---", aggregate, "the Markdown separator row is missing")

    def test_aggregate_carries_the_cost_per_point(self):
        """200 thinking tokens at a score of 8 is 25 per point."""
        self.assertIn("25", self._aggregate())

    def test_cost_per_point_is_not_available_without_thinking(self):
        row = self._row()
        for perf in row.candidates_perf.values():
            perf.reasoning_tokens.clear()
        self.assertIn("n/a", self._aggregate([row]))

    def test_aggregate_task_column_is_not_available_when_never_scored(self):
        """Every task-judge call failing must read n/a, not 0.00."""
        row = self._row()
        row.candidates_perf["Alpha"].scores.clear()
        rows = self._table_rows(self._aggregate([row]))
        alpha_row = next(line for line in rows if "Alpha" in line)
        self.assertIn("n/a", alpha_row)

    def test_global_criteria_legend_is_part_of_the_metadata(self):
        """Living in metadata_text (appended to the system prompt), not a
        verdict_template preamble, keeps it from reading as an instruction
        that outranks the data."""
        self.judge_config.global_criteria = GlobalCriteria(
            mode="llm-judge", llm_judge_criteria="Be polite."
        )
        metadata, _out = self._payload()
        self.assertIn("How the global score was produced:", metadata)
        self.assertIn("Be polite.", metadata)

    def test_global_criteria_legend_matches_the_active_mode(self):
        cases = {
            "llm-judge": ("llm_judge_criteria", "Be polite.", "llm-judge:"),
            "similarity": ("similarity_criteria", "Target text.", "similarity:"),
            "assert": ("assert_criteria", "s: (10, 'ok')", "assert:"),
        }
        for mode, (field, text, bullet) in cases.items():
            with self.subTest(mode=mode):
                self.judge_config.global_criteria = GlobalCriteria(mode=mode, **{field: text})
                metadata, _out = self._payload()
                self.assertIn(bullet, metadata)
                self.assertIn(text, metadata)

    def test_global_criteria_legend_reports_disabled_mode(self):
        self.judge_config.global_criteria = GlobalCriteria(mode="none")
        metadata, _out = self._payload()
        self.assertIn("Global scoring is disabled", metadata)

    def test_global_criteria_legend_falls_back_when_criteria_text_is_blank(self):
        self.judge_config.global_criteria = GlobalCriteria(mode="llm-judge", llm_judge_criteria="")
        metadata, _out = self._payload()
        self.assertIn("(none set)", metadata)

    def test_global_criteria_legend_reindents_multiline_criteria(self):
        """Only the first line inherits the template's leading indent by default;
        continuation lines must be re-indented to match, not left flush left."""
        self.judge_config.global_criteria = GlobalCriteria(
            mode="llm-judge", llm_judge_criteria="1. Be polite.\n2. No harmful content."
        )
        metadata, _out = self._payload()
        self.assertIn("  1. Be polite.\n  2. No harmful content.", metadata)


class TestPerTestCaseCostIsAlwaysShown(_SummaryFixture, unittest.TestCase):
    """The per-test-case Cost figure needs only tokens and a score.

    Unlike the reasoning profile below it, it must not depend on
    judge_config.reasoning_analysis: a project that never runs the analysis
    phase still bills reasoning tokens and still deserves to see what they cost.
    """

    def test_cost_appears_with_reasoning_analysis_disabled(self):
        self.judge_config.reasoning_analysis = REASONING_SCOPE_NONE
        _metadata, out = self._payload()
        self.assertIn("Cost: 25.0", out, "200 thinking tokens at a score of 8 is 25 per point")

    def test_cost_is_the_mean_of_each_repetitions_own_ratio(self):
        """The exact 30/300 example: mean of ratios is 165, not the ratio of the means."""
        row = self._row()
        perf = row.candidates_perf["Alpha"]
        perf.reasoning_tokens.clear()
        perf.scores.clear()
        perf.reasoning_tokens.extend([300, 300])
        perf.scores.extend([10.0, 1.0])
        self.judge_config.reasoning_analysis = REASONING_SCOPE_NONE
        _metadata, out = self._payload([row])
        self.assertIn("Cost: 165.0", out)

    def test_cost_is_absent_without_thinking_tokens(self):
        row = self._row()
        for perf in row.candidates_perf.values():
            perf.reasoning_tokens.clear()
        self.judge_config.reasoning_analysis = REASONING_SCOPE_NONE
        _metadata, out = self._payload([row])
        self.assertNotIn("Cost:", out)

    def test_cost_carries_a_standard_deviation(self):
        row = self._row()
        perf = row.candidates_perf["Alpha"]
        perf.reasoning_tokens.clear()
        perf.scores.clear()
        perf.reasoning_tokens.extend([300, 300])
        perf.scores.extend([10.0, 1.0])
        _metadata, out = self._payload([row])
        self.assertIn("Cost: 165.0 ± ", out)
        self.assertNotIn("Cost: 165.0 ± 0.0", out)


class TestAggregateReasoningProfile(_SummaryFixture, unittest.TestCase):
    """The pooled reasoning profile, which the payload used to omit entirely."""

    def _row_with_analyses(self, **overrides) -> TestCaseResult:
        row = self._row()
        for perf in row.candidates_perf.values():
            perf.reasoning_analyses.append(_reasoning_analysis(**overrides))
        return row

    def test_profile_table_is_absent_when_analysis_is_disabled(self):
        aggregate = self._aggregate(
            [self._row_with_analyses()], reasoning_analysis=REASONING_SCOPE_NONE
        )
        self.assertNotIn("REASONING PROFILE", aggregate)

    def test_profile_table_is_absent_when_nothing_was_analysed(self):
        aggregate = self._aggregate(reasoning_analysis=REASONING_SCOPE_ALL)
        self.assertNotIn("REASONING PROFILE", aggregate)

    def test_profile_table_carries_every_dimension_and_metric(self):
        aggregate = self._aggregate(
            [self._row_with_analyses()], reasoning_analysis=REASONING_SCOPE_ALL
        )
        self.assertIn("REASONING PROFILE", aggregate)
        for dimension in REASONING_DIMENSIONS:
            self.assertIn(dimension, aggregate)
        for column in ("density", "alt", "corr", "align", "drift", "repet", "source"):
            self.assertIn(column, aggregate)
        self.assertIn("20%", aggregate, "coverage_framing 0.20 should render as 20%")

    def test_scope_best_is_declared_in_the_profile_heading(self):
        aggregate = self._aggregate(
            [self._row_with_analyses()], reasoning_analysis=REASONING_SCOPE_BEST
        )
        self.assertIn("highest-scoring repetition", aggregate)

    def test_not_measured_renders_as_not_available(self):
        aggregate = self._aggregate(
            [self._row_with_analyses(alignment_score=-1, coverage_solving=-1.0)],
            reasoning_analysis=REASONING_SCOPE_ALL,
        )
        rows = "\n".join(self._table_rows(aggregate))
        self.assertIn("n/a", rows)
        self.assertNotIn("-1", rows, "the not-measured sentinel leaked into a cell")

    def test_source_says_unknown_rather_than_claiming_a_raw_trace(self):
        """The flag is absent on traces recorded before providers reported it."""
        aggregate = self._aggregate(
            [self._row_with_analyses(reasoning_is_summary=None)],
            reasoning_analysis=REASONING_SCOPE_ALL,
        )
        self.assertIn("unknown", aggregate)

    def test_source_flags_a_provider_summary(self):
        aggregate = self._aggregate(
            [self._row_with_analyses(reasoning_is_summary=True)],
            reasoning_analysis=REASONING_SCOPE_ALL,
        )
        self.assertIn("summary", aggregate)

    def test_mixed_schemas_raise_a_warning(self):
        row = self._row()
        for index, perf in enumerate(row.candidates_perf.values()):
            perf.reasoning_analyses.append(
                _reasoning_analysis(schema_stamp=f"framing+solving@{index}")
            )
        aggregate = self._aggregate([row], reasoning_analysis=REASONING_SCOPE_ALL)
        self.assertIn("DIFFERENT analysis schemas", aggregate)

    def test_a_single_schema_raises_no_warning(self):
        aggregate = self._aggregate(
            [self._row_with_analyses(schema_stamp="framing+solving@same")],
            reasoning_analysis=REASONING_SCOPE_ALL,
        )
        self.assertNotIn("DIFFERENT analysis schemas", aggregate)

    def test_candidate_label_replaces_the_ambiguous_system_label(self):
        _metadata, out = self._payload()
        self.assertIn("> CANDIDATE: Alpha", out)
        self.assertNotIn("> SYSTEM:", out)

    def test_missing_performance_renders_as_not_available(self):
        row = self._row()
        del row.candidates_perf["Beta"]
        _metadata, out = self._payload([row])
        self.assertIn("> CANDIDATE: Beta\n    N/A (no completed repetitions)", out)

    def test_global_score_renders_as_not_available_when_never_scored(self):
        row = self._row()
        row.candidates_perf["Alpha"].global_scores.clear()
        _metadata, out = self._payload([row])
        self.assertIn("Global Score: N/A", out)

    def test_task_score_renders_as_not_available_when_never_scored(self):
        row = self._row()
        row.candidates_perf["Alpha"].scores.clear()
        _metadata, out = self._payload([row])
        self.assertIn("Task Score: N/A", out)

    def test_profile_states_that_coverages_are_not_shares_of_a_whole(self):
        """Three percentages that do not add up invite a judge to explain the gap."""
        aggregate = self._aggregate(
            [self._row_with_analyses()], reasoning_analysis=REASONING_SCOPE_ALL
        )
        self.assertIn("NOT shares of a whole", aggregate)

    def test_metadata_warns_against_inferring_causality_when_a_profile_is_shown(self):
        self.judge_config.reasoning_analysis = REASONING_SCOPE_ALL
        metadata, _out = self._payload([self._row_with_analyses()])
        self.assertIn("Do not infer that a profile caused a score", metadata)

    def test_the_profile_guidance_is_absent_when_no_profile_is_shown(self):
        """Explaining a table the judge will never see is noise in the prompt."""
        metadata, _out = self._payload()
        self.assertNotIn("Do not infer that a profile caused a score", metadata)

    def test_multiline_notes_stay_indented_and_keep_record_boundaries_intact(self):
        row = self._row()
        row.candidates_perf["Alpha"].best_reason = "First line.\nSecond line."
        _metadata, out = self._payload([row])

        # The continuation is indented under Notes, so it cannot be mistaken for
        # a new field, and the next candidate still opens its own record.
        self.assertIn("    Notes: First line.\n      Second line.\n", out)
        self.assertIn("\n  > CANDIDATE: Beta\n", out)


class TestFormatReasoningProfile(unittest.TestCase):
    """Direct coverage of _format_reasoning_profile().

    Its call site (rendering a profile per candidate per test case) is
    deliberately commented out in _build_summary_data — kept for later, too
    heavy for the payload in the meantime — so these cases test the function
    itself rather than through that now-unreachable integration path.
    """

    def test_reports_every_dimension_and_metric(self):
        out = _format_reasoning_profile([_reasoning_analysis()])
        self.assertIn("framing 20.0%", out)
        self.assertIn("solving 40.0%", out)
        self.assertIn("presentation 30.0%", out)
        self.assertIn("Density", out)
        self.assertIn("Alternatives explored: 2.0", out)
        self.assertIn("Self-corrections: 1.0", out)
        self.assertIn("Response/reasoning alignment: 9.0", out)
        self.assertIn("Trace/response similarity: 0.8", out)

    def test_states_that_coverages_are_not_shares_of_a_whole(self):
        """Three percentages that do not add up invite a judge to explain the gap."""
        out = _format_reasoning_profile([_reasoning_analysis()])
        self.assertIn("NOT shares of a whole", out)

    def test_unmeasured_metrics_say_so_instead_of_reporting_a_number(self):
        out = _format_reasoning_profile(
            [_reasoning_analysis(alt_path=-1, coverage_solving=-1.0)]
        )
        self.assertIn("solving not measured", out)
        self.assertIn("Alternatives explored: not measured", out)

    def test_flags_a_provider_summary(self):
        out = _format_reasoning_profile([_reasoning_analysis(reasoning_is_summary=True)])
        self.assertIn("provider SUMMARY", out)

    def test_does_not_flag_a_raw_trace_as_a_summary(self):
        out = _format_reasoning_profile([_reasoning_analysis(reasoning_is_summary=False)])
        self.assertIn("raw thinking trace", out)
        self.assertNotIn("provider SUMMARY", out)

    def test_drops_the_misleading_cognitive_framing(self):
        out = _format_reasoning_profile([_reasoning_analysis()])
        self.assertNotIn("cognitive resources", out)
        self.assertNotIn("interacting with the user", out)

    def test_scope_best_is_declared_in_the_heading(self):
        out = _format_reasoning_profile([_reasoning_analysis()], scope=REASONING_SCOPE_BEST)
        self.assertIn("WHEN IT SUCCEEDS", out)


class TestSanitizeFilenamePart(unittest.TestCase):
    def test_spaces_and_special_characters_become_underscores(self):
        self.assertEqual(_sanitize_filename_part("Default group"), "Default_group")
        self.assertEqual(_sanitize_filename_part("Coding / Writing!"), "Coding_Writing")

    def test_alphanumerics_dashes_and_underscores_pass_through(self):
        self.assertEqual(_sanitize_filename_part("Group-1_ok"), "Group-1_ok")

    def test_empty_or_all_special_falls_back_to_group(self):
        self.assertEqual(_sanitize_filename_part(""), "group")
        self.assertEqual(_sanitize_filename_part("///"), "group")


class TestSaveVerdictDebugFile(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_writes_expected_content(self):
        save_verdict_debug_file(self.project_dir, "sys prompt", "user prompt", "google", "gemini", 0.5)
        debug_file = Path(self.project_dir) / DEFAULT_DEBUG_FILENAME
        content = debug_file.read_text(encoding="utf-8")
        self.assertIn("sys prompt", content)
        self.assertIn("user prompt", content)
        self.assertIn("gemini", content)

    def test_write_failure_logs_warning_instead_of_raising(self):
        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("prompttestenv.logger.log_warning") as mock_warn:
            save_verdict_debug_file(self.project_dir, "s", "u", "google", "gemini", 0.5)
        mock_warn.assert_called_once()

    def test_filename_argument_controls_the_output_path(self):
        save_verdict_debug_file(
            self.project_dir, "s", "u", "google", "gemini", 0.5, filename="custom.txt"
        )
        self.assertFalse((Path(self.project_dir) / DEFAULT_DEBUG_FILENAME).exists())
        self.assertTrue((Path(self.project_dir) / "custom.txt").exists())


class TestGenerateVerdict(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="A", provider="google", model="gemini")]
        self.results = [_make_result("A", 8.0)]

    def _judge_config(self, **overrides):
        jc = JudgeConfig()
        # No {summary_data}/{group_verdicts_data} placeholders: both are
        # prepended by the framework now, so a template is just the trailer
        # text that follows them.
        jc.verdict_judge.verdict_template = "INSTRUCTIONS"
        jc.verdict_judge.global_verdict_template = "GLOBAL INSTRUCTIONS"
        jc.global_criteria = GlobalCriteria(mode="none")
        for key, value in overrides.items():
            setattr(jc.verdict_judge, key, value)
        return jc

    def test_missing_verdict_template_short_circuits_without_llm_call(self):
        jc = self._judge_config(verdict_template="")
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertIsNone(result)
        mock_llm.assert_not_called()

    def test_ungrouped_path_strips_code_fence(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text="```\nFinal verdict text\n```")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertEqual(result, "Final verdict text")
        mock_llm.assert_called_once()

    def test_ungrouped_path_leaves_plain_text_untouched(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text="Plain verdict, no fences.")
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
                LlmResult(text="Group 1 verdict"),
                LlmResult(text="Group 2 verdict"),
                LlmResult(text="Global verdict text"),
            ]
            result = generate_verdict(self.candidates, results, self.project_dir, jc)

        self.assertEqual(mock_llm.call_count, 3)
        data = json.loads(result)
        self.assertTrue(data["is_grouped"])
        self.assertEqual(len(data["groups"]), 2)
        self.assertEqual(data["global_verdict"], "Global verdict text")

    def test_grouped_path_writes_separate_debug_files_per_group_and_global(self):
        """Every call used to overwrite the same file, so group_verdicts: true
        left only the last payload on disk. Each call now gets its own."""
        jc = self._judge_config()
        jc.group_verdicts = True
        results = [_make_result("A", 8.0), _make_result("A", 4.0)]
        results[0].group = "G1"
        results[1].group = "G2"

        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.side_effect = [
                LlmResult(text="Group 1 verdict"),
                LlmResult(text="Group 2 verdict"),
                LlmResult(text="Global verdict text"),
            ]
            generate_verdict(self.candidates, results, self.project_dir, jc)

        project = Path(self.project_dir)
        g1 = (project / "verdict_prompt_debug_group_G1.txt").read_text(encoding="utf-8")
        g2 = (project / "verdict_prompt_debug_group_G2.txt").read_text(encoding="utf-8")
        glob = (project / "verdict_prompt_debug_global.txt").read_text(encoding="utf-8")
        self.assertFalse((project / DEFAULT_DEBUG_FILENAME).exists())
        self.assertNotEqual(g1, g2)
        self.assertIn("# GROUP VERDICTS:", glob)
        self.assertIn("GLOBAL INSTRUCTIONS", glob)

    def test_save_payload_debug_files_false_writes_nothing(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.SAVE_PAYLOAD_DEBUG_FILES", False), \
                patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text="Verdict text")
            generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertFalse((Path(self.project_dir) / DEFAULT_DEBUG_FILENAME).exists())

    def test_summary_data_is_prepended_and_the_template_used_verbatim(self):
        """No more {summary_data} placeholder: verdict_template is plain
        trailer text, appended after the payload, and may contain literal
        braces (e.g. a JSON example) without needing to double them up."""
        jc = self._judge_config(verdict_template='Analyze. Example: {"score": 1}')
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text="Verdict text")
            generate_verdict(self.candidates, self.results, self.project_dir, jc)
        debug = (Path(self.project_dir) / DEFAULT_DEBUG_FILENAME).read_text(encoding="utf-8")
        self.assertLess(debug.index("# OVERALL AGGREGATE"), debug.index("Analyze. Example:"))
        self.assertIn('{"score": 1}', debug)

    def test_group_verdicts_data_is_prepended_and_the_global_template_used_verbatim(self):
        """Same treatment for global_verdict_template: no {group_verdicts_data}
        placeholder, "# GROUP VERDICTS:" is code-owned, and literal braces in
        the template survive."""
        jc = self._judge_config(global_verdict_template='Summarize. Example: {"winner": "A"}')
        jc.group_verdicts = True
        results = [_make_result("A", 8.0), _make_result("A", 4.0)]
        results[0].group = "G1"
        results[1].group = "G2"
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.side_effect = [
                LlmResult(text="Group 1 verdict"),
                LlmResult(text="Group 2 verdict"),
                LlmResult(text="Global verdict text"),
            ]
            generate_verdict(self.candidates, results, self.project_dir, jc)
        debug = (Path(self.project_dir) / "verdict_prompt_debug_global.txt").read_text(encoding="utf-8")
        self.assertLess(debug.index("# GROUP VERDICTS:"), debug.index("Summarize. Example:"))
        self.assertIn('{"winner": "A"}', debug)

    def test_grouped_path_missing_global_template_short_circuits_after_group_calls(self):
        jc = self._judge_config(global_verdict_template="")
        jc.group_verdicts = True
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text="Group verdict")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertIsNone(result)
        mock_llm.assert_called_once()  # only the per-group call, no global call

    def test_llm_exception_is_caught_and_returned_as_error_string(self):
        jc = self._judge_config()
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.side_effect = RuntimeError("provider down")
            result = generate_verdict(self.candidates, self.results, self.project_dir, jc)
        self.assertIsNone(result, "a failed verdict must be None, never text")



class TestVerdictFailuresAreNotPersisted(LoggerResetTestCase):
    """A verdict that failed must stay retryable.

    It used to be returned as an error string, which runner._generate_output
    appended to progress.jsonl as a "verdict" event; every later run and every
    render then resumed that error text instead of retrying the judge.
    """

    def setUp(self):
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="A", provider="google", model="m")]
        self.results = [_make_result("A", 8.0)]
        self.judge_config = JudgeConfig()
        self.judge_config.global_criteria = GlobalCriteria(mode="none")

    def _verdict_events(self) -> list[dict]:
        path = Path(self.project_dir) / "progress.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("type") == "verdict"
        ]

    def test_a_successful_verdict_is_stored(self):
        with patch("prompttestenv.runner.generate_verdict", return_value="Real verdict."), \
                patch("prompttestenv.runner.generate_html_report", return_value="/fake.html"):
            _generate_output(
                "html", self.candidates, self.results, self.project_dir,
                self.judge_config, GlobalCriteria(mode="none"), ProgressState(),
            )
        events = self._verdict_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["content"], "Real verdict.")

    def test_a_failed_verdict_is_not_stored(self):
        with patch("prompttestenv.runner.generate_verdict", return_value=None), \
                patch("prompttestenv.runner.generate_html_report") as mock_html:
            result = _generate_output(
                "html", self.candidates, self.results, self.project_dir,
                self.judge_config, GlobalCriteria(mode="none"), ProgressState(),
            )
        self.assertEqual(self._verdict_events(), [])
        self.assertIn("Error", result)
        mock_html.assert_not_called()


class TestVerdictReadsTheClientResult(unittest.TestCase):
    """generate_verdict must read LlmResult, not unpack a tuple.

    The unpacking outlived the LlmResult refactor and broke every verdict for a
    full development cycle, invisibly: the exception handler turned it into an
    error string that was then stored as the verdict. It survived because the
    mocks here still returned 4-tuples.
    """

    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="A", provider="google", model="m")]
        self.results = [_make_result("A", 8.0)]
        self.judge_config = JudgeConfig()
        self.judge_config.verdict_judge.verdict_template = "INSTRUCTIONS"
        self.judge_config.global_criteria = GlobalCriteria(mode="none")

    def test_the_models_text_reaches_the_verdict(self):
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(
                text="The real verdict.", output_tokens=42, reasoning_tokens=7
            )
            result = generate_verdict(
                self.candidates, self.results, self.project_dir, self.judge_config
            )
        self.assertEqual(result, "The real verdict.")

    def test_a_tuple_return_would_fail_loudly_now(self):
        """Documents the shape the mocks must keep: a tuple is no longer valid."""
        with patch("prompttestenv.verdict.get_llm_response") as mock_llm:
            mock_llm.return_value = ("text", 0, 0, "")
            result = generate_verdict(
                self.candidates, self.results, self.project_dir, self.judge_config
            )
        self.assertIsNone(result, "a stale tuple stub must not silently pass")


if __name__ == "__main__":
    unittest.main()
