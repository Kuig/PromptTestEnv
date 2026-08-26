from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from prompttestenv.models import (
    CandidatePerformance,
    Candidate,
    TestCaseResult,
    pool_by_candidate,
    GlobalCriteria,
    JudgeConfig,
    ProgressState,
    TestCase,
    calculate_stats,
    compute_cost_per_point,
)
from testutils import LoggerResetTestCase


class TestCalculateStats(unittest.TestCase):
    def test_empty_list_returns_default(self):
        mean, std = calculate_stats([], default_val=-1.0)
        self.assertEqual(mean, -1.0)
        self.assertEqual(std, 0.0)

    def test_negative_values_filtered_out(self):
        mean, std = calculate_stats([-1, -1, 8.0])
        self.assertEqual(mean, 8.0)
        self.assertEqual(std, 0.0)

    def test_single_value_has_zero_std(self):
        mean, std = calculate_stats([7.0])
        self.assertEqual(mean, 7.0)
        self.assertEqual(std, 0.0)

    def test_normal_case(self):
        mean, std = calculate_stats([2.0, 4.0, 6.0])
        self.assertAlmostEqual(mean, 4.0)
        self.assertGreater(std, 0.0)


class TestCandidatePerformance(unittest.TestCase):
    def test_defaults_are_empty(self):
        perf = CandidatePerformance()
        self.assertEqual(perf.score_mean, 0.0)
        self.assertEqual(perf.score_std, 0.0)

    def test_global_score_mean_defaults_to_negative_one(self):
        perf = CandidatePerformance()
        self.assertEqual(perf.global_score_mean, -1.0)

    def test_mean_std_reflect_appended_values(self):
        perf = CandidatePerformance()
        perf.scores.extend([6.0, 8.0])
        perf.tokens.extend([100, 200])
        self.assertAlmostEqual(perf.score_mean, 7.0)
        self.assertAlmostEqual(perf.tokens_mean, 150.0)


class TestCandidateLoadAll(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Candidate.load_all(self.project_dir)

    def test_loads_defaults_when_fields_omitted(self):
        cand_file = Path(self.project_dir) / "candidates.json"
        cand_file.write_text(json.dumps([{"name": "A", "model": "m"}]), encoding="utf-8")
        candidates = Candidate.load_all(self.project_dir)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].provider, "google")
        self.assertEqual(candidates[0].temperature, 0.7)
        self.assertIsNone(candidates[0].resolved_system_instruction)

    def test_resolves_system_prompt_file(self):
        sys_dir = Path(self.project_dir) / "system_prompts"
        sys_dir.mkdir()
        (sys_dir / "p.txt").write_text("Be nice.", encoding="utf-8")
        cand_file = Path(self.project_dir) / "candidates.json"
        cand_file.write_text(
            json.dumps([{"name": "A", "model": "m", "system_prompt_file": "p.txt"}]),
            encoding="utf-8",
        )
        candidates = Candidate.load_all(self.project_dir)
        self.assertEqual(candidates[0].resolved_system_instruction, "Be nice.")

    def test_missing_referenced_system_prompt_file_warns_but_does_not_raise(self):
        cand_file = Path(self.project_dir) / "candidates.json"
        cand_file.write_text(
            json.dumps([{"name": "A", "model": "m", "system_prompt_file": "missing.txt"}]),
            encoding="utf-8",
        )
        candidates = Candidate.load_all(self.project_dir)
        self.assertIsNone(candidates[0].resolved_system_instruction)


class TestTestCaseLoadAll(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            TestCase.load_all(self.project_dir)

    def test_defaults_applied(self):
        test_file = Path(self.project_dir) / "test_cases.json"
        test_file.write_text(json.dumps([{"id": "t1", "prompt": "p", "criteria": "c"}]), encoding="utf-8")
        cases = TestCase.load_all(self.project_dir)
        self.assertEqual(cases[0].group, "Default group")
        self.assertEqual(cases[0].judge_type, "llm-judge")
        self.assertIsNone(cases[0].file)


class TestGlobalCriteriaLoad(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_missing_file_degrades_to_none_mode(self):
        gc = GlobalCriteria.load(self.project_dir)
        self.assertEqual(gc.mode, "none")

    def test_corrupted_json_degrades_to_none_mode(self):
        gc_file = Path(self.project_dir) / "global_criteria.json"
        gc_file.write_text("{not valid json", encoding="utf-8")
        gc = GlobalCriteria.load(self.project_dir)
        self.assertEqual(gc.mode, "none")

    def test_valid_file_loads_correctly(self):
        gc_file = Path(self.project_dir) / "global_criteria.json"
        gc_file.write_text(json.dumps({"mode": "assert", "assert_criteria": "s: (10,'ok') if s else (1,'no')"}), encoding="utf-8")
        gc = GlobalCriteria.load(self.project_dir)
        self.assertEqual(gc.mode, "assert")
        self.assertEqual(gc.assert_criteria, "s: (10,'ok') if s else (1,'no')")

    def test_to_verdict_string_branches(self):
        self.assertIn("Cosine similarity", GlobalCriteria(mode="similarity", similarity_criteria="X").to_verdict_string())
        self.assertIn("assertion", GlobalCriteria(mode="assert", assert_criteria="X").to_verdict_string())
        self.assertEqual(GlobalCriteria(mode="llm-judge", llm_judge_criteria="be nice").to_verdict_string(), "be nice")
        self.assertEqual(GlobalCriteria(mode="none").to_verdict_string(), "Global evaluation disabled.")


class TestReasoningScopeParsing(unittest.TestCase):
    """reasoning_analysis is a three-way scope, and a bad value must not enable it."""

    def test_the_three_scopes_round_trip(self):
        for scope in ("none", "best", "all"):
            with self.subTest(scope=scope):
                jc = JudgeConfig.from_dict({"reasoning_analysis": scope})
                self.assertEqual(jc.reasoning_analysis, scope)

    def test_scope_is_case_insensitive(self):
        jc = JudgeConfig.from_dict({"reasoning_analysis": "BEST"})
        self.assertEqual(jc.reasoning_analysis, "best")

    def test_missing_key_disables_analysis(self):
        self.assertFalse(JudgeConfig.from_dict({}).reasoning_enabled)

    def test_unknown_scope_disables_analysis_with_a_warning(self):
        """It must not fall through to a truthiness test, which any string passes."""
        with patch("prompttestenv.logger.log_warning") as mock_warning:
            jc = JudgeConfig.from_dict({"reasoning_analysis": "sometimes"})
        self.assertEqual(jc.reasoning_analysis, "none")
        self.assertFalse(jc.reasoning_enabled)
        mock_warning.assert_called_once()

    def test_old_boolean_spelling_still_loads_and_warns(self):
        for raw, expected in ((True, "all"), (False, "none")):
            with self.subTest(raw=raw):
                with patch("prompttestenv.logger.log_warning") as mock_warning:
                    jc = JudgeConfig.from_dict({"reasoning_analysis": raw})
                self.assertEqual(jc.reasoning_analysis, expected)
                mock_warning.assert_called_once()


class TestJudgeConfigFromDict(unittest.TestCase):
    def test_empty_dict_uses_all_defaults(self):
        jc = JudgeConfig.from_dict({})
        self.assertEqual(jc.repetitions, 5)
        self.assertEqual(jc.test_judge.provider, "google")
        self.assertEqual(jc.global_criteria.mode, "llm-judge")

    def test_global_criteria_as_dict(self):
        jc = JudgeConfig.from_dict({"global_criteria": {"mode": "assert", "assert_criteria": "x"}})
        self.assertEqual(jc.global_criteria.mode, "assert")
        self.assertEqual(jc.global_criteria.assert_criteria, "x")

    def test_global_criteria_as_instance_passthrough(self):
        gc = GlobalCriteria(mode="similarity", similarity_criteria="y")
        jc = JudgeConfig.from_dict({"global_criteria": gc})
        self.assertIs(jc.global_criteria, gc)

    def test_global_criteria_scalar_fallback(self):
        jc_truthy = JudgeConfig.from_dict({"global_criteria": "be polite"})
        self.assertEqual(jc_truthy.global_criteria.mode, "llm-judge")
        self.assertEqual(jc_truthy.global_criteria.llm_judge_criteria, "be polite")

        jc_falsy = JudgeConfig.from_dict({"global_criteria": ""})
        self.assertEqual(jc_falsy.global_criteria.mode, "none")

    def test_reasoning_judge_keeps_only_call_parameters(self):
        """Prompts and taxonomy live in config.json, so judge_config only picks the judge."""
        jc = JudgeConfig.from_dict({
            "reasoning_judge": {
                "provider": "ollama",
                "model": "qwen",
                "context_size": 32000,
                "dimension_mode": "joint",
            }
        })
        self.assertEqual(jc.reasoning_judge.provider, "ollama")
        self.assertEqual(jc.reasoning_judge.context_size, 32000)
        self.assertEqual(jc.reasoning_judge.dimension_mode, "joint")

    def test_reasoning_judge_knobs_default_to_none_so_config_json_decides(self):
        jc = JudgeConfig.from_dict({"reasoning_judge": {}})
        self.assertIsNone(jc.reasoning_judge.dimension_mode)
        self.assertIsNone(jc.reasoning_judge.reliability_k)
        self.assertIsNone(jc.reasoning_judge.max_units_per_call)

    def test_retired_reasoning_prompt_keys_are_ignored(self):
        """Old project configs still load; their dead prompt keys are simply dropped."""
        jc = JudgeConfig.from_dict({
            "reasoning_judge": {
                "model": "m",
                "segmentation_template": "dead",
                "metrics_template": "dead",
                "reasoning_system_prompt": "dead",
            }
        })
        self.assertEqual(jc.reasoning_judge.model, "m")
        self.assertFalse(hasattr(jc.reasoning_judge, "segmentation_template"))


class TestJudgeConfigLoad(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            JudgeConfig.load(self.project_dir)

    def test_valid_file_loads(self):
        cfg_file = Path(self.project_dir) / "judge_config.json"
        cfg_file.write_text(json.dumps({"repetitions": 3}), encoding="utf-8")
        jc = JudgeConfig.load(self.project_dir)
        self.assertEqual(jc.repetitions, 3)


class TestProgressStateLoad(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        # ProgressState.load hashes candidates/judge_config/test_cases/global_criteria.json,
        # so give it something deterministic to hash even though it doesn't validate content.
        for name in ("candidates.json", "judge_config.json", "test_cases.json", "global_criteria.json"):
            (Path(self.project_dir) / name).write_text("{}", encoding="utf-8")

    def test_no_existing_file_creates_one_with_meta_line(self):
        state = ProgressState.load(self.project_dir)
        self.assertTrue(state.hash_match)
        self.assertEqual(state.events, [])
        progress_file = Path(self.project_dir) / "progress.jsonl"
        self.assertTrue(progress_file.exists())
        meta = json.loads(progress_file.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta["type"], "meta")

    def test_hash_mismatch_renames_to_bak(self):
        ProgressState.load(self.project_dir)  # creates progress.jsonl with current hash
        # Now change a config file so the stored hash no longer matches.
        (Path(self.project_dir) / "candidates.json").write_text('{"changed": true}', encoding="utf-8")
        state = ProgressState.load(self.project_dir)
        self.assertFalse(state.hash_match)
        self.assertTrue((Path(self.project_dir) / "progress.jsonl.bak").exists())

    def test_force_restart_discards_existing_progress(self):
        ProgressState.load(self.project_dir)
        progress_file = Path(self.project_dir) / "progress.jsonl"
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0, "output": "x", "tokens": 1, "reasoning_tokens": 0, "elapsed": 0.1}) + "\n")

        state = ProgressState.load(self.project_dir, force_restart=True)
        self.assertEqual(state.events, [])

    def test_completed_gen_and_eval_sets_populated(self):
        progress_file = Path(self.project_dir) / "progress.jsonl"
        from prompttestenv.progress import calculate_config_hash
        current_hash = calculate_config_hash(self.project_dir)
        lines = [
            json.dumps({"type": "meta", "config_hash": current_hash}),
            json.dumps({"type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0, "output": "x", "tokens": 1, "reasoning_tokens": 0, "elapsed": 0.1}),
            json.dumps({"type": "eval", "cand_id": "A", "test_id": "t1", "rep": 0, "score": 8, "global_score": -1, "reason": "ok", "g_reason": "N/A"}),
        ]
        progress_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        state = ProgressState.load(self.project_dir)
        self.assertIn(("A", "t1", 0), state.completed_gen)
        self.assertIn(("A", "t1", 0), state.completed_eval)
        self.assertEqual(len(state.events), 2)

    def test_trailing_corrupted_line_is_ignored(self):
        progress_file = Path(self.project_dir) / "progress.jsonl"
        from prompttestenv.progress import calculate_config_hash
        current_hash = calculate_config_hash(self.project_dir)
        progress_file.write_text(
            json.dumps({"type": "meta", "config_hash": current_hash}) + "\n" + "{not valid json\n",
            encoding="utf-8",
        )
        state = ProgressState.load(self.project_dir)
        self.assertTrue(state.hash_match)
        self.assertEqual(state.events, [])



class TestReasoningCostPerPoint(unittest.TestCase):
    """Thinking tokens per point of task score, with a sentinel instead of a crash."""

    def _perf(self, scores, thinking) -> CandidatePerformance:
        perf = CandidatePerformance()
        perf.scores.extend(scores)
        perf.reasoning_tokens.extend(thinking)
        return perf

    def test_divides_thinking_by_score(self):
        self.assertAlmostEqual(self._perf([8.0], [200]).reasoning_cost_per_point, 25.0)

    def test_uses_the_means_not_the_totals(self):
        perf = self._perf([8.0, 8.0], [100, 300])
        self.assertAlmostEqual(perf.reasoning_cost_per_point, 25.0)

    def test_not_measured_without_thinking_tokens(self):
        self.assertEqual(self._perf([8.0], []).reasoning_cost_per_point, -1.0)
        self.assertEqual(self._perf([8.0], [0]).reasoning_cost_per_point, -1.0)

    def test_not_measured_without_a_score(self):
        """A zero mean score must not raise, and must not read as free thinking."""
        self.assertEqual(self._perf([], [200]).reasoning_cost_per_point, -1.0)
        self.assertEqual(self._perf([0.0], [200]).reasoning_cost_per_point, -1.0)

    def test_empty_record_is_not_measured(self):
        self.assertEqual(CandidatePerformance().reasoning_cost_per_point, -1.0)


class TestComputeCostPerPoint(unittest.TestCase):
    """The shared arithmetic behind both the pooled and the per-repetition cost figure."""

    def test_divides_numerator_by_denominator(self):
        self.assertAlmostEqual(compute_cost_per_point(200, 8), 25.0)

    def test_not_measured_when_numerator_is_zero_or_negative(self):
        self.assertEqual(compute_cost_per_point(0, 8), -1.0)
        self.assertEqual(compute_cost_per_point(-5, 8), -1.0)

    def test_not_measured_when_denominator_is_zero_or_negative(self):
        self.assertEqual(compute_cost_per_point(200, 0), -1.0)
        self.assertEqual(compute_cost_per_point(200, -1), -1.0)

    def test_mean_of_ratios_diverges_from_ratio_of_means(self):
        """The whole reason the per-repetition and pooled figures can disagree."""
        pooled = compute_cost_per_point((300 + 300) / 2, (10 + 1) / 2)
        mean_of_ratios = statistics.mean(
            [compute_cost_per_point(300, 10), compute_cost_per_point(300, 1)]
        )
        self.assertAlmostEqual(pooled, 54.5454545, places=5)
        self.assertAlmostEqual(mean_of_ratios, 165.0)
        self.assertNotAlmostEqual(pooled, mean_of_ratios, places=0)


class TestMeanCostPerPoint(unittest.TestCase):
    """The mean of each repetition's own ratio, always available (no reasoning analysis needed).

    The property this is really about: run_analysis_phase never touches these
    lists, so mean_cost_per_point/std_cost_per_point work identically whether
    or not judge_config.reasoning_analysis is enabled at all.
    """

    def _perf(self, thinking: list[int], scores: list[float]) -> CandidatePerformance:
        perf = CandidatePerformance()
        perf.reasoning_tokens.extend(thinking)
        perf.scores.extend(scores)
        return perf

    def test_single_repetition_matches_the_pooled_figure(self):
        perf = self._perf([200], [8.0])
        self.assertAlmostEqual(perf.mean_cost_per_point, 25.0)
        self.assertAlmostEqual(perf.reasoning_cost_per_point, 25.0)

    def test_diverges_from_the_pooled_figure_when_score_varies(self):
        """The exact 300/10 vs 300/1 example: mean of ratios is 165, not 54.5."""
        perf = self._perf([300, 300], [10.0, 1.0])
        self.assertAlmostEqual(perf.mean_cost_per_point, 165.0)
        self.assertAlmostEqual(perf.reasoning_cost_per_point, 54.5454545, places=5)

    def test_std_is_the_spread_of_the_ratios_not_of_the_raw_lists(self):
        perf = self._perf([300, 300], [10.0, 1.0])
        self.assertGreater(perf.std_cost_per_point, 0.0)

    def test_constant_ratio_gives_zero_std(self):
        """Same ratio each time (12.5), even though neither tokens nor score is constant."""
        perf = self._perf([100, 200], [8.0, 16.0])
        self.assertAlmostEqual(perf.std_cost_per_point, 0.0)

    def test_not_measured_without_any_repetition(self):
        """calculate_stats' own convention: std defaults to 0.0, not the mean's sentinel."""
        perf = CandidatePerformance()
        self.assertEqual(perf.mean_cost_per_point, -1.0)
        self.assertEqual(perf.std_cost_per_point, 0.0)

    def test_unmeasurable_repetitions_are_excluded_not_averaged_as_zero(self):
        """A rep with no thinking tokens must not drag the mean toward zero."""
        perf = self._perf([0, 300], [8.0, 6.0])
        self.assertAlmostEqual(perf.mean_cost_per_point, 50.0)

    def test_safe_against_unequal_length_lists(self):
        """An interrupted run: scores is a prefix of reasoning_tokens. zip() truncates."""
        perf = self._perf([100, 200, 300], [8.0])
        self.assertAlmostEqual(perf.mean_cost_per_point, 12.5)


class TestCombinedAvg(unittest.TestCase):
    def test_averages_task_and_global(self):
        perf = CandidatePerformance()
        perf.scores.append(8.0)
        perf.global_scores.append(6.0)
        self.assertAlmostEqual(perf.combined_avg, 7.0)

    def test_falls_back_to_task_alone_when_global_is_disabled(self):
        """A global mean of -1 means "not computed", not a bad mark."""
        perf = CandidatePerformance()
        perf.scores.append(8.0)
        self.assertAlmostEqual(perf.combined_avg, 8.0)


class TestPoolByCandidate(unittest.TestCase):
    """The single aggregation the report, the verdict and winner_only all read."""

    def setUp(self):
        self.candidates = [
            Candidate(name="A", provider="p", model="m"),
            Candidate(name="B", provider="p", model="m"),
        ]

    def _row(self, test_id: str, score: float) -> TestCaseResult:
        row = TestCaseResult(test_id=test_id, prompt="p", criteria="c")
        perf = CandidatePerformance()
        perf.scores.append(score)
        perf.global_scores.append(score)
        perf.times.append(1.0)
        perf.tokens.append(10)
        perf.reasoning_tokens.append(20)
        perf.reasoning_analyses.append({"avg_density": 1.0})
        row.candidates_perf["A"] = perf
        return row

    def test_pools_every_test_case_into_one_record(self):
        pooled = pool_by_candidate(self.candidates, [self._row("t1", 6.0), self._row("t2", 8.0)])
        self.assertAlmostEqual(pooled["A"].score_mean, 7.0)
        self.assertEqual(len(pooled["A"].tokens), 2)
        self.assertEqual(len(pooled["A"].times), 2)

    def test_pools_the_reasoning_analyses_too(self):
        """Their absence is why the verdict's aggregate carried no profile."""
        pooled = pool_by_candidate(self.candidates, [self._row("t1", 6.0), self._row("t2", 8.0)])
        self.assertEqual(len(pooled["A"].reasoning_analyses), 2)

    def test_candidate_without_data_gets_an_empty_record(self):
        pooled = pool_by_candidate(self.candidates, [self._row("t1", 6.0)])
        self.assertIn("B", pooled)
        self.assertEqual(pooled["B"].scores, [])
        self.assertEqual(pooled["B"].score_mean, 0.0)

    def test_does_not_mutate_the_source_records(self):
        rows = [self._row("t1", 6.0)]
        pool_by_candidate(self.candidates, rows)
        pool_by_candidate(self.candidates, rows)
        self.assertEqual(len(rows[0].candidates_perf["A"].scores), 1)


if __name__ == "__main__":
    unittest.main()
