from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv.models import (
    CandidatePerformance,
    Candidate,
    GlobalCriteria,
    JudgeConfig,
    ProgressState,
    TestCase,
    calculate_stats,
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


if __name__ == "__main__":
    unittest.main()
