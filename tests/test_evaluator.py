from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from prompttestenv.evaluator import run_evaluation_phase
from prompttestenv.models import CandidatePerformance, JudgeConfig, ProgressState, TestCaseResult


def _judge_config(timeout=5.0, eval_delay=0.0, reasoning_analysis=False):
    jc = JudgeConfig()
    jc.max_response_timeout_seconds = timeout
    jc.evaluation_delay_seconds = eval_delay
    jc.reasoning_analysis = reasoning_analysis
    return jc


def _pending_eval(cand_id="A", test_id="t1", rep=0, repetitions=1, output="response", reasoning_text=""):
    test_result = TestCaseResult(test_id=test_id, prompt="p", criteria="c")
    cand_perf = CandidatePerformance()
    test_result.candidates_perf[cand_id] = cand_perf
    return {
        "cand_id": cand_id,
        "cand_model": "m",
        "test_result": test_result,
        "cand_perf": cand_perf,
        "output": output,
        "reasoning_text": reasoning_text,
        "rep": rep,
        "repetitions": repetitions,
        "elapsed": 1.0,
    }, cand_perf


class TestRunEvaluationPhaseNormal(unittest.TestCase):
    def test_normal_path_records_score_and_appends_event(self):
        jc = _judge_config()
        task, cand_perf = _pending_eval()
        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event") as mock_append:
            mock_eval.return_value = {"score": 8, "reasoning": "good", "global_score": -1, "global_reasoning": "N/A"}
            run_evaluation_phase([task], jc, "/fake/project", ProgressState())

        self.assertEqual(cand_perf.scores, [8])
        self.assertEqual(cand_perf.best_output, "response")
        mock_append.assert_called_once()
        self.assertEqual(mock_append.call_args.args[1]["type"], "eval")


class TestRunEvaluationPhaseResume(unittest.TestCase):
    def test_resumed_entry_skips_judge_call(self):
        jc = _judge_config()
        task, cand_perf = _pending_eval()
        eval_event = {
            "type": "eval", "cand_id": "A", "test_id": "t1", "rep": 0,
            "score": 6, "global_score": -1, "reason": "resumed reason", "g_reason": "N/A",
        }
        progress = ProgressState(
            completed_eval={("A", "t1", 0)},
            events=[eval_event],
            eval_events={("A", "t1", 0): eval_event},
        )
        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event") as mock_append:
            run_evaluation_phase([task], jc, "/fake/project", progress)

        mock_eval.assert_not_called()
        mock_append.assert_not_called()
        self.assertEqual(cand_perf.scores, [6])
        self.assertEqual(cand_perf.best_reason, "resumed reason")


class TestRunEvaluationPhaseTimeout(unittest.TestCase):
    def test_timeout_kills_ollama_judge_and_records_sentinel_score(self):
        jc = _judge_config(timeout=0.05)
        jc.test_judge.provider = "ollama"
        jc.test_judge.model = "judge-model"
        task, cand_perf = _pending_eval()

        def slow_eval(*args, **kwargs):
            time.sleep(0.3)
            return {"score": 9}

        with patch("prompttestenv.evaluator.evaluate_with_judge", side_effect=slow_eval), \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event"), \
             patch("prompttestenv.api.subprocess.run") as mock_subprocess:
            run_evaluation_phase([task], jc, "/fake/project", ProgressState())

        self.assertEqual(cand_perf.scores, [0])
        mock_subprocess.assert_called_once()
        self.assertEqual(mock_subprocess.call_args.args[0], ["ollama", "stop", "judge-model"])


class TestRunEvaluationPhaseReasoning(unittest.TestCase):
    def test_reasoning_analysis_appended_when_enabled_and_trace_present(self):
        jc = _judge_config(reasoning_analysis=True)
        task, cand_perf = _pending_eval(reasoning_text="the model thought a lot")
        fake_stats = type("FakeStats", (), {"to_dict": lambda self: {"alt_path": 2}, "pure_reasoning_pct": 50.0, "alignment_score": 8})()

        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.analyze_reasoning") as mock_reasoning, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event"):
            mock_eval.return_value = {"score": 7, "reasoning": "ok", "global_score": -1, "global_reasoning": "N/A"}
            mock_reasoning.return_value = fake_stats
            run_evaluation_phase([task], jc, "/fake/project", ProgressState())

        self.assertEqual(cand_perf.reasoning_analyses, [{"alt_path": 2}])

    def test_reasoning_analysis_skipped_when_no_trace(self):
        jc = _judge_config(reasoning_analysis=True)
        task, cand_perf = _pending_eval(reasoning_text="")

        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.analyze_reasoning") as mock_reasoning, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event"):
            mock_eval.return_value = {"score": 7, "reasoning": "ok", "global_score": -1, "global_reasoning": "N/A"}
            run_evaluation_phase([task], jc, "/fake/project", ProgressState())

        mock_reasoning.assert_not_called()
        self.assertEqual(cand_perf.reasoning_analyses, [])


class TestRunEvaluationPhaseBestScoreTracking(unittest.TestCase):
    def test_only_updates_best_output_when_score_improves(self):
        jc = _judge_config()
        test_result = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        cand_perf = CandidatePerformance()
        test_result.candidates_perf["A"] = cand_perf

        task_low = {
            "cand_id": "A", "cand_model": "m", "test_result": test_result, "cand_perf": cand_perf,
            "output": "mediocre answer", "reasoning_text": "", "rep": 0, "repetitions": 2, "elapsed": 1.0,
        }
        task_high = {
            "cand_id": "A", "cand_model": "m", "test_result": test_result, "cand_perf": cand_perf,
            "output": "great answer", "reasoning_text": "", "rep": 1, "repetitions": 2, "elapsed": 1.0,
        }

        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event"):
            mock_eval.side_effect = [
                {"score": 4, "reasoning": "meh", "global_score": -1, "global_reasoning": "N/A"},
                {"score": 9, "reasoning": "excellent", "global_score": -1, "global_reasoning": "N/A"},
            ]
            run_evaluation_phase([task_low, task_high], jc, "/fake/project", ProgressState())

        self.assertEqual(cand_perf.best_output, "great answer")
        self.assertEqual(cand_perf.scores, [4, 9])


if __name__ == "__main__":
    unittest.main()
