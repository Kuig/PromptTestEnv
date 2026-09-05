from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from prompttestenv.evaluator import run_evaluation_phase
from prompttestenv.models import CandidatePerformance, JudgeConfig, ProgressState, TestCaseResult


def _judge_config(timeout=5.0, eval_delay=0.0, reasoning_analysis="none"):
    # Scope strings, not booleans: the field is assigned directly here (bypassing
    # _parse_reasoning_scope), and reasoning_enabled is `!= "none"` — so a bare
    # False would read as ENABLED, the opposite of how it looks.
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


class TestRunEvaluationPhaseRedoKeys(unittest.TestCase):
    """--retry-errors: a logged key listed in redo_keys is judged again."""

    def _progress(self):
        event = {
            "type": "eval", "cand_id": "A", "test_id": "t1", "rep": 0,
            "score": -1, "global_score": -1,
            "reason": "Error: judge down", "g_reason": "Error: judge down",
        }
        return ProgressState(
            completed_eval={("A", "t1", 0)},
            events=[event],
            eval_events={("A", "t1", 0): event},
        )

    def test_a_redo_key_is_judged_again(self):
        jc = _judge_config()
        task, cand_perf = _pending_eval()
        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event") as mock_append:
            mock_eval.return_value = {
                "score": 9, "reasoning": "good", "global_score": -1, "global_reasoning": "N/A",
            }
            run_evaluation_phase(
                [task], jc, "/fake/project", self._progress(), frozenset({("A", "t1", 0)})
            )

        mock_eval.assert_called_once()
        mock_append.assert_called_once()
        # One entry, carrying the new score rather than the superseded -1.
        self.assertEqual(cand_perf.scores, [9])

    def test_the_same_key_still_resumes_without_redo_keys(self):
        jc = _judge_config()
        task, cand_perf = _pending_eval()
        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event"):
            run_evaluation_phase([task], jc, "/fake/project", self._progress())

        mock_eval.assert_not_called()
        self.assertEqual(cand_perf.scores, [-1])


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

        self.assertEqual(cand_perf.scores, [-1])
        mock_subprocess.assert_called_once()
        self.assertEqual(mock_subprocess.call_args.args[0], ["ollama", "stop", "judge-model"])


class TestRunEvaluationPhaseReasoning(unittest.TestCase):
    """Reasoning analysis is its own phase now, driven off the stored traces.

    Keeping it out of evaluation is what lets it be re-run without re-judging,
    so the evaluation phase must not touch it even when it is enabled.
    """

    def test_evaluation_phase_does_not_analyze_reasoning(self):
        jc = _judge_config(reasoning_analysis="all")
        task, cand_perf = _pending_eval(reasoning_text="the model thought a lot")

        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.analysis.analyze_reasoning") as mock_reasoning, \
             patch("prompttestenv.evaluator.append_event"):
            mock_eval.return_value = {"score": 7, "reasoning": "ok", "global_score": -1, "global_reasoning": "N/A"}
            run_evaluation_phase([task], jc, "/fake/project", ProgressState())

        mock_reasoning.assert_not_called()
        self.assertEqual(cand_perf.reasoning_analyses, [])

    def test_eval_event_carries_no_reasoning_payload(self):
        jc = _judge_config(reasoning_analysis="all")
        task, _ = _pending_eval(reasoning_text="the model thought a lot")

        with patch("prompttestenv.evaluator.evaluate_with_judge") as mock_eval, \
             patch("prompttestenv.evaluator.preload_model_for_run"), \
             patch("prompttestenv.evaluator.append_event") as mock_append:
            mock_eval.return_value = {"score": 7, "reasoning": "ok", "global_score": -1, "global_reasoning": "N/A"}
            run_evaluation_phase([task], jc, "/fake/project", ProgressState())

        event = mock_append.call_args.args[1]
        self.assertEqual(event["type"], "eval")
        self.assertNotIn("reasoning_analysis", event)


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
