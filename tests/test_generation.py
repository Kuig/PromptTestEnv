from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from prompttestenv.generation import run_generation_phase
from prompttestenv.models import Candidate, JudgeConfig, ProgressState, TestCaseResult


def _judge_config(repetitions=1, timeout=5.0, rep_delay=0.0):
    jc = JudgeConfig()
    jc.repetitions = repetitions
    jc.max_response_timeout_seconds = timeout
    jc.repetition_delay_seconds = rep_delay
    return jc


def _candidate(name="A", provider="google"):
    return Candidate(name=name, provider=provider, model="m")


def _result():
    return TestCaseResult(test_id="t1", prompt="p", criteria="c")


class TestRunGenerationPhaseNormal(unittest.TestCase):
    def test_normal_path_records_output_and_appends_event(self):
        jc = _judge_config()
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.preload_model_for_run"), \
             patch("prompttestenv.generation.append_event") as mock_append:
            mock_llm.return_value = ("hello", 42, 0, "")
            progress = ProgressState()
            pending = run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["output"], "hello")
        self.assertEqual(results[0].candidates_perf["A"].tokens, [42])
        mock_append.assert_called_once()
        self.assertEqual(mock_append.call_args.args[1]["type"], "gen")


class TestRunGenerationPhaseResume(unittest.TestCase):
    def test_resumed_entry_skips_llm_call(self):
        jc = _judge_config()
        results = [_result()]
        progress = ProgressState(
            completed_gen={("A", "t1", 0)},
            events=[{
                "type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0,
                "output": "resumed output", "tokens": 7, "reasoning_tokens": 0,
                "reasoning_text": "", "elapsed": 0.5,
            }],
        )
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.preload_model_for_run"), \
             patch("prompttestenv.generation.append_event") as mock_append:
            pending = run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        mock_llm.assert_not_called()
        mock_append.assert_not_called()  # resumed entries are not re-logged
        self.assertEqual(pending[0]["output"], "resumed output")
        self.assertEqual(results[0].candidates_perf["A"].tokens, [7])


class TestRunGenerationPhaseTimeout(unittest.TestCase):
    def test_timeout_kills_ollama_model_and_records_sentinel(self):
        jc = _judge_config(timeout=0.05)
        results = [_result()]

        def slow_response(*args, **kwargs):
            time.sleep(0.3)
            return ("too slow", 0, 0, "")

        with patch("prompttestenv.generation.get_llm_response", side_effect=slow_response), \
             patch("prompttestenv.generation.preload_model_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.generation.subprocess.run") as mock_subprocess:
            progress = ProgressState()
            pending = run_generation_phase([_candidate(provider="ollama")], results, jc, "/fake/project", progress)

        self.assertEqual(pending[0]["output"], "⛔ [TIMEOUT EXCEEDED]")
        mock_subprocess.assert_called_once()
        self.assertEqual(mock_subprocess.call_args.args[0], ["ollama", "stop", "m"])

    def test_timeout_on_non_ollama_provider_does_not_kill_anything(self):
        jc = _judge_config(timeout=0.05)
        results = [_result()]

        def slow_response(*args, **kwargs):
            time.sleep(0.3)
            return ("too slow", 0, 0, "")

        with patch("prompttestenv.generation.get_llm_response", side_effect=slow_response), \
             patch("prompttestenv.generation.preload_model_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.generation.subprocess.run") as mock_subprocess:
            progress = ProgressState()
            run_generation_phase([_candidate(provider="google")], results, jc, "/fake/project", progress)

        mock_subprocess.assert_not_called()


class TestRunGenerationPhaseRepetitionDelay(unittest.TestCase):
    def test_sleeps_between_repetitions_when_delay_configured(self):
        jc = _judge_config(repetitions=2, rep_delay=3.0)
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.preload_model_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.generation.time.sleep") as mock_sleep:
            mock_llm.return_value = ("hi", 1, 0, "")
            progress = ProgressState()
            run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        # rep_delay applies after rep 0 (not after the last rep 1)
        mock_sleep.assert_called_once_with(3.0)

    def test_no_sleep_when_delay_is_zero(self):
        jc = _judge_config(repetitions=2, rep_delay=0.0)
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.preload_model_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.generation.time.sleep") as mock_sleep:
            mock_llm.return_value = ("hi", 1, 0, "")
            progress = ProgressState()
            run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
