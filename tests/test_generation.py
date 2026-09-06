from __future__ import annotations

import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from prompttestenv.generation import run_generation_phase
from prompttestenv.models import (
    Candidate,
    JudgeConfig,
    LlmResult,
    ProgressState,
    TestCaseResult,
)


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


def _result_with_media(test_id, *paths):
    return TestCaseResult(
        test_id=test_id, prompt="p", criteria="c", media_file_paths=list(paths)
    )


def _gen_event(cand_id, test_id, rep):
    return {
        "type": "gen", "cand_id": cand_id, "test_id": test_id, "rep": rep,
        "output": "resumed output", "tokens": 7, "reasoning_tokens": 0,
        "reasoning_text": "", "elapsed": 0.5,
    }


class TestRunGenerationPhaseNormal(unittest.TestCase):
    def test_normal_path_records_output_and_appends_event(self):
        jc = _judge_config()
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event") as mock_append:
            mock_llm.return_value = LlmResult(text="hello", output_tokens=42, reasoning_tokens=0, reasoning_text="")
            progress = ProgressState()
            pending = run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["output"], "hello")
        self.assertEqual(results[0].candidates_perf["A"].tokens, [42])
        mock_append.assert_called_once()
        self.assertEqual(mock_append.call_args.args[1]["type"], "gen")

    def test_generation_call_does_not_retry(self):
        jc = _judge_config()
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event"):
            mock_llm.return_value = LlmResult(text="hello", output_tokens=1)
            run_generation_phase([_candidate()], [_result()], jc, "/fake/project", ProgressState())

        self.assertEqual(mock_llm.call_args.kwargs["max_retries"], 0)


class TestRunGenerationPhaseResume(unittest.TestCase):
    def test_resumed_entry_skips_llm_call(self):
        jc = _judge_config()
        results = [_result()]
        gen_event = {
            "type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0,
            "output": "resumed output", "tokens": 7, "reasoning_tokens": 0,
            "reasoning_text": "", "elapsed": 0.5,
        }
        progress = ProgressState(
            completed_gen={("A", "t1", 0)},
            events=[gen_event],
            gen_events={("A", "t1", 0): gen_event},
        )
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event") as mock_append:
            pending = run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        mock_llm.assert_not_called()
        mock_append.assert_not_called()  # resumed entries are not re-logged
        self.assertEqual(pending[0]["output"], "resumed output")
        self.assertEqual(results[0].candidates_perf["A"].tokens, [7])


class TestRunGenerationPhaseRedoKeys(unittest.TestCase):
    """--retry-errors: a logged key listed in redo_keys is generated again."""

    def _progress(self):
        event = _gen_event("A", "t1", 0)
        return ProgressState(
            completed_gen={("A", "t1", 0)},
            events=[event],
            gen_events={("A", "t1", 0): event},
        )

    def test_a_redo_key_is_generated_again(self):
        jc = _judge_config()
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event") as mock_append:
            mock_llm.return_value = LlmResult(text="fresh", output_tokens=11)
            pending = run_generation_phase(
                [_candidate()], results, jc, "/fake/project", self._progress(),
                frozenset({("A", "t1", 0)}),
            )

        mock_llm.assert_called_once()
        mock_append.assert_called_once()
        self.assertEqual(pending[0]["output"], "fresh")
        # One entry, carrying the new value: the superseded event is not re-counted.
        self.assertEqual(results[0].candidates_perf["A"].tokens, [11])

    def test_the_same_key_still_resumes_without_redo_keys(self):
        jc = _judge_config()
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event"):
            pending = run_generation_phase(
                [_candidate()], results, jc, "/fake/project", self._progress()
            )

        mock_llm.assert_not_called()
        self.assertEqual(pending[0]["output"], "resumed output")

    def test_a_candidate_whose_only_work_is_a_redo_is_warmed_up(self):
        """Otherwise the SDK import and the uploads land on the retried call's elapsed."""
        jc = _judge_config()
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run") as mock_warm, \
             patch("prompttestenv.generation.append_event"):
            mock_llm.return_value = LlmResult(text="fresh", output_tokens=11)
            run_generation_phase(
                [_candidate()], results, jc, "/fake/project", self._progress(),
                frozenset({("A", "t1", 0)}),
            )

        mock_warm.assert_called_once()

    def test_a_candidate_with_nothing_to_do_is_not_warmed_up(self):
        jc = _judge_config()
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response"), \
             patch("prompttestenv.generation.warm_up_for_run") as mock_warm, \
             patch("prompttestenv.generation.append_event"):
            run_generation_phase(
                [_candidate()], results, jc, "/fake/project", self._progress()
            )

        mock_warm.assert_not_called()


class TestRunGenerationPhaseTimeout(unittest.TestCase):
    def test_timeout_kills_ollama_model_and_records_sentinel(self):
        jc = _judge_config(timeout=0.05)
        results = [_result()]

        def slow_response(*args, **kwargs):
            time.sleep(0.3)
            return ("too slow", 0, 0, "")

        with patch("prompttestenv.generation.get_llm_response", side_effect=slow_response), \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.api.subprocess.run") as mock_subprocess:
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
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.api.subprocess.run") as mock_subprocess:
            progress = ProgressState()
            run_generation_phase([_candidate(provider="google")], results, jc, "/fake/project", progress)

        mock_subprocess.assert_not_called()


class TestRunGenerationPhaseWarmUp(unittest.TestCase):
    """The warm-up must pay the one-off provider costs, and pay them only once."""

    def _run(self, candidates, results, progress, jc=None, enabled=True):
        jc = jc or _judge_config()
        cfg = SimpleNamespace(warmup=SimpleNamespace(enabled=enabled))
        with patch("prompttestenv.generation.get_llm_response") as mock_llm,              patch("prompttestenv.generation.warm_up_for_run") as mock_warm,              patch("prompttestenv.generation.get_app_config", return_value=cfg),              patch("prompttestenv.generation.append_event"):
            mock_llm.return_value = LlmResult(text="hi", output_tokens=1, reasoning_tokens=0, reasoning_text="")
            run_generation_phase(candidates, results, jc, "/fake/project", progress)
        return mock_warm

    def test_warms_up_once_per_candidate(self):
        mock_warm = self._run(
            [_candidate("A"), _candidate("B")], [_result()], ProgressState()
        )
        self.assertEqual(mock_warm.call_count, 2)
        self.assertEqual(
            [call.args[0:2] for call in mock_warm.call_args_list],
            [("google", "m"), ("google", "m")],
        )

    def test_passes_the_attachments_of_pending_tests(self):
        results = [
            _result_with_media("t1", "/abs/a.pdf"),
            _result_with_media("t2", "/abs/b.png"),
        ]
        mock_warm = self._run([_candidate("A")], results, ProgressState())
        self.assertEqual(mock_warm.call_args.args[2], ["/abs/a.pdf", "/abs/b.png"])

    def test_a_file_shared_by_two_tests_is_sent_once(self):
        results = [
            _result_with_media("t1", "/abs/same.pdf"),
            _result_with_media("t2", "/abs/same.pdf"),
        ]
        mock_warm = self._run([_candidate("A")], results, ProgressState())
        self.assertEqual(mock_warm.call_args.args[2], ["/abs/same.pdf"])

    def test_media_is_none_when_no_test_has_attachments(self):
        mock_warm = self._run([_candidate("A")], [_result()], ProgressState())
        self.assertIsNone(mock_warm.call_args.args[2])

    def test_skips_a_candidate_already_complete_in_the_log(self):
        """Nothing left to generate means nothing to warm up, and nothing to pay."""
        results = [_result_with_media("t1", "/abs/a.pdf")]
        event = _gen_event("A", "t1", 0)
        progress = ProgressState(
            completed_gen={("A", "t1", 0)},
            events=[event],
            gen_events={("A", "t1", 0): event},
        )
        mock_warm = self._run([_candidate("A")], results, progress)
        mock_warm.assert_not_called()

    def test_a_partially_resumed_candidate_still_warms_up(self):
        """The upload cache is per-process, so a resume always starts cold."""
        results = [_result_with_media("t1", "/abs/a.pdf")]
        event = _gen_event("A", "t1", 0)
        progress = ProgressState(
            completed_gen={("A", "t1", 0)},
            events=[event],
            gen_events={("A", "t1", 0): event},
        )
        mock_warm = self._run(
            [_candidate("A")], results, progress, jc=_judge_config(repetitions=2)
        )
        self.assertEqual(mock_warm.call_args.args[2], ["/abs/a.pdf"])

    def test_attachments_of_completed_tests_are_not_uploaded(self):
        results = [
            _result_with_media("done", "/abs/done.pdf"),
            _result_with_media("todo", "/abs/todo.pdf"),
        ]
        event = _gen_event("A", "done", 0)
        progress = ProgressState(
            completed_gen={("A", "done", 0)},
            events=[event],
            gen_events={("A", "done", 0): event},
        )
        mock_warm = self._run([_candidate("A")], results, progress)
        self.assertEqual(mock_warm.call_args.args[2], ["/abs/todo.pdf"])

    def test_disabled_in_config_means_no_call_at_all(self):
        mock_warm = self._run(
            [_candidate("A")], [_result()], ProgressState(), enabled=False
        )
        mock_warm.assert_not_called()

    def test_warm_up_time_is_not_counted_as_generation_time(self):
        results = [_result()]
        cfg = SimpleNamespace(warmup=SimpleNamespace(enabled=True))

        def slow_warm_up(*_args, **_kwargs):
            time.sleep(0.2)
            return True

        with patch("prompttestenv.generation.get_llm_response") as mock_llm,              patch("prompttestenv.generation.warm_up_for_run", side_effect=slow_warm_up),              patch("prompttestenv.generation.get_app_config", return_value=cfg),              patch("prompttestenv.generation.append_event"):
            mock_llm.return_value = LlmResult(text="hi", output_tokens=1, reasoning_tokens=0, reasoning_text="")
            run_generation_phase([_candidate("A")], results, _judge_config(), "/fake/project", ProgressState())

        self.assertLess(results[0].candidates_perf["A"].times[0], 0.2)


class TestRunGenerationPhaseRepetitionDelay(unittest.TestCase):
    def test_sleeps_between_repetitions_when_delay_configured(self):
        jc = _judge_config(repetitions=2, rep_delay=3.0)
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.generation.time.sleep") as mock_sleep:
            mock_llm.return_value = LlmResult(text="hi", output_tokens=1, reasoning_tokens=0, reasoning_text="")
            progress = ProgressState()
            run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        # rep_delay applies after rep 0 (not after the last rep 1)
        mock_sleep.assert_called_once_with(3.0)

    def test_no_sleep_when_delay_is_zero(self):
        jc = _judge_config(repetitions=2, rep_delay=0.0)
        results = [_result()]
        with patch("prompttestenv.generation.get_llm_response") as mock_llm, \
             patch("prompttestenv.generation.warm_up_for_run"), \
             patch("prompttestenv.generation.append_event"), \
             patch("prompttestenv.generation.time.sleep") as mock_sleep:
            mock_llm.return_value = LlmResult(text="hi", output_tokens=1, reasoning_tokens=0, reasoning_text="")
            progress = ProgressState()
            run_generation_phase([_candidate()], results, jc, "/fake/project", progress)

        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
