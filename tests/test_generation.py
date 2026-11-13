from __future__ import annotations

import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from prompttestenv.api import ModelRefusalError, UnsupportedFileError
from prompttestenv.generation import run_generation_phase
from prompttestenv.progress import (
    GEN_ERROR_TEXT, GEN_TIMEOUT_TEXT, GEN_UNSUPPORTED_TEXT, ProgressState, is_failed_gen,
)
from testutils import ResidencyResetTestCase
from prompttestenv.models import (
    Candidate,
    JudgeConfig,
    LlmResult,
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
        self.assertEqual(pending[0].output, "hello")
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
        self.assertEqual(pending[0].output, "resumed output")
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
        self.assertEqual(pending[0].output, "fresh")
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
        self.assertEqual(pending[0].output, "resumed output")

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


class TestRunGenerationPhaseTimeout(ResidencyResetTestCase):
    def _timed_out_run(self, provider):
        jc = _judge_config(timeout=0.05)
        results = [_result()]

        def slow_response(*args, **kwargs):
            time.sleep(0.3)
            return ("too slow", 0, 0, "")

        with patch("prompttestenv.generation.get_llm_response", side_effect=slow_response),              patch("prompttestenv.generation.warm_up_for_run"),              patch("prompttestenv.generation.append_event"),              patch("prompttestenv.api.unload_model", autospec=True) as mock_unload:
            progress = ProgressState()
            pending = run_generation_phase(
                [_candidate(provider=provider)], results, jc, "/fake/project", progress
            )
        return pending, mock_unload

    def test_timeout_records_the_sentinel(self):
        pending, _ = self._timed_out_run("ollama")
        self.assertEqual(pending[0].output, "⛔ [TIMEOUT EXCEEDED]")

    def test_timeout_does_not_evict_the_local_model(self):
        """The reload would be charged to the next repetition's own elapsed.

        Warm-up runs once per candidate, outside the timed block, so evicting a
        model here makes the next call of that same candidate pay to load it
        again inside the measurement. Nothing leaks by keeping it: the candidate
        boundary and teardown release it either way.
        """
        _, mock_unload = self._timed_out_run("ollama")
        mock_unload.assert_not_called()

    def test_timeout_does_not_evict_a_remote_model_either(self):
        _, mock_unload = self._timed_out_run("google")
        mock_unload.assert_not_called()


class TestRunGenerationPhaseProviderError(ResidencyResetTestCase):
    """A provider that raises must not end the benchmark.

    call_ai raises once its retries are spent, and this phase passes
    GENERATION_MAX_RETRIES = 0, so before this a single 429 ended the whole run
    with a traceback. No test covered the path, which is why the gap survived.
    """

    def _failed_run(self):
        jc = _judge_config()
        results = [_result()]
        events = []

        def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        with patch("prompttestenv.generation.get_llm_response", side_effect=boom),              patch("prompttestenv.generation.warm_up_for_run"),              patch("prompttestenv.generation.append_event", side_effect=lambda _d, e: events.append(e)):
            pending = run_generation_phase(
                [_candidate()], results, jc, "/fake/project", ProgressState()
            )
        return pending, events

    def test_the_run_survives_and_stores_the_error_placeholder(self):
        pending, events = self._failed_run()
        self.assertEqual(pending[0].output, GEN_ERROR_TEXT)
        self.assertEqual(events[0]["output"], GEN_ERROR_TEXT)

    def test_the_placeholder_is_not_the_timeout_one(self):
        """Reusing it would make the log claim a timeout for a 429."""
        _, events = self._failed_run()
        self.assertNotEqual(events[0]["output"], GEN_TIMEOUT_TEXT)

    def test_the_exception_is_kept_out_of_the_payload_but_not_lost(self):
        """output is compared by equality and is what the judge scores, so the
        message goes in its own field."""
        _, events = self._failed_run()
        self.assertNotIn("provider down", events[0]["output"])
        self.assertIn("provider down", events[0]["error"])

    def test_elapsed_is_not_measured_rather_than_a_fast_failure_time(self):
        """A 429 returned in milliseconds measures nothing about the model, and
        calculate_stats filters anything below zero out of time_mean."""
        _, events = self._failed_run()
        self.assertEqual(events[0]["elapsed"], -1.0)

    def test_the_stored_event_is_retryable(self):
        _, events = self._failed_run()
        self.assertTrue(is_failed_gen(events[0]))

    def test_tokens_are_not_measured_rather_than_zero(self):
        """No call completed, so a 0 would drag tokens_mean down as a real count."""
        pending, events = self._failed_run()
        self.assertEqual((events[0]["tokens"], events[0]["reasoning_tokens"]), (-1, -1))
        perf = pending[0].cand_perf
        self.assertEqual((perf.tokens_mean, perf.reasoning_tokens_mean), (0.0, 0.0))
        self.assertEqual(len(perf.tokens), 1)  # kept, so the zip-by-index stays aligned


def _raising_run(exc):
    """Run one generation whose call raises `exc`; return (pending, events)."""
    events = []

    def raise_it(*args, **kwargs):
        raise exc

    with patch("prompttestenv.generation.get_llm_response", side_effect=raise_it), \
         patch("prompttestenv.generation.warm_up_for_run"), \
         patch("prompttestenv.generation.append_event",
               side_effect=lambda _d, e: events.append(e)):
        pending = run_generation_phase(
            [_candidate()], [_result()], _judge_config(), "/fake/project", ProgressState()
        )
    return pending, events


class TestRunGenerationPhaseRefusal(ResidencyResetTestCase):
    """A refusal is a measured outcome, not a failure to obtain one."""

    def _refused(self, partial=""):
        return _raising_run(ModelRefusalError("anthropic", "cyber", "policy", partial))

    def test_the_partial_text_is_the_response(self):
        pending, events = self._refused("I can only say this much")
        self.assertEqual(pending[0].output, "I can only say this much")
        self.assertEqual(events[0]["output"], "I can only say this much")

    def test_the_event_is_flagged_and_keeps_the_reason(self):
        _, events = self._refused()
        self.assertIs(events[0]["refused"], True)
        self.assertIn("cyber", events[0]["error"])

    def test_it_is_counted_on_the_live_record(self):
        pending, _ = self._refused()
        self.assertEqual(pending[0].cand_perf.refusals, 1)

    def test_its_time_is_a_measurement(self):
        _, events = self._refused()
        self.assertGreaterEqual(events[0]["elapsed"], 0.0)

    def test_it_is_not_retried(self):
        _, events = self._refused()
        self.assertFalse(is_failed_gen(events[0]))


class TestRunGenerationPhaseUnsupportedAttachment(ResidencyResetTestCase):
    """An attachment the provider cannot take skips that test for the candidate."""

    def _skipped(self):
        return _raising_run(UnsupportedFileError("ollama cannot take clip.mp4"))

    def test_stores_its_own_placeholder(self):
        pending, events = self._skipped()
        self.assertEqual(pending[0].output, GEN_UNSUPPORTED_TEXT)
        self.assertEqual(events[0]["output"], GEN_UNSUPPORTED_TEXT)
        self.assertIn("clip.mp4", events[0]["error"])

    def test_measures_nothing(self):
        _, events = self._skipped()
        self.assertEqual(
            (events[0]["elapsed"], events[0]["tokens"], events[0]["reasoning_tokens"]),
            (-1.0, -1, -1),
        )

    def test_is_not_retried(self):
        """The same request would be refused the same way, before sending."""
        _, events = self._skipped()
        self.assertFalse(is_failed_gen(events[0]))
        self.assertNotIn("refused", events[0])


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
