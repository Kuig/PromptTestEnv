from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from prompttestenv.api import (
    LOCAL_KEEP_ALIVE,
    PROVIDER_TIMEOUT_BUFFER_SECONDS,
    call_with_timeout,
    configure_ai_logging,
    cosine_similarity,
    get_llm_response,
    get_text_embedding,
    parse_json_object,
    preload_model_for_run,
    release_previous_candidate,
    warm_up_for_run,
    teardown,
)
from unified_ai_client import AiResponse
from testutils import ResidencyResetTestCase


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_empty_vectors_return_zero(self):
        self.assertEqual(cosine_similarity([], []), 0.0)

    def test_mismatched_lengths_return_zero(self):
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestGetLlmResponse(unittest.TestCase):
    def _fake_response(self, text="hi", out_tok=10, r_tok=0, reasoning="", is_summary=False):
        """spec=AiResponse, so the mock has the real class's attributes only.

        A bare MagicMock answers to anything, so a field renamed in
        UnifiedAiClient would have left this whole suite green and surfaced
        only in production. get_llm_response is the single place this project
        touches AiResponse, which makes this the one boundary worth typing.
        """
        resp = MagicMock(spec=AiResponse)
        resp.text = text
        resp.output_tokens = out_tok
        resp.reasoning_tokens = r_tok
        resp.reasoning_text = reasoning
        resp.reasoning_is_summary = is_summary
        return resp

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_returns_populated_result_from_response(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response(
            text="hello", out_tok=5, r_tok=2, reasoning="thinking...", is_summary=True
        )
        result = get_llm_response("google", "gemini", None, "prompt")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.reasoning_tokens, 2)
        self.assertEqual(result.reasoning_text, "thinking...")
        self.assertTrue(result.reasoning_is_summary)

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_none_reasoning_text_becomes_empty_string(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response(reasoning=None)
        self.assertEqual(get_llm_response("google", "gemini", None, "prompt").reasoning_text, "")

    def test_every_field_this_module_reads_exists_on_the_real_response(self):
        """The guard the mock alone cannot be: if UnifiedAiClient renames one of
        these, this fails here instead of silently changing a report."""
        for name in ("text", "output_tokens", "reasoning_tokens",
                     "reasoning_text", "reasoning_is_summary"):
            with self.subTest(name):
                self.assertTrue(hasattr(AiResponse(text=""), name))

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_a_missing_attribute_is_not_papered_over(self, mock_call_ai):
        """reasoning_is_summary is read directly now. A defensive getattr would
        turn a rename into "raw" on every trace, which is the one reading the
        report must never invent."""
        resp = self._fake_response(reasoning="raw trace")
        del resp.reasoning_is_summary
        mock_call_ai.return_value = resp
        with self.assertRaises(AttributeError):
            get_llm_response("google", "gemini", None, "prompt")

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_thinking_string_true_normalized_to_bool(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", thinking="true")
        self.assertIs(mock_call_ai.call_args.kwargs["thinking"], True)

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_thinking_string_false_normalized_to_bool(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", thinking="false")
        self.assertIs(mock_call_ai.call_args.kwargs["thinking"], False)

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_thinking_default_string_stays_default(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", thinking="default")
        self.assertEqual(mock_call_ai.call_args.kwargs["thinking"], "default")

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_disable_safety_sets_extra_option(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", disable_safety=True)
        self.assertEqual(mock_call_ai.call_args.kwargs["extra_options"], {"disable_safety": True})

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_disable_safety_false_leaves_extra_options_empty(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", disable_safety=False)
        self.assertEqual(mock_call_ai.call_args.kwargs["extra_options"], {})

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_provider_is_lowercased(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("GOOGLE", "gemini", None, "prompt")
        self.assertEqual(mock_call_ai.call_args.kwargs["provider"], "google")

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_json_mime_type_enables_format_json(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", response_mime_type="application/json")
        self.assertTrue(mock_call_ai.call_args.kwargs["format_json"])

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_no_timeout_kwarg_when_max_response_timeout_seconds_omitted(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt")
        self.assertNotIn("timeout", mock_call_ai.call_args.kwargs)

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_the_provider_deadline_matches_the_one_we_enforce(self, mock_call_ai):
        """The buffer is 0: the abandoned call's socket must close when we stop
        waiting for it, not seconds later while the next call is being timed."""
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", max_response_timeout_seconds=240)
        # Literal on purpose, so a change to PROVIDER_TIMEOUT_BUFFER_SECONDS has
        # to be made deliberately here too: 240 + 0.
        self.assertEqual(mock_call_ai.call_args.kwargs["timeout"], 240)

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_a_local_provider_is_pinned_resident(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("ollama", "gemma4", None, "prompt", max_response_timeout_seconds=240)
        self.assertEqual(
            mock_call_ai.call_args.kwargs["extra_options"]["keep_alive"], LOCAL_KEEP_ALIVE
        )

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_the_pin_does_not_depend_on_a_timeout_being_given(self, mock_call_ai):
        """Residency is a decision of the run, not a function of the deadline."""
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("ollama", "gemma4", None, "prompt")
        self.assertEqual(
            mock_call_ai.call_args.kwargs["extra_options"]["keep_alive"], LOCAL_KEEP_ALIVE
        )

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_non_ollama_provider_gets_no_keep_alive(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", max_response_timeout_seconds=240)
        self.assertNotIn("keep_alive", mock_call_ai.call_args.kwargs["extra_options"])

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_max_retries_defaults_to_five_for_judge_and_verdict_calls(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt")
        self.assertEqual(mock_call_ai.call_args.kwargs["max_retries"], 5)

    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_max_retries_is_forwarded_when_given(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", max_retries=0)
        self.assertEqual(mock_call_ai.call_args.kwargs["max_retries"], 0)


class TestConfigureAiLogging(unittest.TestCase):
    def _cfg(self, level):
        cfg = MagicMock()
        cfg.logging.level = level
        return cfg

    @patch("prompttestenv.api.set_verbosity", autospec=True)
    @patch("prompttestenv.api.get_app_config")
    def test_applies_the_configured_level(self, mock_get_cfg, mock_set_verbosity):
        mock_get_cfg.return_value = self._cfg("debug")
        configure_ai_logging()
        mock_set_verbosity.assert_called_once_with("debug")

    @patch("prompttestenv.logger.log_warning")
    @patch("prompttestenv.api.set_verbosity", autospec=True)
    @patch("prompttestenv.api.get_app_config")
    def test_unknown_level_falls_back_to_warning(self, mock_get_cfg, mock_set_verbosity, mock_warn):
        mock_get_cfg.return_value = self._cfg("chatty")
        mock_set_verbosity.side_effect = [ValueError("bad level"), None]
        configure_ai_logging()
        self.assertEqual(
            [c.args[0] for c in mock_set_verbosity.call_args_list], ["chatty", "warning"]
        )
        mock_warn.assert_called_once()


class TestGetTextEmbedding(unittest.TestCase):
    @patch("prompttestenv.api.get_embedding", autospec=True)
    def test_delegates_with_lowercased_provider(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1, 0.2]
        result = get_text_embedding("OLLAMA", "bge-m3", "some text")
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(mock_get_embedding.call_args.kwargs["provider"], "ollama")

    @patch("prompttestenv.api.get_embedding", autospec=True)
    def test_bounds_retries_and_timeout_like_get_llm_response(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1]
        get_text_embedding("ollama", "bge-m3", "t",
                           max_response_timeout_seconds=30, max_retries=1)
        kwargs = mock_get_embedding.call_args.kwargs
        self.assertEqual(kwargs["max_retries"], 1)
        self.assertEqual(kwargs["timeout"], 30 + PROVIDER_TIMEOUT_BUFFER_SECONDS)

    @patch("prompttestenv.api.get_embedding", autospec=True)
    def test_no_deadline_leaves_the_client_timeout(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1]
        get_text_embedding("ollama", "bge-m3", "t")
        self.assertNotIn("timeout", mock_get_embedding.call_args.kwargs)


class TestParseJsonObject(unittest.TestCase):
    """response_mime_type="application/json" makes a fenced or list-wrapped
    reply the rare case, not the common one — Ollama and OpenAI-compatible
    backends do not enforce JSON mode the way Google does. Shared by
    test_judge.py, reasoning.py and verdict.py, so a case missed here was
    previously reimplemented (or missed) independently up to three times."""

    def test_plain_object(self):
        self.assertEqual(parse_json_object('{"score": 7}'), {"score": 7})

    def test_wrapped_in_a_markdown_code_fence(self):
        text = '```json\n{"score": 7}\n```'
        self.assertEqual(parse_json_object(text), {"score": 7})

    def test_fence_with_no_language_tag(self):
        text = '```\n{"score": 7}\n```'
        self.assertEqual(parse_json_object(text), {"score": 7})

    def test_single_element_list_is_unwrapped(self):
        self.assertEqual(parse_json_object('[{"score": 7}]'), {"score": 7})

    def test_a_fenced_list_is_both_unwrapped_and_unfenced(self):
        text = '```json\n[{"score": 7}]\n```'
        self.assertEqual(parse_json_object(text), {"score": 7})

    def test_empty_list_returns_none(self):
        self.assertIsNone(parse_json_object("[]"))

    def test_list_of_non_dicts_returns_none(self):
        self.assertIsNone(parse_json_object("[1, 2, 3]"))

    def test_a_bare_scalar_returns_none(self):
        self.assertIsNone(parse_json_object("7"))

    def test_invalid_json_returns_none_instead_of_raising(self):
        self.assertIsNone(parse_json_object("not json at all"))


class TestPreloadModelForRun(ResidencyResetTestCase):
    """ResidencyResetTestCase, not plain TestCase: a preload now writes the
    residency slot, so without the reset these would leak into whatever runs
    next and make the suite order-dependent."""

    @patch("prompttestenv.api.preload_model", autospec=True)
    def test_preloads_for_ollama(self, mock_preload):
        preload_model_for_run("ollama", "gemma4")
        mock_preload.assert_called_once()

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.preload_model", autospec=True)
    def test_a_preload_records_residency(self, _mock_preload, mock_unload):
        """Without the record the judge's model stayed pinned until teardown."""
        preload_model_for_run("ollama", "judge-model")
        release_previous_candidate("ollama", "something-else")
        mock_unload.assert_called_once_with("ollama", "judge-model")

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.preload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_a_preload_releases_whatever_was_pinned_before(
        self, _mock_warm, _mock_preload, mock_unload
    ):
        """Phase 3 used to run its judge while the last candidate's model was
        still holding VRAM."""
        warm_up_for_run("ollama", "candidate-model")
        preload_model_for_run("ollama", "judge-model")
        mock_unload.assert_called_once_with("ollama", "candidate-model")

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.preload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_a_preload_of_the_same_model_releases_nothing(
        self, _mock_warm, _mock_preload, mock_unload
    ):
        warm_up_for_run("ollama", "same-model")
        preload_model_for_run("ollama", "same-model")
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.preload_model", autospec=True)
    def test_a_non_ollama_preload_touches_neither_slot_nor_provider(
        self, mock_preload, mock_unload
    ):
        preload_model_for_run("google", "gemini")
        mock_preload.assert_not_called()
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.preload_model", autospec=True)
    def test_skips_for_non_ollama_providers(self, mock_preload):
        preload_model_for_run("google", "gemini-2.5-flash")
        mock_preload.assert_not_called()

    @patch("prompttestenv.api.preload_model", autospec=True)
    def test_preloads_pinned_resident(self, mock_preload):
        preload_model_for_run("ollama", "gemma4")
        self.assertEqual(mock_preload.call_args.kwargs["keep_alive"], LOCAL_KEEP_ALIVE)


class TestWarmUpForRun(ResidencyResetTestCase):
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_forwards_media_paths_as_file_paths(self, mock_warm_up):
        mock_warm_up.return_value = True
        result = warm_up_for_run("google", "gemini-3-flash", ["/abs/paper.pdf"])
        self.assertTrue(result)
        self.assertEqual(
            mock_warm_up.call_args.kwargs,
            {
                "provider": "google",
                "model": "gemini-3-flash",
                "file_paths": ["/abs/paper.pdf"],
                "keep_alive": None,
                "timeout": None,
            },
        )

    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_media_paths_default_to_none(self, mock_warm_up):
        warm_up_for_run("ollama", "gemma4")
        self.assertIsNone(mock_warm_up.call_args.kwargs["file_paths"])

    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_propagates_a_negative_result(self, mock_warm_up):
        mock_warm_up.return_value = False
        self.assertFalse(warm_up_for_run("script", "s.py"))

    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_a_local_model_is_warmed_up_pinned(self, mock_warm_up):
        warm_up_for_run("ollama", "gemma4")
        self.assertEqual(mock_warm_up.call_args.kwargs["keep_alive"], LOCAL_KEEP_ALIVE)

    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_the_load_gets_the_buffered_timeout(self, mock_warm_up):
        """A slow local model must not be cut off by the client's own default."""
        warm_up_for_run("ollama", "gemma4", max_response_timeout_seconds=240)
        # 240 + PROVIDER_TIMEOUT_BUFFER_SECONDS, spelled out deliberately.
        self.assertEqual(mock_warm_up.call_args.kwargs["timeout"], 240)


class TestCandidateBoundary(ResidencyResetTestCase):
    """Local models are pinned, so something has to release the previous one."""

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_the_first_candidate_releases_nothing(self, _mock_warm_up, mock_unload):
        release_previous_candidate("ollama", "gemma4")
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_a_different_model_releases_the_previous_one(self, _mock_warm_up, mock_unload):
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_called_once_with("ollama", "gemma4")

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_the_same_model_is_not_reloaded(self, _mock_warm_up, mock_unload):
        """Candidates differing only by system prompt must not pay a reload."""
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("ollama", "gemma4")
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_a_cloud_candidate_still_releases_a_local_one(self, _mock_warm_up, mock_unload):
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("google", "gemini")
        mock_unload.assert_called_once_with("ollama", "gemma4")

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    def test_a_release_is_not_repeated(self, _mock_warm_up, mock_unload):
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("ollama", "qwen4")
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_called_once()

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.call_ai", autospec=True)
    def test_a_generated_call_marks_residency_without_a_warm_up(self, mock_call_ai, mock_unload):
        """Warm-up can be disabled, or skipped for a fully resumed candidate."""
        resp = MagicMock()
        resp.text, resp.output_tokens, resp.reasoning_tokens = "hi", 1, 0
        resp.reasoning_text, resp.reasoning_is_summary = "", False
        mock_call_ai.return_value = resp

        get_llm_response("ollama", "gemma4", None, "prompt")
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_called_once_with("ollama", "gemma4")


class TestTeardown(ResidencyResetTestCase):
    @patch("prompttestenv.api.cleanup", autospec=True)
    def test_calls_cleanup(self, mock_cleanup):
        teardown()
        mock_cleanup.assert_called_once()

    @patch("prompttestenv.api.unload_model", autospec=True)
    @patch("prompttestenv.api.warm_up", autospec=True)
    @patch("prompttestenv.api.cleanup", autospec=True)
    def test_forgets_the_resident_candidate(self, _mock_cleanup, _mock_warm_up, mock_unload):
        """cleanup() already released it; a second run must not release it again."""
        warm_up_for_run("ollama", "gemma4")
        teardown()
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.configure_provider", autospec=True)
    @patch("prompttestenv.api.preload_model", autospec=True)
    @patch("prompttestenv.api.cleanup", autospec=True)
    def test_takes_back_a_registered_context_size(self, _cleanup, _preload, mock_configure):
        preload_model_for_run("ollama", "gemma4", context_size=32768)
        teardown()
        mock_configure.assert_called_once_with("ollama", context_size=None)

        teardown()
        mock_configure.assert_called_once()

    @patch("prompttestenv.api.configure_provider", autospec=True)
    @patch("prompttestenv.api.preload_model", autospec=True)
    @patch("prompttestenv.api.cleanup", autospec=True)
    def test_leaves_the_config_alone_when_nothing_was_registered(
        self, _cleanup, _preload, mock_configure,
    ):
        preload_model_for_run("ollama", "gemma4")
        teardown()
        mock_configure.assert_not_called()


class TestCallWithTimeout(ResidencyResetTestCase):
    def test_success_path_returns_result_and_false(self):
        result, timed_out = call_with_timeout(lambda x: x * 2, 3, timeout=5.0)
        self.assertEqual(result, 6)
        self.assertFalse(timed_out)

    def test_it_returns_at_the_deadline_instead_of_waiting_for_the_call(self):
        """The regression the rest of this class could not see.

        Every other test here asserts only the return VALUE, so the pooled
        version passed them all while taking as long as the call it had given
        up on. The bound is deliberately loose (5x the deadline, a quarter of
        the call): Windows timer granularity is ~15ms and the suite runs under
        load, so a tight bound would flake without catching anything more.
        """
        started = time.perf_counter()
        result, timed_out = call_with_timeout(lambda: time.sleep(2.0), timeout=0.2)
        elapsed = time.perf_counter() - started

        self.assertTrue(timed_out)
        self.assertIsNone(result)
        self.assertLess(elapsed, 1.0, f"waited {elapsed:.2f}s for a 0.2s deadline")

    def test_the_abandoned_worker_is_a_daemon(self):
        """Process exit cannot be asserted from inside the process, so assert
        the property that decides it: a non-daemon worker is joined by the
        interpreter at exit, which is where the pooled version moved its block."""
        captured = {}
        entered = threading.Event()

        def slow():
            captured["thread"] = threading.current_thread()
            entered.set()
            time.sleep(1.0)

        call_with_timeout(slow, timeout=0.05)

        self.assertTrue(entered.wait(1.0))
        self.assertTrue(captured["thread"].daemon)

    def test_an_exception_inside_fn_still_reaches_the_caller(self):
        """Future.result() re-raised for free; the thread version has to do it
        by hand, and run_project's teardown contract depends on it."""
        def boom():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            call_with_timeout(boom, timeout=5.0)

    @patch("prompttestenv.api.unload_model", autospec=True)
    def test_timeout_leaves_residency_alone(self, mock_unload):
        """A timeout must not evict the model the next call will reuse.

        Warm-up runs once per candidate, outside the timed block, so a reload
        forced here would be charged to the next repetition's own elapsed. The
        model is the running candidate's, so the candidate boundary and
        teardown free it anyway.
        """
        def slow(*args, **kwargs):
            time.sleep(0.3)
            return "too slow"

        result, timed_out = call_with_timeout(slow, timeout=0.05)
        self.assertIsNone(result)
        self.assertTrue(timed_out)
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.unload_model", autospec=True)
    def test_the_timed_out_model_stays_in_the_residency_slot(self, mock_unload):
        """The slot must keep telling the truth: the model is still loaded.

        Were it cleared, the next candidate boundary would skip the release and
        leave the model resident for the rest of the run.
        """
        with patch("prompttestenv.api.call_ai", autospec=True) as mock_call:
            mock_call.return_value = MagicMock(
                text="hi", output_tokens=1, reasoning_tokens=0,
                reasoning_text="", reasoning_is_summary=False,
            )
            get_llm_response("ollama", "gemma4", "sys", "p")

        call_with_timeout(lambda: time.sleep(0.3), timeout=0.05)

        release_previous_candidate("ollama", "other-model")
        mock_unload.assert_called_once_with("ollama", "gemma4")

    def test_fn_kwargs_are_passed_through(self):
        result, _timed_out = call_with_timeout(
            lambda a, b: a + b, 2, fn_kwargs={"b": 4}, timeout=5.0,
        )
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
