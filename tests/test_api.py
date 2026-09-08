from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from prompttestenv.api import (
    LOCAL_KEEP_ALIVE,
    call_with_timeout,
    configure_ai_logging,
    cosine_similarity,
    get_llm_response,
    get_text_embedding,
    preload_model_for_run,
    release_previous_candidate,
    warm_up_for_run,
    teardown,
)
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
        resp = MagicMock()
        resp.text = text
        resp.output_tokens = out_tok
        resp.reasoning_tokens = r_tok
        resp.reasoning_text = reasoning
        resp.reasoning_is_summary = is_summary
        return resp

    @patch("prompttestenv.api.call_ai")
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

    @patch("prompttestenv.api.call_ai")
    def test_none_reasoning_text_becomes_empty_string(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response(reasoning=None)
        self.assertEqual(get_llm_response("google", "gemini", None, "prompt").reasoning_text, "")

    @patch("prompttestenv.api.call_ai")
    def test_provider_without_the_summary_flag_defaults_to_raw(self, mock_call_ai):
        """An older UnifiedAiClient has no reasoning_is_summary, and raw is the safe reading."""
        resp = self._fake_response(reasoning="raw trace")
        del resp.reasoning_is_summary
        mock_call_ai.return_value = resp
        self.assertFalse(get_llm_response("google", "gemini", None, "prompt").reasoning_is_summary)

    @patch("prompttestenv.api.call_ai")
    def test_thinking_string_true_normalized_to_bool(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", thinking="true")
        self.assertIs(mock_call_ai.call_args.kwargs["thinking"], True)

    @patch("prompttestenv.api.call_ai")
    def test_thinking_string_false_normalized_to_bool(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", thinking="false")
        self.assertIs(mock_call_ai.call_args.kwargs["thinking"], False)

    @patch("prompttestenv.api.call_ai")
    def test_thinking_default_string_stays_default(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", thinking="default")
        self.assertEqual(mock_call_ai.call_args.kwargs["thinking"], "default")

    @patch("prompttestenv.api.call_ai")
    def test_disable_safety_sets_extra_option(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", disable_safety=True)
        self.assertEqual(mock_call_ai.call_args.kwargs["extra_options"], {"disable_safety": True})

    @patch("prompttestenv.api.call_ai")
    def test_disable_safety_false_leaves_extra_options_empty(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", disable_safety=False)
        self.assertEqual(mock_call_ai.call_args.kwargs["extra_options"], {})

    @patch("prompttestenv.api.call_ai")
    def test_provider_is_lowercased(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("GOOGLE", "gemini", None, "prompt")
        self.assertEqual(mock_call_ai.call_args.kwargs["provider"], "google")

    @patch("prompttestenv.api.call_ai")
    def test_json_mime_type_enables_format_json(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", response_mime_type="application/json")
        self.assertTrue(mock_call_ai.call_args.kwargs["format_json"])

    @patch("prompttestenv.api.call_ai")
    def test_no_timeout_kwarg_when_max_response_timeout_seconds_omitted(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt")
        self.assertNotIn("timeout", mock_call_ai.call_args.kwargs)

    @patch("prompttestenv.api.call_ai")
    def test_timeout_is_max_response_timeout_seconds_plus_buffer(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", max_response_timeout_seconds=240)
        self.assertEqual(mock_call_ai.call_args.kwargs["timeout"], 250)

    @patch("prompttestenv.api.call_ai")
    def test_a_local_provider_is_pinned_resident(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("ollama", "gemma4", None, "prompt", max_response_timeout_seconds=240)
        self.assertEqual(
            mock_call_ai.call_args.kwargs["extra_options"]["keep_alive"], LOCAL_KEEP_ALIVE
        )

    @patch("prompttestenv.api.call_ai")
    def test_the_pin_does_not_depend_on_a_timeout_being_given(self, mock_call_ai):
        """Residency is a decision of the run, not a function of the deadline."""
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("ollama", "gemma4", None, "prompt")
        self.assertEqual(
            mock_call_ai.call_args.kwargs["extra_options"]["keep_alive"], LOCAL_KEEP_ALIVE
        )

    @patch("prompttestenv.api.call_ai")
    def test_non_ollama_provider_gets_no_keep_alive(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", max_response_timeout_seconds=240)
        self.assertNotIn("keep_alive", mock_call_ai.call_args.kwargs["extra_options"])

    @patch("prompttestenv.api.call_ai")
    def test_max_retries_defaults_to_five_for_judge_and_verdict_calls(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt")
        self.assertEqual(mock_call_ai.call_args.kwargs["max_retries"], 5)

    @patch("prompttestenv.api.call_ai")
    def test_max_retries_is_forwarded_when_given(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response()
        get_llm_response("google", "gemini", None, "prompt", max_retries=0)
        self.assertEqual(mock_call_ai.call_args.kwargs["max_retries"], 0)


class TestConfigureAiLogging(unittest.TestCase):
    def _cfg(self, level):
        cfg = MagicMock()
        cfg.logging.level = level
        return cfg

    @patch("prompttestenv.api.set_verbosity")
    @patch("prompttestenv.api.get_app_config")
    def test_applies_the_configured_level(self, mock_get_cfg, mock_set_verbosity):
        mock_get_cfg.return_value = self._cfg("debug")
        configure_ai_logging()
        mock_set_verbosity.assert_called_once_with("debug")

    @patch("prompttestenv.logger.log_warning")
    @patch("prompttestenv.api.set_verbosity")
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
    @patch("prompttestenv.api.get_embedding")
    def test_delegates_with_lowercased_provider(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1, 0.2]
        result = get_text_embedding("OLLAMA", "bge-m3", "some text")
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(mock_get_embedding.call_args.kwargs["provider"], "ollama")


class TestPreloadModelForRun(unittest.TestCase):
    @patch("prompttestenv.api.preload_model")
    def test_preloads_for_ollama(self, mock_preload):
        preload_model_for_run("ollama", "gemma4")
        mock_preload.assert_called_once()

    @patch("prompttestenv.api.preload_model")
    def test_skips_for_non_ollama_providers(self, mock_preload):
        preload_model_for_run("google", "gemini-2.5-flash")
        mock_preload.assert_not_called()

    @patch("prompttestenv.api.preload_model")
    def test_preloads_pinned_resident(self, mock_preload):
        preload_model_for_run("ollama", "gemma4")
        self.assertEqual(mock_preload.call_args.kwargs["keep_alive"], LOCAL_KEEP_ALIVE)


class TestWarmUpForRun(ResidencyResetTestCase):
    @patch("prompttestenv.api.warm_up")
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

    @patch("prompttestenv.api.warm_up")
    def test_media_paths_default_to_none(self, mock_warm_up):
        warm_up_for_run("ollama", "gemma4")
        self.assertIsNone(mock_warm_up.call_args.kwargs["file_paths"])

    @patch("prompttestenv.api.warm_up")
    def test_propagates_a_negative_result(self, mock_warm_up):
        mock_warm_up.return_value = False
        self.assertFalse(warm_up_for_run("script", "s.py"))

    @patch("prompttestenv.api.warm_up")
    def test_a_local_model_is_warmed_up_pinned(self, mock_warm_up):
        warm_up_for_run("ollama", "gemma4")
        self.assertEqual(mock_warm_up.call_args.kwargs["keep_alive"], LOCAL_KEEP_ALIVE)

    @patch("prompttestenv.api.warm_up")
    def test_the_load_gets_the_buffered_timeout(self, mock_warm_up):
        """A slow local model must not be cut off by the client's own default."""
        warm_up_for_run("ollama", "gemma4", max_response_timeout_seconds=240)
        self.assertEqual(mock_warm_up.call_args.kwargs["timeout"], 250)


class TestCandidateBoundary(ResidencyResetTestCase):
    """Local models are pinned, so something has to release the previous one."""

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.warm_up")
    def test_the_first_candidate_releases_nothing(self, _mock_warm_up, mock_unload):
        release_previous_candidate("ollama", "gemma4")
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.warm_up")
    def test_a_different_model_releases_the_previous_one(self, _mock_warm_up, mock_unload):
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_called_once_with("ollama", "gemma4")

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.warm_up")
    def test_the_same_model_is_not_reloaded(self, _mock_warm_up, mock_unload):
        """Candidates differing only by system prompt must not pay a reload."""
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("ollama", "gemma4")
        mock_unload.assert_not_called()

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.warm_up")
    def test_a_cloud_candidate_still_releases_a_local_one(self, _mock_warm_up, mock_unload):
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("google", "gemini")
        mock_unload.assert_called_once_with("ollama", "gemma4")

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.warm_up")
    def test_a_release_is_not_repeated(self, _mock_warm_up, mock_unload):
        warm_up_for_run("ollama", "gemma4")
        release_previous_candidate("ollama", "qwen4")
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_called_once()

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.call_ai")
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
    @patch("prompttestenv.api.cleanup")
    def test_calls_cleanup(self, mock_cleanup):
        teardown()
        mock_cleanup.assert_called_once()

    @patch("prompttestenv.api.unload_model")
    @patch("prompttestenv.api.warm_up")
    @patch("prompttestenv.api.cleanup")
    def test_forgets_the_resident_candidate(self, _mock_cleanup, _mock_warm_up, mock_unload):
        """cleanup() already released it; a second run must not release it again."""
        warm_up_for_run("ollama", "gemma4")
        teardown()
        release_previous_candidate("ollama", "qwen4")
        mock_unload.assert_not_called()


class TestCallWithTimeout(unittest.TestCase):
    def test_success_path_returns_result_and_false(self):
        result, timed_out = call_with_timeout(
            lambda x: x * 2, 3, timeout=5.0, provider="google", model="gemini",
        )
        self.assertEqual(result, 6)
        self.assertFalse(timed_out)

    @patch("prompttestenv.api.unload_model")
    def test_timeout_releases_the_model(self, mock_unload):
        def slow(*args, **kwargs):
            time.sleep(0.3)
            return "too slow"

        result, timed_out = call_with_timeout(
            slow, timeout=0.05, provider="ollama", model="gemma4",
        )
        self.assertIsNone(result)
        self.assertTrue(timed_out)
        mock_unload.assert_called_once_with("ollama", "gemma4")

    @patch("prompttestenv.api.unload_model")
    def test_no_release_when_the_call_succeeds(self, mock_unload):
        call_with_timeout(
            lambda: "fast", timeout=5.0, provider="ollama", model="gemma4",
        )
        mock_unload.assert_not_called()

    def test_fn_kwargs_are_passed_through(self):
        result, _timed_out = call_with_timeout(
            lambda a, b: a + b, 2, fn_kwargs={"b": 4}, timeout=5.0, provider="google", model="gemini",
        )
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
