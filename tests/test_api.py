from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from prompttestenv.api import (
    cosine_similarity,
    get_llm_response,
    get_text_embedding,
    preload_model_for_run,
    teardown,
)


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
    def _fake_response(self, text="hi", out_tok=10, r_tok=0, reasoning=""):
        resp = MagicMock()
        resp.text = text
        resp.output_tokens = out_tok
        resp.reasoning_tokens = r_tok
        resp.reasoning_text = reasoning
        return resp

    @patch("prompttestenv.api.call_ai")
    def test_returns_tuple_from_response(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response(text="hello", out_tok=5, r_tok=2, reasoning="thinking...")
        result = get_llm_response("google", "gemini", None, "prompt")
        self.assertEqual(result, ("hello", 5, 2, "thinking..."))

    @patch("prompttestenv.api.call_ai")
    def test_none_reasoning_text_becomes_empty_string(self, mock_call_ai):
        mock_call_ai.return_value = self._fake_response(reasoning=None)
        _, _, _, reasoning_text = get_llm_response("google", "gemini", None, "prompt")
        self.assertEqual(reasoning_text, "")

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


class TestTeardown(unittest.TestCase):
    @patch("prompttestenv.api.cleanup")
    def test_calls_cleanup(self, mock_cleanup):
        teardown()
        mock_cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
