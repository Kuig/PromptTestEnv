"""catalog.py: the provider/model listing and the offline capability checks.

The listing is tested against fabricated ProviderInfo/ModelInfo, with the client
patched (autospec, so a renamed parameter fails here). The offline checks run
against the client's REAL provider registry instead: get_provider() needs no
credentials and makes no request, and the capability tables it exposes are
exactly what these checks must stay in step with.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from unified_ai_client import ModelInfo, ProviderInfo

from prompttestenv.catalog import list_catalog, offline_findings
from prompttestenv.models import Candidate, GlobalCriteria, JudgeConfig, TestCase


def _provider(name="ollama", local=True):
    return ProviderInfo(
        name=name, local=local, requires_api_key=not local,
        file_types=frozenset({"image", "audio"}), supports_unload=local,
        supports_embeddings=True,
    )


class TestListProviders(unittest.TestCase):
    @patch("prompttestenv.catalog.list_providers", autospec=True)
    def test_providers_are_json_safe_and_keep_registry_order(self, mock_list):
        mock_list.return_value = [_provider("ollama"), _provider("google", local=False)]
        out = list_catalog()
        self.assertIsNone(out["error"])
        self.assertEqual([p["name"] for p in out["providers"]], ["ollama", "google"])
        self.assertEqual(out["providers"][0]["file_types"], ["audio", "image"])

    @patch("prompttestenv.catalog.list_providers", autospec=True)
    def test_probe_local_is_forwarded(self, mock_list):
        mock_list.return_value = []
        list_catalog(probe_local=False)
        self.assertEqual(mock_list.call_args.kwargs, {"probe_local": False})

    @patch("prompttestenv.catalog.list_providers", autospec=True)
    def test_a_failure_is_reported_not_raised(self, mock_list):
        mock_list.side_effect = RuntimeError("boom")
        out = list_catalog()
        self.assertEqual(out["providers"], [])
        self.assertIn("boom", out["error"])

    @patch("prompttestenv.catalog.list_providers", autospec=True)
    def test_a_hanging_listing_is_bounded(self, mock_list):
        mock_list.side_effect = lambda **_kw: time.sleep(1) or []
        started = time.monotonic()
        out = list_catalog(timeout=0.1)
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertIn("timed out", out["error"])


class TestListModels(unittest.TestCase):
    @patch("prompttestenv.catalog.list_models", autospec=True)
    def test_models_are_sorted_and_unknown_stays_null(self, mock_list):
        mock_list.return_value = [
            ModelInfo(id="zeta", provider="ollama", reasoning=True,
                      file_types=frozenset({"image"}), context_length=8192),
            ModelInfo(id="alpha", provider="ollama"),
        ]
        out = list_catalog("Ollama")
        self.assertEqual(out["provider"], "ollama")
        self.assertEqual([m["id"] for m in out["models"]], ["alpha", "zeta"])
        alpha, zeta = out["models"]
        # None means the provider did not say; it must never turn into False.
        self.assertIsNone(alpha["reasoning"])
        self.assertIsNone(alpha["file_types"])
        self.assertEqual(zeta["file_types"], ["image"])
        self.assertEqual(mock_list.call_args.args, ("ollama",))

    @patch("prompttestenv.catalog.list_models", autospec=True)
    def test_a_provider_that_cannot_enumerate_gets_an_empty_list(self, mock_list):
        mock_list.side_effect = NotImplementedError
        out = list_catalog("script")
        self.assertEqual((out["models"], out["error"]), ([], None))

    @patch("prompttestenv.catalog.list_models", autospec=True)
    def test_a_missing_key_is_reported_not_raised(self, mock_list):
        mock_list.side_effect = ValueError("no key for openai")
        out = list_catalog("openai")
        self.assertEqual(out["models"], [])
        self.assertIn("no key", out["error"])


class OfflineFindingsTestCase(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_catalog_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        os.makedirs(os.path.join(self.project_dir, "test_files"))
        for name in ("clip.mp4", "notes.md", "photo.png"):
            with open(os.path.join(self.project_dir, "test_files", name), "w") as f:
                f.write("x")
        self.jc = JudgeConfig()
        self.jc.global_criteria = GlobalCriteria(mode="none")

    @staticmethod
    def _test(test_id="t1", judge_type="llm-judge", file=None, criteria="c"):
        return TestCase(id=test_id, prompt="p", criteria=criteria,
                        judge_type=judge_type, file=file)

    def _run(self, candidates=(), tests=(), verdict=False):
        return offline_findings(list(candidates), self.jc, list(tests), self.project_dir,
                                verdict=verdict)


class TestJudgeErrors(OfflineFindingsTestCase):
    def test_a_clean_project_has_no_findings(self):
        self.assertEqual(self._run([Candidate("A", "google", "m")], [self._test()]), ([], []))

    def test_anthropic_as_test_judge_is_an_error(self):
        self.jc.test_judge.provider = "anthropic"
        errors, _ = self._run(tests=[self._test()])
        self.assertTrue(any("test_judge" in e and "anthropic" in e for e in errors))

    def test_anthropic_as_test_judge_is_fine_when_no_llm_judge_runs(self):
        self.jc.test_judge.provider = "anthropic"
        errors, _ = self._run(tests=[self._test(judge_type="assert", criteria="s: (5, 'x')")])
        self.assertEqual(errors, [])

    def test_anthropic_as_reasoning_judge_is_an_error_only_when_analysing(self):
        self.jc.reasoning_judge.provider = "anthropic"
        self.assertEqual(self._run(tests=[self._test()])[0], [])
        self.jc.reasoning_analysis = "best"
        errors, _ = self._run(tests=[self._test()])
        self.assertTrue(any("reasoning_judge" in e for e in errors))

    def test_a_similarity_judge_without_embeddings_is_an_error(self):
        self.jc.similarity_judge.provider = "anthropic"
        errors, _ = self._run(tests=[self._test(judge_type="similarity")])
        self.assertTrue(any("similarity_judge" in e and "embeddings" in e for e in errors))

    def test_an_assert_lambda_using_similarity_counts_as_embedding(self):
        self.jc.similarity_judge.provider = "anthropic"
        plain = self._test(judge_type="assert", criteria="s: (5, 'x')")
        self.assertEqual(self._run(tests=[plain])[0], [])
        uses = self._test(judge_type="assert", criteria="s: (10 * similarity(s, 'a'), 'x')")
        self.assertTrue(self._run(tests=[uses])[0])

    def test_an_unknown_judge_provider_is_an_error(self):
        self.jc.test_judge.provider = "nosuchprovider"
        errors, _ = self._run(tests=[self._test()])
        self.assertTrue(any("unknown provider" in e for e in errors))

    def test_the_verdict_judge_is_only_checked_when_a_verdict_is_written(self):
        self.jc.verdict_judge.provider = "nosuchprovider"
        self.assertEqual(self._run(tests=[self._test()], verdict=False)[0], [])
        self.assertTrue(self._run(tests=[self._test()], verdict=True)[0])

    def test_media_the_test_judge_cannot_take_is_an_error_when_passed(self):
        self.jc.test_judge.provider = "ollama"
        test = self._test(file="test_files/clip.mp4")
        self.assertEqual(self._run(tests=[test])[0], [])  # pass_media_to_judge off
        self.jc.pass_media_to_judge = True
        errors, _ = self._run(tests=[test])
        self.assertTrue(any("clip.mp4" in e for e in errors))

    def test_text_attachments_are_always_accepted(self):
        self.jc.test_judge.provider = "ollama"
        self.jc.pass_media_to_judge = True
        self.assertEqual(self._run(tests=[self._test(file="test_files/notes.md")])[0], [])


class TestCandidateWarnings(OfflineFindingsTestCase):
    def test_an_unsupported_attachment_warns_once_per_candidate_and_test(self):
        tests = [self._test(file=["test_files/clip.mp4", "test_files/clip.mp4"])]
        errors, warnings = self._run([Candidate("Local", "ollama", "m")], tests)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("'Local' will skip test 't1'", warnings[0])

    def test_a_supported_attachment_is_silent(self):
        tests = [self._test(file="test_files/photo.png")]
        self.assertEqual(self._run([Candidate("Local", "ollama", "m")], tests), ([], []))

    def test_script_is_never_checked(self):
        tests = [self._test(file="test_files/clip.mp4")]
        self.assertEqual(self._run([Candidate("S", "script", "m")], tests), ([], []))

    def test_an_unknown_candidate_provider_warns_but_does_not_block(self):
        errors, warnings = self._run([Candidate("X", "nosuchprovider", "m")], [self._test()])
        self.assertEqual(errors, [])
        self.assertTrue(any("unknown provider" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
