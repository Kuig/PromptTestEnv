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

from prompttestenv.catalog import list_catalog, offline_findings, online_findings
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

    def test_json_output_on_an_anthropic_candidate_warns_but_does_not_block(self):
        json_test = TestCase(id="j1", prompt="p", criteria="c", json_output=True)
        errors, warnings = self._run([Candidate("C", "anthropic", "m")],
                                     [self._test(), json_test])
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("'C'", warnings[0])
        self.assertIn("j1", warnings[0])
        self.assertNotIn("t1", warnings[0])

    def test_json_output_is_silent_where_the_provider_has_a_json_mode(self):
        json_test = TestCase(id="j1", prompt="p", criteria="c", json_output=True)
        self.assertEqual(self._run([Candidate("G", "google", "m")], [json_test]), ([], []))

    def test_an_anthropic_candidate_without_json_tests_is_silent(self):
        self.assertEqual(self._run([Candidate("C", "anthropic", "m")], [self._test()]), ([], []))

    def test_an_unknown_candidate_provider_warns_but_does_not_block(self):
        errors, warnings = self._run([Candidate("X", "nosuchprovider", "m")], [self._test()])
        self.assertEqual(errors, [])
        self.assertTrue(any("unknown provider" in w for w in warnings))


def _info(model_id, **capabilities):
    return {"id": model_id, "display_name": None, "reasoning": None, "tool_use": None,
            "file_types": None, "embeddings": None, "context_length": None, **capabilities}


def _listing(provider, *models, error=None):
    return {"provider": provider, "models": list(models), "error": error}


class TestOnlineFindings(unittest.TestCase):
    def setUp(self):
        self.jc = JudgeConfig()
        self.jc.global_criteria = GlobalCriteria(mode="none")
        self.tests = [TestCase(id="t1", prompt="p", criteria="c")]

    def _run(self, candidates=(), listings=None, verdict=False):
        return online_findings(list(candidates), self.jc, self.tests, listings or {},
                               verdict=verdict)

    def test_a_listed_model_with_unknown_capabilities_is_fine(self):
        self.jc.test_judge.provider, self.jc.test_judge.model = "ollama", "qwen3:8b"
        self.assertEqual(self._run(listings={"ollama": _listing("ollama", _info("qwen3:8b"))}), [])

    def test_a_model_the_provider_does_not_list_is_named(self):
        findings = self._run([Candidate("A", "ollama", "llama9")],
                             {"ollama": _listing("ollama", _info("qwen3:8b"))})
        self.assertEqual(findings, ["Candidate 'A': 'ollama' lists no model 'llama9'."])

    def test_an_untagged_ollama_name_means_latest(self):
        self.assertEqual(self._run([Candidate("A", "ollama", "qwen3")],
                                   {"ollama": _listing("ollama", _info("qwen3:latest"))}), [])

    def test_no_listing_or_a_failed_one_checks_nothing(self):
        cand = [Candidate("A", "ollama", "llama9")]
        self.assertEqual(self._run(cand, {}), [])
        self.assertEqual(self._run(cand, {"ollama": _listing("ollama", error="down")}), [])
        self.assertEqual(self._run(cand, {"ollama": _listing("ollama")}), [])

    def test_embedding_and_chat_models_in_the_wrong_seat(self):
        self.tests = [TestCase(id="t1", prompt="p", criteria="c", judge_type="similarity")]
        self.jc.similarity_judge.provider, self.jc.similarity_judge.model = "ollama", "qwen3:8b"
        listings = {"ollama": _listing("ollama", _info("qwen3:8b", embeddings=False),
                                       _info("bge-m3", embeddings=True))}
        findings = self._run([Candidate("A", "ollama", "bge-m3")], listings)
        self.assertIn("Candidate 'A': 'bge-m3' is an embedding model, not a chat model.", findings)
        self.assertIn("similarity_judge: 'qwen3:8b' is not an embedding model.", findings)

    def test_thinking_on_a_model_that_does_not_reason(self):
        listings = {"ollama": _listing("ollama", _info("gemma3", reasoning=False))}
        thinking = Candidate("A", "ollama", "gemma3", thinking=True)
        default = Candidate("B", "ollama", "gemma3")
        self.assertEqual(self._run([thinking, default], listings),
                         ["Candidate 'A': thinking is on, but 'gemma3' does not reason."])

    def test_a_context_size_beyond_the_model(self):
        self.jc.reasoning_analysis = "best"
        rj = self.jc.reasoning_judge
        rj.provider, rj.model, rj.context_size = "ollama", "small", 65536
        self.jc.similarity_judge.provider = "google"  # the analysis embeds too
        findings = self._run(listings={"ollama": _listing("ollama", _info("small",
                                                                         context_length=8192))})
        self.assertEqual(findings,
                         ["reasoning_judge: context_size 65536 exceeds the 8192 'small' supports."])

    def test_only_judges_in_use_are_checked(self):
        self.jc.verdict_judge.provider, self.jc.verdict_judge.model = "ollama", "absent"
        listings = {"ollama": _listing("ollama", _info("qwen3:8b"))}
        self.assertEqual(self._run(listings=listings, verdict=False), [])
        self.assertTrue(self._run(listings=listings, verdict=True))


if __name__ == "__main__":
    unittest.main()
