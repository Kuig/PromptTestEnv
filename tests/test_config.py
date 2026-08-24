from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prompttestenv.config as config
from testutils import LoggerResetTestCase


class TestInitProject(LoggerResetTestCase):
    def setUp(self):
        # Two separate temp dirs: one stands in for the "real repo root" that
        # config._PROJECT_ROOT normally points to (so secrets.json never
        # touches the actual repo), one is the project_dir under test.
        self.fake_repo_root = Path(tempfile.mkdtemp(prefix="prompttestenv_test_root_"))
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_project_")
        self.addCleanup(shutil.rmtree, str(self.fake_repo_root), ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self._patcher = patch.object(config, "_PROJECT_ROOT", self.fake_repo_root)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_creates_all_expected_files(self):
        config.init_project(self.project_dir)

        p = Path(self.project_dir)
        self.assertTrue((p / "candidates.json").exists())
        self.assertTrue((p / "judge_config.json").exists())
        self.assertTrue((p / "test_cases.json").exists())
        self.assertTrue((p / "global_criteria.json").exists())
        self.assertTrue((p / "system_prompts" / "pirate_prompt.txt").exists())
        self.assertTrue((p / "test_files" / "sample.txt").exists())
        self.assertTrue((self.fake_repo_root / "secrets.json").exists())

    def test_default_candidates_have_expected_shape(self):
        config.init_project(self.project_dir)
        data = json.loads((Path(self.project_dir) / "candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0]["name"], "Baseline (Flash 2.5)")

    def test_does_not_touch_the_real_repo_secrets(self):
        # Sanity check that the patch actually redirected _PROJECT_ROOT away
        # from the real repository root.
        real_root = Path(__file__).parent.parent
        config.init_project(self.project_dir)
        self.assertNotEqual(config._PROJECT_ROOT, real_root)

    def test_second_call_does_not_overwrite_existing_files(self):
        config.init_project(self.project_dir)
        cand_file = Path(self.project_dir) / "candidates.json"
        cand_file.write_text('[{"name": "custom"}]', encoding="utf-8")

        config.init_project(self.project_dir)

        data = json.loads(cand_file.read_text(encoding="utf-8"))
        self.assertEqual(data, [{"name": "custom"}])

    def test_custom_candidates_with_system_prompt_text_are_saved_to_file(self):
        config.init_project(
            self.project_dir,
            custom_candidates=[{"name": "X", "model": "m", "system_prompt_text": "Be nice."}],
        )
        data = json.loads((Path(self.project_dir) / "candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(data[0]["system_prompt_file"], "custom_prompt_0.txt")
        self.assertNotIn("system_prompt_text", data[0])
        prompt_path = Path(self.project_dir) / "system_prompts" / "custom_prompt_0.txt"
        self.assertEqual(prompt_path.read_text(encoding="utf-8"), "Be nice.")


class TestGetApiKey(unittest.TestCase):
    @patch("prompttestenv.config.load_secrets")
    def test_returns_key_when_present(self, mock_load_secrets):
        mock_load_secrets.return_value = {"google_api_key": "real-key"}
        self.assertEqual(config.get_api_key(), "real-key")

    @patch("prompttestenv.config.load_secrets")
    def test_raises_when_key_missing(self, mock_load_secrets):
        mock_load_secrets.return_value = {}
        with self.assertRaises(ValueError):
            config.get_api_key()

    @patch("prompttestenv.config.load_secrets")
    def test_raises_when_key_is_placeholder(self, mock_load_secrets):
        mock_load_secrets.return_value = {"google_api_key": "INSERT_YOUR_API_KEY_HERE"}
        with self.assertRaises(ValueError):
            config.get_api_key()


if __name__ == "__main__":
    unittest.main()
