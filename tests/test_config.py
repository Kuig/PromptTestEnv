from __future__ import annotations

import importlib.resources
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prompttestenv.config as config
from prompttestenv.models import REASONING_DIMENSIONS
from testutils import LoggerResetTestCase

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

        # Regression guard for the templates/default_*.json resource-file
        # loading path: confirm actual prompt content was written, not just
        # that the file exists.
        judge_data = json.loads((p / "judge_config.json").read_text(encoding="utf-8"))
        self.assertTrue(judge_data["test_judge"]["evaluation_template"].strip())

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



class TestAppConfig(LoggerResetTestCase):
    """config.json holds the measurement instrument, so its resolution is part of the contract."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(os.chdir, os.getcwd())

    def _write(self, directory, payload):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_explicit_path_is_read(self):
        path = self._write(self.tmp, {"local_providers": ["from-file"]})
        self.assertEqual(config.AppConfig.load(path).local_providers, ["from-file"])

    def test_working_directory_takes_precedence_over_the_repo_root(self):
        self._write(self.tmp, {"local_providers": ["from-cwd"]})
        os.chdir(self.tmp)
        self.assertEqual(config.AppConfig.load().local_providers, ["from-cwd"])

    def test_falls_back_to_the_repo_root_when_the_cwd_has_none(self):
        empty = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        os.chdir(empty)
        loaded = config.AppConfig.load()
        self.assertEqual(loaded.reasoning_schema.dimension_names, list(REASONING_DIMENSIONS))

    def test_falls_back_to_the_packaged_default_when_no_file_exists(self):
        """An install from requirements_prod.txt has no config.json anywhere near the cwd."""
        loaded = config.AppConfig.load(Path(self.tmp) / "absent.json")
        self.assertEqual(loaded.reasoning_schema.dimension_names, list(REASONING_DIMENSIONS))
        self.assertTrue(loaded.reasoning_schema.dimension_template)

    def test_unreadable_file_degrades_instead_of_raising(self):
        path = Path(self.tmp) / "config.json"
        path.write_text("{ not json", encoding="utf-8")
        with patch("prompttestenv.logger.log_warning"):
            loaded = config.AppConfig.load(path)
        self.assertEqual(loaded.reasoning_schema.dimension_names, list(REASONING_DIMENSIONS))

    def test_missing_sections_fall_back_to_field_defaults(self):
        loaded = config.AppConfig.load(self._write(self.tmp, {}))
        self.assertEqual(loaded.reasoning_defaults.dimension_mode, "split")
        self.assertEqual(loaded.unit_splitting.min_unit_chars, 15)
        self.assertEqual(loaded.reasoning_schema.dimensions, [])

    def test_unknown_keys_are_ignored(self):
        path = self._write(self.tmp, {"unit_splitting": {"min_unit_chars": 9, "future_knob": 1}})
        self.assertEqual(config.AppConfig.load(path).unit_splitting.min_unit_chars, 9)

    def test_schema_stamp_tracks_meaning_not_presentation(self):
        base = config.AppConfig.load().reasoning_schema

        recoloured = config.AppConfig.load().reasoning_schema
        recoloured.dimensions[0].color = "#000000"
        self.assertEqual(base.stamp, recoloured.stamp, "a colour is presentation, not measurement")

        reworded = config.AppConfig.load().reasoning_schema
        reworded.dimensions[0].definition += " (reworded)"
        self.assertNotEqual(base.stamp, reworded.stamp)

    def test_get_app_config_caches_until_asked_to_reload(self):
        first = config.get_app_config()
        self.assertIs(config.get_app_config(), first)
        self.assertIsNot(config.get_app_config(reload=True), first)

    def test_shipped_default_matches_the_repo_root_config(self):
        """Two copies exist so a prod install still resolves one; they must not drift apart."""
        root = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
        packaged = json.loads(
            importlib.resources.files("prompttestenv")
            .joinpath("templates/default_config.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(root, packaged)


if __name__ == "__main__":
    unittest.main()
