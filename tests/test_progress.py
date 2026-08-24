from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv.progress import append_event, calculate_config_hash


class TestCalculateConfigHash(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def _write_configs(self, candidates="[]", judge="{}", tests="[]", global_criteria="{}"):
        (Path(self.project_dir) / "candidates.json").write_text(candidates, encoding="utf-8")
        (Path(self.project_dir) / "judge_config.json").write_text(judge, encoding="utf-8")
        (Path(self.project_dir) / "test_cases.json").write_text(tests, encoding="utf-8")
        (Path(self.project_dir) / "global_criteria.json").write_text(global_criteria, encoding="utf-8")

    def test_deterministic_for_same_content(self):
        self._write_configs()
        h1 = calculate_config_hash(self.project_dir)
        h2 = calculate_config_hash(self.project_dir)
        self.assertEqual(h1, h2)

    def test_changes_when_any_file_changes(self):
        self._write_configs()
        h1 = calculate_config_hash(self.project_dir)
        (Path(self.project_dir) / "candidates.json").write_text('[{"name": "x"}]', encoding="utf-8")
        h2 = calculate_config_hash(self.project_dir)
        self.assertNotEqual(h1, h2)

    def test_missing_files_hash_the_literal_missing_marker(self):
        # No files written at all — every one of the 4 hashes b"missing".
        h_missing = calculate_config_hash(self.project_dir)

        import hashlib
        hasher = hashlib.md5()
        for _ in range(4):
            hasher.update(b"missing")
        self.assertEqual(h_missing, hasher.hexdigest())


class TestAppendEvent(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_appends_one_json_line_per_call(self):
        append_event(self.project_dir, {"type": "gen", "cand_id": "A"})
        append_event(self.project_dir, {"type": "eval", "cand_id": "A"})

        progress_file = Path(self.project_dir) / "progress.jsonl"
        lines = progress_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["type"], "gen")
        self.assertEqual(json.loads(lines[1])["type"], "eval")


if __name__ == "__main__":
    unittest.main()
