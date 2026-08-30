"""Tests for the headless project-editing API.

The headline case is `test_an_empty_patch_writes_nothing_at_all`: the whole
point of a patch language over "read it, edit it, write it back" is that the
byte-fidelity invariant the Streamlit editor keeps survives the trip through a
CLI or an LLM. Everything else here exists to keep that true, plus the two
guards that stop an agent destroying work it cannot see: the hash gate and the
referential check on asset deletion.

No LLM call is made anywhere in this module. This layer makes none.
"""
from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from prompttestenv import projectedit as pe
from prompttestenv import projectio as pio
from prompttestenv.progress import HASHED_FILENAMES, append_event, calculate_config_hash

from testutils import make_temp_project


class ProjectEditTestCase(unittest.TestCase):
    """A throwaway copy of the smoke fixture, with byte snapshots to compare."""

    def setUp(self):
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def _bytes(self) -> dict[str, bytes | None]:
        base = Path(self.project_dir)
        return {
            name: (base / name).read_bytes() if (base / name).exists() else None
            for name in HASHED_FILENAMES
        }

    def _json(self, name: str):
        return json.loads((Path(self.project_dir) / name).read_text(encoding="utf-8"))

    def _stamp_progress(self) -> str:
        """Give the project a progress.jsonl matching its current config."""
        stored = calculate_config_hash(self.project_dir)
        append_event(self.project_dir, {"type": "meta", "config_hash": stored})
        return stored


class TestReadProject(ProjectEditTestCase):
    def test_reports_the_files_verbatim_and_the_assets_asymmetrically(self):
        data = pe.read_project(self.project_dir)
        self.assertTrue(data["exists"])
        self.assertEqual(data["candidates"], self._json("candidates.json"))
        self.assertEqual(data["judge_config"], self._json("judge_config.json"))
        # system prompts carry their text, attachments only their size
        self.assertTrue(all(isinstance(v, str) for v in data["system_prompts"].values()))
        for entry in data["test_files"]:
            self.assertEqual(set(entry), {"name", "path", "bytes"})

    def test_a_project_with_no_log_is_reported_as_still_valid(self):
        data = pe.read_project(self.project_dir)
        self.assertIsNone(data["stored_hash"])
        self.assertTrue(data["progress_valid"])

    def test_a_stale_log_is_reported_as_invalid(self):
        self._stamp_progress()
        pe.edit_project(self.project_dir, {"judge_config": {"repetitions": 9}}, force=True)
        data = pe.read_project(self.project_dir)
        self.assertFalse(data["progress_valid"])
        self.assertNotEqual(data["stored_hash"], data["config_hash"])

    def test_a_missing_directory_reads_as_empty_rather_than_raising(self):
        data = pe.read_project(str(Path(self.project_dir) / "nope"))
        self.assertFalse(data["exists"])
        self.assertEqual(data["candidates"], [])


class TestNoOpPatch(ProjectEditTestCase):
    def test_an_empty_patch_writes_nothing_at_all(self):
        before = self._bytes()
        result = pe.edit_project(self.project_dir, {})
        self.assertTrue(result.ok)
        self.assertEqual(result.written, [])
        self.assertEqual(self._bytes(), before)

    def test_resending_a_value_a_file_already_holds_writes_nothing(self):
        current = self._json("candidates.json")[0]
        before = self._bytes()
        result = pe.edit_project(self.project_dir, {"candidates": [dict(current)]})
        self.assertTrue(result.ok)
        self.assertEqual(result.written, [])
        self.assertEqual(self._bytes(), before)

    def test_only_the_file_the_patch_touches_is_rewritten(self):
        before = self._bytes()
        result = pe.edit_project(
            self.project_dir,
            {"candidates": [{"name": self._json("candidates.json")[0]["name"],
                             "temperature": 0.123}]},
        )
        self.assertEqual(result.written, ["candidates.json"])
        after = self._bytes()
        for name in HASHED_FILENAMES:
            if name != "candidates.json":
                self.assertEqual(after[name], before[name], name)


class TestUpsertSemantics(ProjectEditTestCase):
    def test_an_existing_candidate_keeps_its_other_keys_and_their_order(self):
        original = self._json("candidates.json")[0]
        pe.edit_project(
            self.project_dir,
            {"candidates": [{"name": original["name"], "temperature": 0.42}]},
        )
        updated = self._json("candidates.json")[0]
        self.assertEqual(updated["temperature"], 0.42)
        self.assertEqual(list(updated), list(original))
        for key, value in original.items():
            if key != "temperature":
                self.assertEqual(updated[key], value, key)

    def test_a_new_candidate_is_appended_with_its_required_keys(self):
        count = len(self._json("candidates.json"))
        result = pe.edit_project(
            self.project_dir,
            {"candidates": [{"name": "Fresh", "model": "gemini-3-flash"}]},
        )
        self.assertTrue(result.ok, result.errors)
        candidates = self._json("candidates.json")
        self.assertEqual(len(candidates), count + 1)
        self.assertEqual(candidates[-1],
                         {"name": "Fresh", "provider": "google", "model": "gemini-3-flash"})

    def test_an_entry_with_no_identity_key_is_rejected(self):
        result = pe.edit_project(self.project_dir, {"candidates": [{"model": "m"}]})
        self.assertFalse(result.ok)
        self.assertIn("identifies it", result.errors[0])

    def test_a_test_case_attachment_accepts_a_string_a_list_and_null(self):
        pe.edit_project(self.project_dir, {"test_files": {"a.txt": "a", "b.txt": "b"}})
        tid = self._json("test_cases.json")[0]["id"]

        pe.edit_project(self.project_dir,
                        {"test_cases": [{"id": tid, "file": ["test_files/a.txt"]}]})
        self.assertEqual(self._json("test_cases.json")[0]["file"], "test_files/a.txt")

        pe.edit_project(
            self.project_dir,
            {"test_cases": [{"id": tid,
                             "file": ["test_files/a.txt", "test_files/b.txt"]}]},
        )
        self.assertEqual(self._json("test_cases.json")[0]["file"],
                         ["test_files/a.txt", "test_files/b.txt"])

        pe.edit_project(self.project_dir, {"test_cases": [{"id": tid, "file": None}]})
        self.assertNotIn("file", self._json("test_cases.json")[0])

    def test_backslashes_in_an_attachment_path_are_normalised(self):
        pe.edit_project(self.project_dir, {"test_files": {"a.txt": "a"}})
        tid = self._json("test_cases.json")[0]["id"]
        pe.edit_project(self.project_dir,
                        {"test_cases": [{"id": tid, "file": "test_files\\a.txt"}]})
        self.assertEqual(self._json("test_cases.json")[0]["file"], "test_files/a.txt")


class TestJudgeAndCriteria(ProjectEditTestCase):
    def test_a_nested_block_merges_without_disturbing_its_siblings(self):
        before = self._json("judge_config.json")
        pe.edit_project(self.project_dir,
                        {"judge_config": {"test_judge": {"temperature": 0.55}}})
        after = self._json("judge_config.json")
        self.assertEqual(after["test_judge"]["temperature"], 0.55)
        self.assertEqual(after["verdict_judge"], before["verdict_judge"])
        self.assertEqual(list(after), list(before))

    def test_global_criteria_merges_in_place(self):
        before = self._json("global_criteria.json")
        pe.edit_project(self.project_dir, {"global_criteria": {"mode": "none"}})
        after = self._json("global_criteria.json")
        self.assertEqual(after["mode"], "none")
        self.assertEqual(list(after), list(before))

    def test_a_section_of_the_wrong_shape_is_rejected(self):
        for patch in ({"judge_config": []}, {"global_criteria": "x"}, {"candidates": {}}):
            with self.subTest(patch=patch):
                self.assertFalse(pe.edit_project(self.project_dir, patch).ok)


class TestUnknownKeys(ProjectEditTestCase):
    def test_an_unknown_top_level_key_is_an_error_not_a_no_op(self):
        """A typo in a machine-written patch must never pass silently."""
        result = pe.edit_project(self.project_dir, {"candidate": [{"name": "X"}]})
        self.assertFalse(result.ok)
        self.assertIn("Unknown patch key 'candidate'", result.errors[0])

    def test_an_unknown_delete_key_is_rejected(self):
        result = pe.edit_project(self.project_dir, {"delete": {"reports": ["x"]}})
        self.assertFalse(result.ok)

    def test_a_patch_that_is_not_an_object_is_rejected(self):
        self.assertFalse(pe.edit_project(self.project_dir, [1, 2]).ok)


class TestDelete(ProjectEditTestCase):
    def test_deleting_a_candidate_removes_only_that_entry(self):
        names = [c["name"] for c in self._json("candidates.json")]
        result = pe.edit_project(self.project_dir, {"delete": {"candidates": [names[0]]}})
        self.assertTrue(result.ok, result.errors)
        self.assertEqual([c["name"] for c in self._json("candidates.json")], names[1:])

    def test_deleting_something_absent_is_an_error(self):
        result = pe.edit_project(self.project_dir, {"delete": {"candidates": ["ghost"]}})
        self.assertFalse(result.ok)
        self.assertIn("no entry with name", result.errors[0])

    def test_an_asset_still_referenced_cannot_be_deleted(self):
        pe.edit_project(self.project_dir, {"system_prompts": {"terse.txt": "Be terse."}})
        pe.edit_project(
            self.project_dir,
            {"candidates": [{"name": "T", "model": "m", "system_prompt_file": "terse.txt"}]},
        )
        result = pe.edit_project(self.project_dir,
                                 {"delete": {"system_prompts": ["terse.txt"]}})
        self.assertFalse(result.ok)
        self.assertIn("still used by", result.errors[0])
        self.assertTrue((Path(self.project_dir) / "system_prompts" / "terse.txt").exists())

    def test_dropping_the_asset_and_its_only_user_in_one_patch_is_allowed(self):
        """The check runs against the post-patch draft, not what is on disk."""
        pe.edit_project(self.project_dir, {"system_prompts": {"terse.txt": "Be terse."}})
        pe.edit_project(
            self.project_dir,
            {"candidates": [{"name": "T", "model": "m", "system_prompt_file": "terse.txt"}]},
        )
        result = pe.edit_project(
            self.project_dir,
            {"delete": {"candidates": ["T"], "system_prompts": ["terse.txt"]}},
        )
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.deleted, ["system_prompts/terse.txt"])
        self.assertFalse((Path(self.project_dir) / "system_prompts" / "terse.txt").exists())


class TestOrder(ProjectEditTestCase):
    def test_reordering_moves_the_whole_entry(self):
        names = [c["name"] for c in self._json("candidates.json")]
        result = pe.edit_project(self.project_dir,
                                 {"order": {"candidates": list(reversed(names))}})
        self.assertTrue(result.ok, result.errors)
        self.assertEqual([c["name"] for c in self._json("candidates.json")],
                         list(reversed(names)))

    def test_a_partial_order_is_rejected_rather_than_dropping_entries(self):
        names = [c["name"] for c in self._json("candidates.json")]
        result = pe.edit_project(self.project_dir, {"order": {"candidates": names[:1]}})
        self.assertFalse(result.ok)
        self.assertIn("every entry exactly once", result.errors[0])


class TestHashGate(ProjectEditTestCase):
    def test_an_edit_that_invalidates_a_run_is_refused_by_default(self):
        self._stamp_progress()
        before = self._bytes()
        result = pe.edit_project(self.project_dir, {"judge_config": {"repetitions": 9}})
        self.assertFalse(result.ok)
        self.assertTrue(result.hash_changed)
        self.assertIn("progress.jsonl", result.errors[0])
        self.assertEqual(self._bytes(), before)

    def test_force_writes_it_anyway_and_never_deletes_the_log(self):
        self._stamp_progress()
        result = pe.edit_project(self.project_dir,
                                 {"judge_config": {"repetitions": 9}}, force=True)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(self._json("judge_config.json")["repetitions"], 9)
        self.assertTrue((Path(self.project_dir) / "progress.jsonl").exists())

    def test_a_reasoning_only_edit_never_trips_the_gate(self):
        """reasoning_analysis and reasoning_judge are stripped before hashing."""
        self._stamp_progress()
        result = pe.edit_project(self.project_dir,
                                 {"judge_config": {"reasoning_analysis": "best"}})
        self.assertTrue(result.ok, result.errors)
        self.assertFalse(result.hash_changed)
        self.assertEqual(self._json("judge_config.json")["reasoning_analysis"], "best")

    def test_a_project_with_no_log_needs_no_force(self):
        result = pe.edit_project(self.project_dir, {"judge_config": {"repetitions": 9}})
        self.assertTrue(result.ok, result.errors)
        self.assertFalse(result.hash_changed)


class TestDryRun(ProjectEditTestCase):
    def test_it_reports_what_would_change_and_writes_nothing(self):
        before = self._bytes()
        result = pe.edit_project(self.project_dir,
                                 {"judge_config": {"repetitions": 9}}, dry_run=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.written, ["judge_config.json"])
        self.assertEqual(self._bytes(), before)

    def test_it_answers_instead_of_refusing_over_the_hash(self):
        """Asking what an edit would do must not itself require force."""
        self._stamp_progress()
        result = pe.edit_project(self.project_dir,
                                 {"judge_config": {"repetitions": 9}}, dry_run=True)
        self.assertTrue(result.ok)
        self.assertTrue(result.hash_changed)
        self.assertIn("force", result.summary())

    def test_a_staged_asset_is_not_written(self):
        result = pe.edit_project(self.project_dir,
                                 {"system_prompts": {"x.txt": "body"}}, dry_run=True)
        self.assertEqual(result.written, ["system_prompts/x.txt"])
        self.assertFalse((Path(self.project_dir) / "system_prompts" / "x.txt").exists())


class TestAssetsThroughAPatch(ProjectEditTestCase):
    def test_a_system_prompt_is_written_and_flagged_as_unhashed(self):
        result = pe.edit_project(self.project_dir,
                                 {"system_prompts": {"terse.txt": "Be terse."}})
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.written, ["system_prompts/terse.txt"])
        path = Path(self.project_dir) / "system_prompts" / "terse.txt"
        self.assertEqual(path.read_text(encoding="utf-8"), "Be terse.")
        self.assertTrue(any("not part of the run hash" in w for w in result.warnings))

    def test_writing_an_asset_alone_never_touches_a_config_file(self):
        before = self._bytes()
        pe.edit_project(self.project_dir, {"test_files": {"notes.md": "hello"}})
        self.assertEqual(self._bytes(), before)

    def test_a_prompt_and_the_candidate_using_it_land_in_one_patch(self):
        result = pe.edit_project(self.project_dir, {
            "system_prompts": {"terse.txt": "Be terse."},
            "candidates": [{"name": "Terse", "model": "m",
                            "system_prompt_file": "terse.txt"}],
        })
        self.assertTrue(result.ok, result.errors)
        # The prompt is staged, not yet on disk, when validate() runs. It must
        # still count as present, or every such patch would warn about a file it
        # is creating in the same call.
        self.assertFalse(any("terse.txt" in w and "missing" in w for w in result.warnings))

    def test_a_bad_filename_and_a_wrong_suffix_are_refused(self):
        for spec in ({"sub/x.txt": "b"}, {"x.md": "b"}, {"..": "b"}):
            with self.subTest(spec=spec):
                self.assertFalse(
                    pe.edit_project(self.project_dir, {"system_prompts": spec}).ok
                )

    def test_binary_content_is_refused_with_an_explanation(self):
        result = pe.edit_project(self.project_dir, {"test_files": {"x.bin": [0, 1]}})
        self.assertFalse(result.ok)
        self.assertIn("must be text", result.errors[0])

    def test_a_null_value_points_at_the_delete_key(self):
        result = pe.edit_project(self.project_dir, {"system_prompts": {"x.txt": None}})
        self.assertFalse(result.ok)
        self.assertIn("delete.system_prompts", result.errors[0])


class TestValidationGate(ProjectEditTestCase):
    def test_a_project_already_holding_a_duplicate_name_cannot_be_edited(self):
        """upsert cannot create a duplicate, but a hand-edited file can hold one.

        The gate then blocks every later patch until it is fixed, exactly as the
        editor's Save button does.
        """
        existing = self._json("candidates.json")[0]["name"]
        draft = pio.load_project(self.project_dir)
        draft.candidates.append({"name": existing, "provider": "google", "model": "m"})
        pio.save_project(draft)
        before = self._bytes()

        result = pe.edit_project(self.project_dir, {"judge_config": {"repetitions": 4}})
        self.assertFalse(result.ok)
        self.assertTrue(any("used more than once" in e for e in result.errors))
        self.assertEqual(self._bytes(), before)

    def test_an_unparseable_assert_lambda_blocks_the_write(self):
        tid = self._json("test_cases.json")[0]["id"]
        before = self._bytes()
        result = pe.edit_project(self.project_dir, {
            "test_cases": [{"id": tid, "judge_type": "assert", "criteria": "s: ("}],
        })
        self.assertFalse(result.ok)
        self.assertEqual(self._bytes(), before)

    def test_a_missing_attachment_only_warns(self):
        tid = self._json("test_cases.json")[0]["id"]
        result = pe.edit_project(
            self.project_dir,
            {"test_cases": [{"id": tid, "file": "test_files/ghost.csv"}]},
        )
        self.assertTrue(result.ok, result.errors)
        self.assertTrue(any("ghost.csv" in w for w in result.warnings))


class TestFailureModes(ProjectEditTestCase):
    def test_a_missing_project_points_at_init(self):
        result = pe.edit_project(str(Path(self.project_dir) / "nope"), {})
        self.assertFalse(result.ok)
        self.assertIn("Use 'init' first", result.errors[0])

    def test_summary_prefixes_a_failure_the_way_the_runner_does(self):
        result = pe.edit_project(str(Path(self.project_dir) / "nope"), {})
        self.assertTrue(result.summary().startswith("Error:"))

    def test_to_dict_is_json_serialisable(self):
        result = pe.edit_project(self.project_dir, {})
        json.dumps(result.to_dict())  # must not raise
        self.assertIn("hash_changed", result.to_dict())


class TestParsePatch(unittest.TestCase):
    def test_valid_object(self):
        self.assertEqual(pe.parse_patch('{"candidates": []}'), {"candidates": []})

    def test_broken_json_and_non_objects_are_rejected(self):
        for text in ("{{{", "[1, 2]", '"x"'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                pe.parse_patch(text)


if __name__ == "__main__":
    unittest.main()
