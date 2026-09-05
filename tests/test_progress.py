from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv.progress import (
    GEN_TIMEOUT_TEXT,
    HASHED_FILENAMES,
    JUDGE_TIMEOUT_TEXT,
    append_event,
    calculate_config_hash,
    config_hash_from_bytes,
    failed_eval_keys,
    failed_gen_keys,
    hashable_bytes,
    is_failed_eval,
    is_failed_gen,
    read_stored_hash,
)


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


class TestConfigHashFromBytes(unittest.TestCase):
    """The in-memory hash predictor an editor uses before writing anything."""

    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def _write(self, name, text):
        (Path(self.project_dir) / name).write_text(text, encoding="utf-8")

    def _disk_bytes(self):
        out = {}
        for name in HASHED_FILENAMES:
            path = Path(self.project_dir) / name
            out[name] = path.read_bytes() if path.exists() else None
        return out

    def test_matches_calculate_config_hash_on_the_same_content(self):
        self._write("candidates.json", '[{"name": "A"}]')
        self._write("judge_config.json", '{"repetitions": 3, "reasoning_analysis": "best"}')
        self._write("test_cases.json", '[{"id": "t1"}]')
        self._write("global_criteria.json", '{"mode": "none"}')

        self.assertEqual(
            config_hash_from_bytes(self._disk_bytes()),
            calculate_config_hash(self.project_dir),
        )

    def test_absent_file_is_none_and_matches_a_genuinely_missing_one(self):
        self._write("candidates.json", "[]")
        self._write("test_cases.json", "[]")
        self._write("global_criteria.json", "{}")
        # judge_config.json deliberately not written.
        self.assertEqual(
            config_hash_from_bytes(self._disk_bytes()),
            calculate_config_hash(self.project_dir),
        )

    def test_missing_key_is_treated_as_absent(self):
        self.assertEqual(
            config_hash_from_bytes({}),
            config_hash_from_bytes({name: None for name in HASHED_FILENAMES}),
        )


class TestHashableBytes(unittest.TestCase):
    """judge_config.json is canonicalised; the other three are hashed raw."""

    def test_judge_config_ignores_reasoning_keys(self):
        base = hashable_bytes("judge_config.json", b'{"repetitions": 2}')
        with_reasoning = hashable_bytes(
            "judge_config.json",
            b'{"repetitions": 2, "reasoning_analysis": "all", '
            b'"reasoning_judge": {"provider": "google", "temperature": 0.9}}',
        )
        self.assertEqual(base, with_reasoning)

    def test_judge_config_ignores_key_order_and_indentation(self):
        compact = hashable_bytes("judge_config.json", b'{"b": 2, "a": 1}')
        pretty = hashable_bytes("judge_config.json", b'{\n    "a": 1,\n    "b": 2\n}\n')
        self.assertEqual(compact, pretty)

    def test_judge_config_still_sees_a_real_change(self):
        self.assertNotEqual(
            hashable_bytes("judge_config.json", b'{"repetitions": 2}'),
            hashable_bytes("judge_config.json", b'{"repetitions": 3}'),
        )

    def test_judge_config_degrades_to_raw_when_unparseable(self):
        raw = b'{"repetitions": 2,,,'
        self.assertEqual(hashable_bytes("judge_config.json", raw), raw)

    def test_other_files_are_byte_sensitive(self):
        """Whitespace-only edits DO invalidate a run for the raw-hashed files.

        This is what the editor's byte-faithful writer exists to avoid.
        """
        self.assertNotEqual(
            hashable_bytes("candidates.json", b'[{"name": "A"}]'),
            hashable_bytes("candidates.json", b'[\n    {\n        "name": "A"\n    }\n]'),
        )

    def test_none_is_the_missing_marker(self):
        self.assertEqual(hashable_bytes("candidates.json", None), b"missing")


class TestReadStoredHash(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_returns_none_without_a_log(self):
        self.assertIsNone(read_stored_hash(self.project_dir))

    def test_does_not_create_the_log(self):
        read_stored_hash(self.project_dir)
        self.assertFalse((Path(self.project_dir) / "progress.jsonl").exists())

    def test_reads_the_hash_off_the_meta_line(self):
        append_event(self.project_dir, {"type": "meta", "config_hash": "abc123"})
        append_event(self.project_dir, {"type": "gen", "cand_id": "A"})
        self.assertEqual(read_stored_hash(self.project_dir), "abc123")

    def test_returns_none_when_the_first_line_is_not_meta(self):
        append_event(self.project_dir, {"type": "gen", "cand_id": "A"})
        self.assertIsNone(read_stored_hash(self.project_dir))

    def test_returns_none_on_a_corrupt_first_line(self):
        (Path(self.project_dir) / "progress.jsonl").write_text("not json\n", encoding="utf-8")
        self.assertIsNone(read_stored_hash(self.project_dir))


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


def _gen(output):
    return {"type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0, "output": output}


def _eval(score=8, reason="fine", global_score=-1, g_reason="N/A"):
    return {
        "type": "eval", "cand_id": "A", "test_id": "t1", "rep": 0,
        "score": score, "reason": reason,
        "global_score": global_score, "g_reason": g_reason,
    }


class TestIsFailedGen(unittest.TestCase):
    def test_timeout_placeholder_is_failed(self):
        self.assertTrue(is_failed_gen(_gen(GEN_TIMEOUT_TEXT)))

    def test_a_real_answer_is_not_failed(self):
        self.assertFalse(is_failed_gen(_gen("Dear customer, ...")))

    def test_an_empty_answer_is_not_failed(self):
        """An empty response is a real measurement, not a failure to obtain one."""
        self.assertFalse(is_failed_gen(_gen("")))

    def test_an_event_without_output_is_not_failed(self):
        self.assertFalse(is_failed_gen({"type": "gen", "cand_id": "A"}))


class TestIsFailedEval(unittest.TestCase):
    def test_judge_timeout_is_failed(self):
        self.assertTrue(is_failed_eval(
            _eval(score=-1, reason=JUDGE_TIMEOUT_TEXT, global_score=-1, g_reason=JUDGE_TIMEOUT_TEXT)
        ))

    def test_error_prefix_is_failed(self):
        self.assertTrue(is_failed_eval(_eval(score=-1, reason="Error: boom")))

    def test_bare_error_from_the_dispatcher_is_failed(self):
        """The dispatcher's global_reasoning is the word 'Error', with no colon."""
        self.assertTrue(is_failed_eval(
            _eval(score=-1, reason="Error: boom", global_score=-1, g_reason="Error")
        ))

    def test_llm_evaluation_failed_is_failed(self):
        self.assertTrue(is_failed_eval(_eval(score=-1, reason="LLM evaluation failed: 429")))

    def test_template_error_is_failed(self):
        """'Error in evaluation template: ...' is why the prefix carries no colon."""
        self.assertTrue(is_failed_eval(
            _eval(score=-1, reason="Error in evaluation template: missing field 'criteria'")
        ))

    def test_global_mode_none_is_not_failed(self):
        """global_criteria.mode 'none' stores -1/'N/A' forever, by design."""
        self.assertFalse(is_failed_eval(_eval(score=8, reason="good", global_score=-1, g_reason="N/A")))

    def test_deliberate_assert_minus_one_is_not_failed(self):
        """An assert lambda may return -1 to mean 'not applicable'. That is the author's call."""
        self.assertFalse(is_failed_eval(
            _eval(score=-1, reason="Not applicable to this response.")
        ))

    def test_a_real_score_mentioning_errors_is_not_failed(self):
        """The score == -1 conjunct is what stops a judge's prose triggering a retry."""
        self.assertFalse(is_failed_eval(_eval(score=8, reason="Errors were found in the answer.")))

    def test_a_failed_global_side_alone_is_failed(self):
        self.assertTrue(is_failed_eval(
            _eval(score=8, reason="good", global_score=-1, g_reason="Error: judge down")
        ))


class TestFailedKeySets(unittest.TestCase):
    def test_selects_only_the_failed_gen_keys(self):
        events = {
            ("A", "t1", 0): _gen(GEN_TIMEOUT_TEXT),
            ("A", "t1", 1): _gen("a real answer"),
            ("B", "t2", 0): _gen(GEN_TIMEOUT_TEXT),
        }
        self.assertEqual(failed_gen_keys(events), {("A", "t1", 0), ("B", "t2", 0)})

    def test_selects_only_the_failed_eval_keys(self):
        events = {
            ("A", "t1", 0): _eval(score=-1, reason="Error: boom"),
            ("A", "t1", 1): _eval(score=7, reason="fine"),
            ("B", "t2", 0): _eval(score=-1, reason="Author says not applicable"),
        }
        self.assertEqual(failed_eval_keys(events), {("A", "t1", 0)})

    def test_empty_log_selects_nothing(self):
        self.assertEqual(failed_gen_keys({}), set())
        self.assertEqual(failed_eval_keys({}), set())


if __name__ == "__main__":
    unittest.main()
