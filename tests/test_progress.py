from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompttestenv.progress import (
    GEN_TIMEOUT_TEXT,
    HASHED_FILENAMES,
    JUDGE_TIMEOUT_TEXT,
    ProgressState,
    RunKey,
    append_event,
    calculate_config_hash,
    client_version,
    config_hash_from_bytes,
    failed_eval_keys,
    failed_gen_keys,
    hashable_bytes,
    is_failed_eval,
    is_failed_gen,
    read_stored_hash,
)
from testutils import LoggerResetTestCase


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


class TestRunKey(unittest.TestCase):
    """A NamedTuple standing in for the plain 3-tuple it replaced: every use
    as a dict key, set member, and unpacking target must keep working
    against code that still builds or expects a literal (a, b, c)."""

    def test_equal_to_the_literal_tuple_it_replaces(self):
        self.assertEqual(RunKey("gpt", "t1", 0), ("gpt", "t1", 0))

    def test_hashes_the_same_as_the_literal_tuple(self):
        self.assertEqual(hash(RunKey("gpt", "t1", 0)), hash(("gpt", "t1", 0)))

    def test_a_literal_tuple_looks_up_a_dict_keyed_by_runkey(self):
        events = {RunKey("gpt", "t1", 0): {"score": 7}}
        self.assertEqual(events[("gpt", "t1", 0)], {"score": 7})

    def test_a_runkey_looks_up_a_dict_keyed_by_the_literal_tuple(self):
        events = {("gpt", "t1", 0): {"score": 7}}
        self.assertEqual(events[RunKey("gpt", "t1", 0)], {"score": 7})

    def test_positional_unpacking(self):
        cand, test_id, rep = RunKey("gpt", "t1", 2)
        self.assertEqual((cand, test_id, rep), ("gpt", "t1", 2))

    def test_field_access(self):
        key = RunKey("gpt", "t1", 2)
        self.assertEqual(key.candidate, "gpt")
        self.assertEqual(key.test_id, "t1")
        self.assertEqual(key.repetition, 2)

    def test_membership_in_a_set_of_literal_tuples(self):
        completed = {("gpt", "t1", 0), ("gpt", "t1", 1)}
        self.assertIn(RunKey("gpt", "t1", 0), completed)


class TestProgressStateLoad(LoggerResetTestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        # ProgressState.load hashes candidates/judge_config/test_cases/global_criteria.json,
        # so give it something deterministic to hash even though it doesn't validate content.
        for name in ("candidates.json", "judge_config.json", "test_cases.json", "global_criteria.json"):
            (Path(self.project_dir) / name).write_text("{}", encoding="utf-8")

    def test_no_existing_file_creates_one_with_meta_line(self):
        state = ProgressState.load(self.project_dir)
        self.assertTrue(state.hash_match)
        self.assertEqual(state.events, [])
        progress_file = Path(self.project_dir) / "progress.jsonl"
        self.assertTrue(progress_file.exists())
        meta = json.loads(progress_file.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(meta["type"], "meta")
        self.assertEqual(meta["client_version"], client_version())

    def _write_meta(self, **extra) -> None:
        meta = {"type": "meta", "config_hash": calculate_config_hash(self.project_dir), **extra}
        (Path(self.project_dir) / "progress.jsonl").write_text(
            json.dumps(meta) + "\n", encoding="utf-8",
        )

    def test_resuming_under_another_client_version_warns_and_still_resumes(self):
        self._write_meta(client_version="0.0.1")
        with patch("prompttestenv.logger.log_warning") as mock_warn:
            state = ProgressState.load(self.project_dir)
        self.assertTrue(state.hash_match)
        mock_warn.assert_called_once()
        self.assertIn("0.0.1", mock_warn.call_args.args[0])

    def test_same_or_unrecorded_client_version_is_silent(self):
        for extra in ({"client_version": client_version()}, {}):
            with self.subTest(extra=extra):
                self._write_meta(**extra)
                with patch("prompttestenv.logger.log_warning") as mock_warn:
                    ProgressState.load(self.project_dir)
                mock_warn.assert_not_called()

    def test_a_line_of_the_wrong_shape_is_skipped_not_fatal(self):
        """One odd line must not cost the whole log.

        The reader caught JSONDecodeError but then indexed event["type"], so a
        line that is valid JSON of the wrong shape raised TypeError out of
        load() - and since run, analyze and render all go through here, that
        made the project unreadable by every command, render included.
        """
        ProgressState.load(self.project_dir)
        good = {"type": "gen", "cand_id": "C", "test_id": "t", "rep": 0,
                "output": "real answer", "tokens": 5, "reasoning_tokens": 0,
                "reasoning_text": "", "elapsed": 1.0}
        with open(Path(self.project_dir) / "progress.jsonl", "a", encoding="utf-8") as f:
            print(json.dumps("a bare string line"), file=f)
            print(json.dumps({"no_type": "at all"}), file=f)
            print(json.dumps([1, 2, 3]), file=f)
            print(json.dumps(good), file=f)

        state = ProgressState.load(self.project_dir)

        self.assertIn(("C", "t", 0), state.completed_gen)
        self.assertEqual(state.gen_events[("C", "t", 0)]["output"], "real answer")

    def test_an_event_missing_a_key_the_reader_needs_is_skipped(self):
        ProgressState.load(self.project_dir)
        with open(Path(self.project_dir) / "progress.jsonl", "a", encoding="utf-8") as f:
            print(json.dumps({"type": "gen", "cand_id": "C"}), file=f)

        state = ProgressState.load(self.project_dir)

        self.assertEqual(state.completed_gen, set())

    def test_hash_mismatch_renames_to_bak(self):
        ProgressState.load(self.project_dir)  # creates progress.jsonl with current hash
        # Now change a config file so the stored hash no longer matches.
        (Path(self.project_dir) / "candidates.json").write_text('{"changed": true}', encoding="utf-8")
        state = ProgressState.load(self.project_dir)
        self.assertFalse(state.hash_match)
        self.assertTrue((Path(self.project_dir) / "progress.jsonl.bak").exists())

    def test_force_restart_discards_existing_progress(self):
        ProgressState.load(self.project_dir)
        progress_file = Path(self.project_dir) / "progress.jsonl"
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0, "output": "x", "tokens": 1, "reasoning_tokens": 0, "elapsed": 0.1}) + "\n")

        state = ProgressState.load(self.project_dir, force_restart=True)
        self.assertEqual(state.events, [])

    def test_completed_gen_and_eval_sets_populated(self):
        progress_file = Path(self.project_dir) / "progress.jsonl"
        current_hash = calculate_config_hash(self.project_dir)
        lines = [
            json.dumps({"type": "meta", "config_hash": current_hash}),
            json.dumps({"type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0, "output": "x", "tokens": 1, "reasoning_tokens": 0, "elapsed": 0.1}),
            json.dumps({"type": "eval", "cand_id": "A", "test_id": "t1", "rep": 0, "score": 8, "global_score": -1, "reason": "ok", "g_reason": "N/A"}),
        ]
        progress_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        state = ProgressState.load(self.project_dir)
        self.assertIn(("A", "t1", 0), state.completed_gen)
        self.assertIn(("A", "t1", 0), state.completed_eval)
        self.assertEqual(len(state.events), 2)

    def test_trailing_corrupted_line_is_ignored(self):
        progress_file = Path(self.project_dir) / "progress.jsonl"
        current_hash = calculate_config_hash(self.project_dir)
        progress_file.write_text(
            json.dumps({"type": "meta", "config_hash": current_hash}) + "\n" + "{not valid json\n",
            encoding="utf-8",
        )
        state = ProgressState.load(self.project_dir)
        self.assertTrue(state.hash_match)
        self.assertEqual(state.events, [])


class TestProgressStateStaleness(LoggerResetTestCase):
    """The log's line ORDER is what records an interrupted retry.

    An append-only log supersedes a failed step by adding a corrected event, so
    an eval (or reasoning, or verdict) with newer results after it was computed
    against text that is no longer stored. Nothing else in the state can see
    that: the failure predicates read the surviving values, not their position.
    """

    KEY = ("A", "t1", 0)

    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        for name in ("candidates.json", "judge_config.json", "test_cases.json", "global_criteria.json"):
            (Path(self.project_dir) / name).write_text("{}", encoding="utf-8")

    def _load(self, *events):
        lines = [json.dumps({"type": "meta", "config_hash": calculate_config_hash(self.project_dir)})]
        lines += [json.dumps(event) for event in events]
        (Path(self.project_dir) / "progress.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return ProgressState.load(self.project_dir, readonly=True)

    def _gen(self, output="answer"):
        return {"type": "gen", "cand_id": "A", "test_id": "t1", "rep": 0,
                "output": output, "tokens": 1, "reasoning_tokens": 0,
                "reasoning_text": "a trace", "elapsed": 0.1}

    def _eval(self, score=8):
        return {"type": "eval", "cand_id": "A", "test_id": "t1", "rep": 0,
                "score": score, "global_score": -1, "reason": "ok", "g_reason": "N/A"}

    def _reasoning(self):
        return {"type": "reasoning", "cand_id": "A", "test_id": "t1", "rep": 0}

    def test_a_gen_after_its_eval_makes_the_eval_stale(self):
        state = self._load(self._gen(), self._eval(), self._gen("regenerated"))
        self.assertEqual(state.stale_eval_keys, frozenset({self.KEY}))

    def test_the_normal_order_is_not_stale(self):
        state = self._load(self._gen(), self._eval())
        self.assertEqual(state.stale_eval_keys, frozenset())

    def test_a_gen_with_no_eval_is_not_stale(self):
        """Unjudged, not stale: the evaluation phase already picks it up."""
        state = self._load(self._gen())
        self.assertEqual(state.stale_eval_keys, frozenset())

    def test_staleness_is_self_healing(self):
        """Once the eval is re-appended it follows the gen again, by itself."""
        state = self._load(self._gen(), self._eval(), self._gen("regenerated"), self._eval(9))
        self.assertEqual(state.stale_eval_keys, frozenset())

    def test_a_gen_after_its_reasoning_makes_the_analysis_stale(self):
        state = self._load(self._gen(), self._reasoning(), self._gen("regenerated"))
        self.assertEqual(state.stale_reasoning_keys, frozenset({self.KEY}))

    def test_results_after_the_verdict_make_it_stale(self):
        state = self._load(
            self._gen(), self._eval(), {"type": "verdict", "content": "V"},
            self._gen("regenerated"), self._eval(9),
        )
        self.assertTrue(state.stale_verdict)

    def test_a_verdict_written_last_is_not_stale(self):
        state = self._load(self._gen(), self._eval(), {"type": "verdict", "content": "V"})
        self.assertFalse(state.stale_verdict)

    def test_no_verdict_at_all_is_not_stale(self):
        state = self._load(self._gen(), self._eval())
        self.assertFalse(state.stale_verdict)

    def test_a_later_reasoning_event_does_not_stale_the_verdict(self):
        """`analyze` is meant to be re-run over a finished project whenever the
        schema is retuned, so it must not cost a verdict call every time."""
        state = self._load(
            self._gen(), self._eval(), {"type": "verdict", "content": "V"}, self._reasoning(),
        )
        self.assertFalse(state.stale_verdict)

    def test_early_return_paths_leave_the_signals_at_their_defaults(self):
        (Path(self.project_dir) / "progress.jsonl").write_text(
            json.dumps({"type": "meta", "config_hash": "deadbeef"}) + "\n", encoding="utf-8"
        )
        state = ProgressState.load(self.project_dir, readonly=True)
        self.assertFalse(state.hash_match)
        self.assertEqual(state.stale_eval_keys, frozenset())
        self.assertEqual(state.stale_reasoning_keys, frozenset())
        self.assertFalse(state.stale_verdict)


if __name__ == "__main__":
    unittest.main()
