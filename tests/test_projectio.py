"""Fidelity tests for the shared raw-dict load/serialise layer.

No Streamlit here: this is the layer both editing paths rest on, the Streamlit
editor and the headless patch API alike, and it is testable on its own. The
headline case is
`test_round_trip_is_byte_identical_and_hash_stable` — everything else exists to
keep that true as the code moves.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv import projectio as pio
from prompttestenv.progress import append_event, calculate_config_hash
from prompttestenv.models import (
    Candidate,
    GlobalCriteria,
    JudgeConfig,
    ReasoningJudgeSettings,
    SimilarityJudgeSettings,
    TestCase,
    TestJudgeSettings,
    VerdictJudgeSettings,
)
from prompttestenv.progress import HASHED_FILENAMES, calculate_config_hash, config_hash_from_bytes

from testutils import FIXTURES_DIR, make_temp_project

_FIXTURE_PROJECTS = ("smoke_project", "FeaturesTest", "QuickTest", "RemoteLLMTest", "LocalLLMTest")


class TestRoundTripFidelity(unittest.TestCase):
    """Opening a project and saving it unchanged must not touch a single byte."""

    def _copy_fixture(self, name):
        target = tempfile.mkdtemp(prefix="prompttestenv_test_")
        shutil.copytree(FIXTURES_DIR / name, target, dirs_exist_ok=True)
        self.addCleanup(shutil.rmtree, target, ignore_errors=True)
        return target

    def test_round_trip_is_byte_identical_and_hash_stable(self):
        for name in _FIXTURE_PROJECTS:
            with self.subTest(project=name):
                project_dir = self._copy_fixture(name)
                before = calculate_config_hash(project_dir)

                draft = pio.load_project(project_dir)
                pending = pio.serialize_all(draft)

                for filename in HASHED_FILENAMES:
                    self.assertEqual(
                        pending[filename], draft.disk[filename],
                        msg=f"{name}/{filename} would be rewritten by a no-op save",
                    )
                self.assertEqual(config_hash_from_bytes(pending), before)

    def test_no_op_save_writes_nothing(self):
        project_dir = self._copy_fixture("FeaturesTest")
        mtimes = {
            name: (Path(project_dir) / name).stat().st_mtime_ns for name in HASHED_FILENAMES
        }
        draft = pio.load_project(project_dir)

        self.assertEqual(pio.changed_files(draft), {})
        self.assertEqual(pio.save_project(draft), [])
        for name, mtime in mtimes.items():
            self.assertEqual((Path(project_dir) / name).stat().st_mtime_ns, mtime)

    def test_only_the_changed_file_is_written(self):
        project_dir = self._copy_fixture("FeaturesTest")
        draft = pio.load_project(project_dir)
        draft.judge["repetitions"] = 99

        self.assertEqual(pio.save_project(draft), ["judge_config.json"])
        self.assertEqual(JudgeConfig.load(project_dir).repetitions, 99)


class TestProjectDirScaffolding(unittest.TestCase):
    def setUp(self):
        self.target = Path(tempfile.mkdtemp(prefix="prompttestenv_test_")) / "NewProject"
        self.addCleanup(shutil.rmtree, self.target.parent, ignore_errors=True)

    def test_seed_creates_a_loadable_project(self):
        created = pio.seed_project(str(self.target))

        self.assertIn("candidates.json", created)
        self.assertIn("system_prompts/pirate_prompt.txt", created)
        self.assertTrue(Candidate.load_all(str(self.target)))
        self.assertTrue(TestCase.load_all(str(self.target)))
        self.assertEqual(JudgeConfig.load(str(self.target)).repetitions, 5)

    def test_seed_writes_no_secrets_file_anywhere(self):
        """init_project would drop one in the CWD; the editor must not."""
        pio.seed_project(str(self.target))
        self.assertFalse((self.target / "secrets.json").exists())
        self.assertFalse((self.target.parent / "secrets.json").exists())

    def test_seeded_project_round_trips_byte_identically(self):
        pio.seed_project(str(self.target))
        draft = pio.load_project(str(self.target))
        for name, payload in pio.serialize_all(draft).items():
            self.assertEqual(payload, draft.disk[name], msg=name)

    def test_seed_never_overwrites(self):
        pio.seed_project(str(self.target))
        (self.target / "candidates.json").write_text('[{"name": "mine"}]', encoding="utf-8")
        created = pio.seed_project(str(self.target))

        self.assertEqual(created, [])
        self.assertEqual(Candidate.load_all(str(self.target))[0].name, "mine")


class TestMergePreservingShape(unittest.TestCase):
    def test_key_order_follows_the_original(self):
        original = {"model": "m", "name": "A", "provider": "google"}
        merged = pio.merge_preserving_shape(original, {"name": "B"}, Candidate)
        self.assertEqual(list(merged), ["model", "name", "provider"])
        self.assertEqual(merged["name"], "B")

    def test_unknown_keys_survive_in_place(self):
        original = {"name": "A", "__note": "keep me", "model": "m"}
        merged = pio.merge_preserving_shape(original, {"name": "A"}, Candidate)
        self.assertEqual(list(merged), ["name", "__note", "model"])
        self.assertEqual(merged["__note"], "keep me")

    def test_absent_default_valued_key_stays_absent(self):
        original = {"name": "A", "provider": "google", "model": "m"}
        merged = pio.merge_preserving_shape(
            original, {"name": "A", "provider": "google", "model": "m", "thinking": "default"},
            Candidate,
        )
        self.assertNotIn("thinking", merged)

    def test_absent_non_default_key_gets_emitted(self):
        original = {"name": "A", "provider": "google", "model": "m"}
        merged = pio.merge_preserving_shape(
            original, {"thinking": "true"}, Candidate,
        )
        self.assertEqual(merged["thinking"], "true")
        self.assertEqual(list(merged)[-1], "thinking")

    def test_always_emit_keys_are_written_even_at_their_default(self):
        merged = pio.merge_preserving_shape(
            {}, {"name": "A", "provider": "google", "model": "m"}, Candidate,
        )
        self.assertEqual(merged, {"name": "A", "provider": "google", "model": "m"})

    def test_derived_field_is_never_written(self):
        original = {"name": "A", "resolved_system_instruction": "leaked"}
        merged = pio.merge_preserving_shape(
            original, {"resolved_system_instruction": "still leaked"}, Candidate,
        )
        self.assertNotIn("resolved_system_instruction", merged)


class TestPreserveForm(unittest.TestCase):
    """st.number_input hands back floats; an on-disk int must stay an int."""

    def test_int_stays_int_when_value_is_unchanged(self):
        self.assertIsInstance(pio.preserve_form(2, 2.0), int)
        self.assertIsInstance(pio.preserve_form(300, 300.0), int)

    def test_a_real_change_wins(self):
        self.assertEqual(pio.preserve_form(2, 3.0), 3.0)

    def test_float_on_disk_stays_float(self):
        self.assertIsInstance(pio.preserve_form(0.7, 0.7), float)

    def test_booleans_are_not_confused_with_one_and_zero(self):
        self.assertIs(pio.preserve_form(True, 1), 1)
        self.assertIs(pio.preserve_form(1, True), True)

    def test_survives_a_full_serialisation_round_trip(self):
        merged = pio.merge_preserving_shape(
            {"repetitions": 2}, {"repetitions": 2.0}, JudgeConfig,
        )
        payload = pio.serialize(merged, trailing_newline=True)
        self.assertIn(b'"repetitions": 2', payload)
        self.assertNotIn(b"2.0", payload)


class TestAttachmentValue(unittest.TestCase):
    """0 -> absent, 1 -> plain string, N -> list.

    A single-element list would rewrite every existing single-attachment
    project on a save that changed nothing.
    """

    def test_no_selection_drops_the_key(self):
        self.assertIsNone(pio.attachment_value([]))
        self.assertIsNone(pio.attachment_value([""]))

    def test_one_selection_stays_a_plain_string(self):
        self.assertEqual(pio.attachment_value(["test_files/a.txt"]), "test_files/a.txt")

    def test_several_selections_become_a_list(self):
        self.assertEqual(
            pio.attachment_value(["test_files/a.txt", "test_files/b.md"]),
            ["test_files/a.txt", "test_files/b.md"],
        )

    def test_separators_are_normalised_on_save(self):
        """The one deliberate exception to byte fidelity — see the module docstring."""
        self.assertEqual(pio.attachment_value(["test_files\\a.txt"]), "test_files/a.txt")

    def test_a_single_attachment_survives_a_serialisation_round_trip(self):
        merged = pio.merge_preserving_shape(
            {"id": "t", "prompt": "p", "criteria": "c", "file": "test_files/a.txt"},
            {"file": pio.attachment_value(["test_files/a.txt"])},
            TestCase,
        )
        self.assertEqual(merged["file"], "test_files/a.txt")


class TestTrailingNewline(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def _write(self, name, text):
        (Path(self.project_dir) / name).write_text(text, encoding="utf-8", newline="\n")

    def test_absence_and_presence_both_survive(self):
        self._write("candidates.json", "[]")             # no trailing newline
        self._write("judge_config.json", "{}\n")         # trailing newline
        self._write("test_cases.json", "[]")
        self._write("global_criteria.json", "{}")

        draft = pio.load_project(self.project_dir)
        self.assertFalse(draft.trailing_newline["candidates.json"])
        self.assertTrue(draft.trailing_newline["judge_config.json"])

        payloads = pio.serialize_all(draft)
        self.assertFalse(payloads["candidates.json"].endswith(b"\n"))
        self.assertTrue(payloads["judge_config.json"].endswith(b"\n"))
        self.assertEqual(pio.changed_files(draft), {})


class TestDefaultsClassificationGuard(unittest.TestCase):
    """Every field must have exactly one known source of truth for its default.

    This fails loudly the day a field is added to any config dataclass, which is
    the only thing keeping the tables in projectio honest.
    """

    _CLASSES = (
        Candidate, TestCase, JudgeConfig, GlobalCriteria,
        TestJudgeSettings, SimilarityJudgeSettings,
        ReasoningJudgeSettings, VerdictJudgeSettings,
    )

    def test_every_field_has_a_resolvable_default_or_is_excluded(self):
        for cls in self._CLASSES:
            for f in dataclasses.fields(cls):
                with self.subTest(cls=cls.__name__, field=f.name):
                    if f.name in pio._NEVER_EMIT.get(cls, ()):
                        self.assertNotIn(f.name, pio.editable_fields(cls))
                        continue
                    pio.effective_default(cls, f.name)  # must not raise

    def test_loader_defaults_only_cover_fields_that_lack_a_dataclass_default(self):
        for cls, defaults in pio._LOADER_DEFAULTS.items():
            by_name = {f.name: f for f in dataclasses.fields(cls)}
            for name in defaults:
                with self.subTest(cls=cls.__name__, field=name):
                    self.assertIn(name, by_name)
                    self.assertIs(by_name[name].default, dataclasses.MISSING)

    def test_dataclass_defaults_win_where_they_exist(self):
        self.assertEqual(pio.effective_default(Candidate, "temperature"), 0.7)
        self.assertEqual(pio.effective_default(TestCase, "judge_type"), "llm-judge")
        self.assertEqual(pio.effective_default(Candidate, "provider"), "google")

    def test_new_row_covers_exactly_the_always_emitted_keys(self):
        """A blank row must carry every key a loader would otherwise fumble.

        _NEW_ROW cannot be derived from effective_default: the loader fallback
        for `name` and `model` is None, which is right for "the key is absent"
        and wrong for "nobody has typed it yet".
        """
        for cls, row in pio._NEW_ROW.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(set(row), set(pio._ALWAYS_EMIT[cls]))
                self.assertEqual(pio.blank_row(cls), row)
                self.assertIsNot(pio.blank_row(cls), row)  # a fresh dict each time


class TestValidation(unittest.TestCase):
    def _draft(self, **kwargs):
        project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, project_dir, ignore_errors=True)
        draft = pio.load_project(project_dir)
        for key, value in kwargs.items():
            setattr(draft, key, value)
        return draft

    def _errors(self, draft):
        return pio.validate(draft)[0]

    def _warnings(self, draft):
        return pio.validate(draft)[1]

    def test_duplicate_candidate_name_is_an_error(self):
        draft = self._draft(candidates=[
            {"name": "A", "model": "m"}, {"name": "A", "model": "m"},
        ])
        self.assertTrue(any("used more than once" in e for e in self._errors(draft)))

    def test_duplicate_test_id_is_an_error(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "c"},
            {"id": "t", "prompt": "p", "criteria": "c"},
        ])
        self.assertTrue(any("'t' is used more than once" in e for e in self._errors(draft)))

    def test_missing_model_is_an_error(self):
        draft = self._draft(candidates=[{"name": "A"}])
        self.assertTrue(any("has no model" in e for e in self._errors(draft)))

    def test_missing_system_prompt_is_only_a_warning(self):
        draft = self._draft(candidates=[
            {"name": "A", "model": "m", "system_prompt_file": "nope.txt"},
        ])
        errors, warnings = pio.validate(draft)
        self.assertEqual([e for e in errors if "nope.txt" in e], [])
        self.assertTrue(any("nope.txt" in w for w in warnings))

    def test_missing_attachment_is_only_a_warning(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "c", "file": "test_files/nope.txt"},
        ])
        errors, warnings = pio.validate(draft)
        self.assertEqual([e for e in errors if "nope.txt" in e], [])
        self.assertTrue(any("nope.txt" in w for w in warnings))

    def test_every_missing_attachment_of_a_list_is_warned_about(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "c",
             "file": ["test_files/sample.txt", "test_files/one.txt", "test_files/two.txt"]},
        ])
        warnings = self._warnings(draft)
        self.assertTrue(any("one.txt" in w for w in warnings))
        self.assertTrue(any("two.txt" in w for w in warnings))
        self.assertEqual([w for w in warnings if "sample.txt" in w], [])

    def test_malformed_attachment_value_is_an_error(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "c", "file": 7},
        ])
        self.assertTrue(any("must be a string or a list" in e for e in self._errors(draft)))

    def test_attachment_used_by_a_list_is_not_reported_as_an_orphan(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "c",
             "file": ["test_files/sample.txt"]},
        ])
        self.assertEqual(
            [w for w in self._warnings(draft) if "not used by any test case" in w], []
        )

    def test_backslash_attachment_matches_the_file_on_disk(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "c", "file": "test_files\\sample.txt"},
        ])
        warnings = self._warnings(draft)
        self.assertEqual([w for w in warnings if "sample.txt" in w], [])

    def test_broken_assert_lambda_is_an_error(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "criteria": "s: (", "judge_type": "assert"},
        ])
        self.assertTrue(any("cannot be parsed" in e for e in self._errors(draft)))

    def test_valid_assert_lambda_passes(self):
        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "judge_type": "assert",
             "criteria": "s: (10, 'ok') if s else (1, 'ko')"},
        ])
        self.assertEqual([e for e in self._errors(draft) if "parsed" in e], [])

    def test_assert_criteria_is_never_executed(self):
        """test_judge.py eval()s these unsandboxed at run time. The editor must not."""
        import builtins
        from unittest.mock import patch

        draft = self._draft(tests=[
            {"id": "t", "prompt": "p", "judge_type": "assert",
             "criteria": "s: (10, 'ok') if s else (1, 'ko')"},
        ])
        with patch.object(builtins, "eval") as mock_eval:
            pio.validate(draft)
        mock_eval.assert_not_called()

    def test_stray_brace_in_evaluation_template_is_an_error(self):
        draft = self._draft()
        draft.judge.setdefault("test_judge", {})["evaluation_template"] = "literal { brace"
        self.assertTrue(any("evaluation_template" in e for e in self._errors(draft)))

    def test_unknown_placeholder_in_evaluation_template_is_an_error(self):
        draft = self._draft()
        draft.judge.setdefault("test_judge", {})["evaluation_template"] = "{criteria}{oops}"
        self.assertTrue(any("evaluation_template" in e for e in self._errors(draft)))

    def test_missing_placeholder_is_only_a_warning(self):
        draft = self._draft()
        draft.judge.setdefault("test_judge", {})["evaluation_template"] = "{criteria}{user_prompt}"
        errors, warnings = pio.validate(draft)
        self.assertEqual([e for e in errors if "evaluation_template" in e], [])
        self.assertTrue(any("candidate_response" in w for w in warnings))

    def test_verdict_template_braces_are_never_validated(self):
        """verdict_template is no longer str.format()ed — literal braces are legal."""
        draft = self._draft()
        draft.judge.setdefault("verdict_judge", {})["verdict_template"] = 'Return {"a": 1}'
        self.assertEqual([e for e in self._errors(draft) if "verdict_template" in e], [])

    def test_non_standard_thinking_value_is_a_warning_not_a_rewrite(self):
        draft = self._draft(candidates=[{"name": "A", "model": "m", "thinking": "high"}])
        self.assertTrue(any("non-standard thinking" in w for w in self._warnings(draft)))
        self.assertEqual(draft.candidates[0]["thinking"], "high")


class TestEffectiveLambda(unittest.TestCase):
    """Must mirror test_judge.py's normalisation exactly."""

    def test_bare_body_gets_the_lambda_prefix(self):
        self.assertEqual(pio.effective_lambda("s: 1"), "lambda s: 1")

    def test_a_full_lambda_is_left_alone(self):
        self.assertEqual(pio.effective_lambda("lambda s: 1"), "lambda s: 1")

    def test_startswith_is_lambda_not_lambda_space(self):
        """test_judge.py uses startswith("lambda"), so 'lambdas:' is NOT prefixed."""
        self.assertEqual(pio.effective_lambda("lambdas: 1"), "lambdas: 1")


class TestFilenameValidation(unittest.TestCase):
    def test_rejects_traversal_and_absolute_and_empty(self):
        for bad in ("../evil", "a/b.txt", "", ".", ".."):
            with self.subTest(name=bad):
                self.assertIsNotNone(pio.check_filename(bad))

    def test_accepts_a_plain_name(self):
        self.assertIsNone(pio.check_filename("prompt.txt"))


class TestExternalModification(unittest.TestCase):
    def test_detects_a_change_made_behind_the_editors_back(self):
        project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, project_dir, ignore_errors=True)
        draft = pio.load_project(project_dir)
        self.assertEqual(pio.externally_modified(draft), [])

        (Path(project_dir) / "candidates.json").write_text('[{"name": "elsewhere"}]',
                                                           encoding="utf-8")
        self.assertEqual(pio.externally_modified(draft), ["candidates.json"])


class TestPackaging(unittest.TestCase):
    """Every code directory must be a real package, or it ships in no wheel.

    pyproject.toml uses `packages.find` with include=["prompttestenv*"], and
    declares only `templates/*` as package-data. So a directory holding .py
    files but no __init__.py is invisible to the build: `prompttestenv gui` then
    works from an editable install (which points at the source tree) and fails
    everywhere else. That is exactly the state gui/ was in before it got one.

    Checked against the filesystem rather than via setuptools.find_packages, so
    that it actually runs: a venv created by Python 3.12+ has no setuptools, and
    a skipped guard guards nothing.
    """

    def test_every_code_directory_has_an_init(self):
        import prompttestenv

        package_root = Path(prompttestenv.__file__).parent
        missing = [
            directory.relative_to(package_root.parent).as_posix()
            for directory in package_root.rglob("*")
            if directory.is_dir()
            and directory.name != "__pycache__"
            and any(directory.glob("*.py"))
            and not (directory / "__init__.py").exists()
        ]
        self.assertEqual(missing, [], f"not importable, so not packaged: {missing}")

    def test_gui_is_one_of_them(self):
        import prompttestenv

        self.assertTrue(
            (Path(prompttestenv.__file__).parent / "gui" / "__init__.py").exists()
        )

    def test_importing_the_shared_helpers_does_not_touch_the_logger_backend(self):
        """gui/app.py and gui/editor.py flip it at import time; common must not."""
        import prompttestenv.logger as logger

        logger.set_backend("console")
        before = logger._emit
        import prompttestenv.gui.common  # noqa: F401
        import prompttestenv.projectio  # noqa: F401
        self.assertIs(logger._emit, before)


class TestLoadResilience(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def test_missing_files_load_as_empty(self):
        draft = pio.load_project(self.project_dir)
        self.assertEqual((draft.candidates, draft.tests, draft.judge, draft.criteria),
                         ([], [], {}, {}))

    def test_unparseable_file_loads_as_empty_rather_than_raising(self):
        (Path(self.project_dir) / "candidates.json").write_text("{{{", encoding="utf-8")
        self.assertEqual(pio.load_project(self.project_dir).candidates, [])


class SharedLayerTestCase(unittest.TestCase):
    """A throwaway copy of the smoke fixture, loaded as a draft."""

    def setUp(self):
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.draft = pio.load_project(self.project_dir)


class TestUpsert(SharedLayerTestCase):
    """Merge-or-append by the on-disk identity key."""

    def test_existing_entry_is_merged_not_replaced(self):
        original = dict(self.draft.candidates[0])
        merged = pio.upsert(
            self.draft.candidates, "name",
            {"name": original["name"], "temperature": 0.99}, Candidate,
        )
        self.assertEqual(len(merged), len(self.draft.candidates))
        self.assertEqual(merged[0]["temperature"], 0.99)
        rest = {k: v for k, v in merged[0].items() if k != "temperature"}
        self.assertEqual(rest, {k: v for k, v in original.items() if k != "temperature"})
        self.assertEqual(list(merged[0]), list(original))

    def test_unknown_key_appends_a_row_carrying_the_required_fields(self):
        merged = pio.upsert(
            self.draft.candidates, "name", {"name": "Fresh", "model": "m"}, Candidate,
        )
        self.assertEqual(len(merged), len(self.draft.candidates) + 1)
        self.assertEqual(merged[-1], {"name": "Fresh", "provider": "google", "model": "m"})

    def test_the_original_list_is_left_alone(self):
        before = [dict(c) for c in self.draft.candidates]
        pio.upsert(self.draft.candidates, "name", {"name": "Fresh", "model": "m"}, Candidate)
        self.assertEqual(self.draft.candidates, before)


class TestMergeJudge(SharedLayerTestCase):
    def test_scalars_and_blocks_merge_independently(self):
        merged = pio.merge_judge(
            self.draft.judge,
            {"repetitions": 7, "test_judge": {"temperature": 0.9}},
        )
        self.assertEqual(merged["repetitions"], 7)
        self.assertEqual(merged["test_judge"]["temperature"], 0.9)
        for key, value in self.draft.judge["test_judge"].items():
            if key != "temperature":
                self.assertEqual(merged["test_judge"][key], value)

    def test_naming_a_block_the_file_lacked_creates_it(self):
        """Unlike the editor, where a block appearing on tab open would be a bug.

        Here naming the block IS the edit, so creating it is what was asked for.
        """
        without = {k: v for k, v in self.draft.judge.items() if k != "reasoning_judge"}
        merged = pio.merge_judge(without, {"reasoning_judge": {"temperature": 0.1}})
        self.assertEqual(merged["reasoning_judge"], {"temperature": 0.1})

    def test_a_block_that_is_not_an_object_is_rejected(self):
        with self.assertRaises(ValueError):
            pio.merge_judge(self.draft.judge, {"test_judge": "nope"})


class TestSaveStatus(SharedLayerTestCase):
    def test_freshly_loaded_project_has_nothing_changed(self):
        status = pio.save_status(self.draft)
        self.assertEqual(status.changed, [])
        self.assertEqual(status.reformat_only, [])
        self.assertFalse(status.invalidates)

    def test_no_progress_log_never_invalidates(self):
        self.draft.candidates[0]["temperature"] = 0.123
        status = pio.save_status(self.draft)
        self.assertEqual(status.changed, ["candidates.json"])
        self.assertIsNone(status.stored_hash)
        self.assertFalse(status.invalidates)

    def test_a_stale_log_invalidates_and_the_hashes_differ(self):
        append_event(
            self.project_dir,
            {"type": "meta", "config_hash": calculate_config_hash(self.project_dir)},
        )
        untouched = pio.save_status(pio.load_project(self.project_dir))
        self.assertFalse(untouched.invalidates)

        draft = pio.load_project(self.project_dir)
        draft.candidates[0]["temperature"] = 0.123
        status = pio.save_status(draft)
        self.assertTrue(status.invalidates)
        self.assertNotEqual(status.stored_hash, status.would_be_hash)

    def test_pure_reformatting_is_reported_separately(self):
        """Same JSON, different bytes: still fatal for a byte-hashed file."""
        path = Path(self.project_dir) / "candidates.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        status = pio.save_status(pio.load_project(self.project_dir))
        self.assertEqual(status.changed, ["candidates.json"])
        self.assertEqual(status.reformat_only, ["candidates.json"])


class TestPendingAssets(SharedLayerTestCase):
    def test_a_pending_write_is_visible_before_it_reaches_disk(self):
        self.draft.pending_assets["system_prompts"]["new.txt"] = "hi"
        self.assertIn("new.txt", self.draft.system_prompt_names())
        self.assertFalse((self.draft.system_prompts_dir / "new.txt").exists())

    def test_a_pending_delete_hides_a_file_that_is_still_on_disk(self):
        existing = self.draft.system_prompt_names()[0]
        self.draft.pending_assets["system_prompts"][existing] = None
        self.assertNotIn(existing, self.draft.system_prompt_names())
        self.assertTrue((self.draft.system_prompts_dir / existing).exists())

    def test_validate_sees_an_attachment_that_is_only_pending(self):
        """The reason pending_assets exists at all.

        One patch that adds a file AND the test case using it must not report
        the file as missing.
        """
        self.draft.tests = [{"id": "t", "prompt": "p", "criteria": "c",
                             "file": "test_files/fresh.csv"}]
        self.assertTrue([w for w in pio.validate(self.draft)[1] if "fresh.csv" in w])

        self.draft.pending_assets["test_files"]["fresh.csv"] = "a,b"
        self.assertEqual([w for w in pio.validate(self.draft)[1] if "fresh.csv" in w], [])


class TestAssets(SharedLayerTestCase):
    def test_asset_users_finds_both_kinds(self):
        self.draft.candidates = [{"name": "A", "model": "m",
                                  "system_prompt_file": "p.txt"}]
        self.draft.tests = [{"id": "t", "prompt": "", "criteria": "",
                             "file": "test_files/s.txt"}]
        self.assertEqual(pio.asset_users(self.draft, "system_prompts", "p.txt"), ["A"])
        self.assertEqual(pio.asset_users(self.draft, "test_files", "s.txt"), ["t"])
        self.assertEqual(pio.asset_users(self.draft, "test_files", "other.txt"), [])

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            pio.asset_users(self.draft, "reports", "x.txt")

    def test_write_asset_refuses_a_path_and_a_wrong_suffix(self):
        for name in ("sub/x.txt", "..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pio.write_asset(self.draft, "system_prompts", name, "body")
        with self.assertRaises(ValueError):
            pio.write_asset(self.draft, "system_prompts", "x.md", "body")

    def test_write_asset_uses_lf_regardless_of_platform(self):
        pio.write_asset(self.draft, "system_prompts", "lf.txt", "one\ntwo\n")
        raw = (self.draft.system_prompts_dir / "lf.txt").read_bytes()
        self.assertNotIn(b"\r\n", raw)

    def test_write_asset_takes_bytes_for_an_attachment(self):
        pio.write_asset(self.draft, "test_files", "b.bin", b"\x00\x01")
        self.assertEqual((self.draft.test_files_dir / "b.bin").read_bytes(), b"\x00\x01")

    def test_delete_asset_refuses_while_something_still_refers_to_it(self):
        name = self.draft.system_prompt_names()[0]
        self.draft.candidates = [{"name": "A", "model": "m", "system_prompt_file": name}]
        with self.assertRaises(ValueError) as caught:
            pio.delete_asset(self.draft, "system_prompts", name)
        self.assertIn("still used by", str(caught.exception))
        self.assertTrue((self.draft.system_prompts_dir / name).exists())

    def test_delete_asset_removes_an_orphan(self):
        name = self.draft.system_prompt_names()[0]
        self.draft.candidates = []
        pio.delete_asset(self.draft, "system_prompts", name)
        self.assertFalse((self.draft.system_prompts_dir / name).exists())

    def test_delete_asset_reports_a_missing_file(self):
        with self.assertRaises(ValueError):
            pio.delete_asset(self.draft, "system_prompts", "never_existed.txt")


if __name__ == "__main__":
    unittest.main()
