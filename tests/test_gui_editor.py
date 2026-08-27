"""AppTest coverage for the Streamlit project editor.

The cases here exist to pin the design decisions that are easy to regress:
uid-keyed rows (so deleting a row does not shift the others' widget values), the
generation counter (so Discard actually reloads), and the save-confirmation gate
(so a hash-invalidating write never happens silently).
"""
from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import prompttestenv
from prompttestenv.progress import HASHED_FILENAMES, append_event, calculate_config_hash

from testutils import LoggerResetTestCase, make_temp_project

APP_PATH = str(Path(prompttestenv.__file__).parent / "gui" / "editor.py")

TAB_LABELS = ["Candidates", "Test cases", "Judge config", "Global criteria",
              "System prompts", "Test files"]


class EditorTestCase(LoggerResetTestCase):
    """Opens a throwaway copy of the smoke fixture in a fresh editor app."""

    def setUp(self):
        super().setUp()
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def _app(self, *, open_project=True) -> AppTest:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        if open_project:
            at.text_input[0].set_value(self.project_dir)
            at.run()
            self._click(at, "📂 Open")
            at.run()
        return at

    @staticmethod
    def _click(at: AppTest, label: str) -> None:
        for button in at.button:
            if button.label == label:
                button.click()
                return
        raise AssertionError(f"no button labelled {label!r}; have "
                             f"{[b.label for b in at.button]}")

    @staticmethod
    def _has_button(at: AppTest, label: str) -> bool:
        return any(b.label == label for b in at.button)

    def _disk(self) -> dict:
        return {
            name: (Path(self.project_dir) / name).read_bytes()
            for name in HASHED_FILENAMES
        }

    def _add_progress_log(self) -> None:
        """Give the project a progress.jsonl whose hash matches its config."""
        append_event(self.project_dir, {
            "type": "meta", "config_hash": calculate_config_hash(self.project_dir),
        })

    @staticmethod
    def _row_inputs(at: AppTest, kind: str, field: str) -> list:
        """Row widgets by key suffix.

        Not by label: the judge blocks also have a "Model" input, and st.tabs
        renders every tab body on every run, so a label match would collide.
        """
        return [w for w in at.text_input if w.proto.id and
                f":{kind}:" in w.proto.id and w.proto.id.endswith(f":{field}")]

    def _candidate_names(self, at: AppTest) -> list[str]:
        return [w.value for w in self._row_inputs(at, "cand", "name")]


class TestEditorLoads(EditorTestCase):
    def test_loads_without_exception_and_prompts_for_a_project(self):
        at = self._app(open_project=False)
        self.assertEqual(list(at.exception), [])
        self.assertTrue(self._has_button(at, "📂 Open"))
        self.assertTrue(any("Open an existing project" in i.value for i in at.info))

    def test_open_populates_the_forms(self):
        at = self._app()
        self.assertEqual(list(at.exception), [])
        self.assertEqual([t.label for t in at.tabs][:6], TAB_LABELS)

        on_disk = json.loads((Path(self.project_dir) / "candidates.json").read_text("utf-8"))
        self.assertEqual(self._candidate_names(at), [c["name"] for c in on_disk])

    def test_open_requires_a_path(self):
        at = self._app(open_project=False)
        self._click(at, "📂 Open")
        at.run()
        self.assertTrue(any("required" in e.value for e in at.error))

    def test_does_not_run_benchmarks(self):
        at = self._app()
        labels = {b.label for b in at.button}
        self.assertNotIn("▶️ Run Benchmark", labels)


class TestTabsStayPut(EditorTestCase):
    """The element tree above st.tabs must not change shape as you type.

    Streamlit identifies the tabs by their position, so an st.error appearing or
    disappearing above them makes it rebuild the widget and reset the selection
    to the first tab. Editing the evaluation template used to do exactly that —
    its validation flips between error, warning and neither — bouncing the user
    back to Candidates on every blur.
    """

    @staticmethod
    def _shape(at: AppTest) -> list[str]:
        return [
            getattr(child, "type", type(child).__name__)
            for child in at.main.children.values()
        ]

    def test_top_level_shape_is_identical_across_validation_states(self):
        at = self._app()
        clean = self._shape(at)
        self.assertIn("tab_container", clean)

        # Formats fine, but omits a placeholder -> a warning is added.
        at.text_area(key="tj:template").set_value("{criteria} {user_prompt}")
        at.run()
        warned = self._shape(at)

        # A stray brace -> an error instead.
        at.text_area(key="tj:template").set_value("literal { brace")
        at.run()
        errored = self._shape(at)

        self.assertTrue(at.error, "expected the broken template to report an error")
        self.assertEqual(clean, warned)
        self.assertEqual(clean, errored)
        self.assertEqual(
            clean.index("tab_container"), errored.index("tab_container"),
            "the tabs moved, so Streamlit will reset the selected tab",
        )

    def test_the_slots_above_the_tabs_are_always_present(self):
        at = self._app()
        shape = self._shape(at)
        # title, flash slot, confirm slot, issues slot, tabs.
        self.assertEqual(shape.index("tab_container"), 4)
        self.assertEqual(shape[1:4], ["flex_container"] * 3)


class TestNoOpSave(EditorTestCase):
    def test_freshly_opened_project_has_nothing_to_save(self):
        at = self._app()
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))

    def test_opening_and_saving_changes_no_bytes(self):
        before = self._disk()
        at = self._app()
        # Save is disabled with nothing to save, so simply prove the round trip
        # produced no pending write at all.
        self.assertEqual(self._disk(), before)
        self.assertEqual(list(at.exception), [])


class TestRowIdentity(EditorTestCase):
    """Widget keys are uid-scoped, so rows keep their values across mutations."""

    def _seed_three_candidates(self):
        (Path(self.project_dir) / "candidates.json").write_text(
            json.dumps([
                {"name": "Alpha", "provider": "google", "model": "m1"},
                {"name": "Beta", "provider": "google", "model": "m2"},
                {"name": "Gamma", "provider": "google", "model": "m3"},
            ], indent=4),
            encoding="utf-8",
        )

    @staticmethod
    def _row_button(at: AppTest, prefix: str, index: int):
        """The `prefix` button belonging to the row at `index`."""
        return [b for b in at.button if f"-{prefix}:" in b.proto.id][index]

    def test_deleting_a_row_does_not_shift_the_others_values(self):
        self._seed_three_candidates()
        at = self._app()
        self.assertEqual(self._candidate_names(at), ["Alpha", "Beta", "Gamma"])

        self._row_button(at, "crm", 0).click()
        at.run()

        # The survivors keep THEIR OWN values. With index-keyed widgets they
        # would read Alpha/Beta here — the staleness the uids exist to prevent.
        self.assertEqual(self._candidate_names(at), ["Beta", "Gamma"])

    def test_reordering_moves_the_values_with_the_row(self):
        self._seed_three_candidates()
        at = self._app()
        self._row_button(at, "cdn", 0).click()
        at.run()
        self.assertEqual(self._candidate_names(at), ["Beta", "Alpha", "Gamma"])

    def test_added_candidate_writes_its_required_keys_even_at_defaults(self):
        at = self._app()
        self._click(at, "➕ Add candidate")
        at.run()

        self._row_inputs(at, "cand", "name")[-1].set_value("Newcomer")
        at.run()
        self._row_inputs(at, "cand", "model")[-1].set_value("brand-new")
        at.run()

        self._click(at, "💾 Save")
        at.run()

        written = json.loads((Path(self.project_dir) / "candidates.json").read_text("utf-8"))
        self.assertEqual(written[-1]["name"], "Newcomer")
        # provider is at its default but must still be emitted, or load_all
        # would hand the run a None model / missing provider.
        self.assertEqual(written[-1]["provider"], "google")
        self.assertEqual(written[-1]["model"], "brand-new")


class TestSaveConfirmation(EditorTestCase):
    def test_reasoning_only_edit_saves_without_confirmation(self):
        self._add_progress_log()
        hash_before = calculate_config_hash(self.project_dir)
        at = self._app()

        at.radio(key="jc:scope").set_value("all")
        at.run()
        self._click(at, "💾 Save")
        at.run()

        self.assertFalse(at.session_state["ed"]["confirm_save"])
        self.assertTrue(any("Saved" in s.value for s in at.success))
        # The whole point: the scope is stripped before hashing.
        self.assertEqual(calculate_config_hash(self.project_dir), hash_before)
        self.assertEqual(
            json.loads((Path(self.project_dir) / "judge_config.json").read_text("utf-8"))
            ["reasoning_analysis"],
            "all",
        )

    def test_other_edits_ask_first_and_write_nothing_yet(self):
        self._add_progress_log()
        before = self._disk()
        at = self._app()

        at.number_input(key="jc:repetitions").set_value(42)
        at.run()
        self._click(at, "💾 Save")
        at.run()

        self.assertTrue(at.session_state["ed"]["confirm_save"])
        self.assertEqual(self._disk(), before)
        self.assertTrue(self._has_button(at, "Save anyway"))

    def test_confirming_writes(self):
        self._add_progress_log()
        at = self._app()
        at.number_input(key="jc:repetitions").set_value(42)
        at.run()
        self._click(at, "💾 Save")
        at.run()
        self._click(at, "Save anyway")
        at.run()

        written = json.loads((Path(self.project_dir) / "judge_config.json").read_text("utf-8"))
        self.assertEqual(written["repetitions"], 42)

    def test_cancelling_writes_nothing_but_keeps_the_edit(self):
        self._add_progress_log()
        before = self._disk()
        at = self._app()
        at.number_input(key="jc:repetitions").set_value(42)
        at.run()
        self._click(at, "💾 Save")
        at.run()
        self._click(at, "Cancel")
        at.run()

        self.assertFalse(at.session_state["ed"]["confirm_save"])
        self.assertEqual(self._disk(), before)
        self.assertEqual(at.session_state["ed"]["draft"].judge["repetitions"], 42)

    def test_no_progress_log_means_no_confirmation(self):
        at = self._app()
        at.number_input(key="jc:repetitions").set_value(7)
        at.run()
        self._click(at, "💾 Save")
        at.run()

        self.assertFalse(at.session_state["ed"]["confirm_save"])
        written = json.loads((Path(self.project_dir) / "judge_config.json").read_text("utf-8"))
        self.assertEqual(written["repetitions"], 7)


class TestValidationGate(EditorTestCase):
    def test_duplicate_candidate_name_blocks_saving(self):
        (Path(self.project_dir) / "candidates.json").write_text(
            json.dumps([
                {"name": "Same", "provider": "google", "model": "m1"},
                {"name": "Same", "provider": "google", "model": "m2"},
            ], indent=4),
            encoding="utf-8",
        )
        before = self._disk()
        at = self._app()

        self.assertTrue(any("used more than once" in e.value for e in at.error))
        save = next(b for b in at.button if b.label == "💾 Save")
        self.assertTrue(save.proto.disabled)
        self.assertEqual(self._disk(), before)

    def test_broken_evaluation_template_blocks_saving(self):
        at = self._app()
        at.text_area(key="tj:template").set_value("literal { brace")
        at.run()
        self.assertTrue(any("evaluation_template" in e.value for e in at.error))
        save = next(b for b in at.button if b.label == "💾 Save")
        self.assertTrue(save.proto.disabled)

    def test_missing_placeholder_warns_but_still_allows_saving(self):
        at = self._app()
        at.text_area(key="tj:template").set_value("{criteria} {user_prompt}")
        at.run()
        self.assertTrue(any("candidate_response" in w.value for w in at.warning))
        save = next(b for b in at.button if b.label == "💾 Save")
        self.assertFalse(save.proto.disabled)

    def test_literal_braces_in_verdict_template_are_fine(self):
        at = self._app()
        at.text_area(key="vj:template").set_value('Answer with {"verdict": "..."}')
        at.run()
        self.assertEqual([e.value for e in at.error if "verdict_template" in e.value], [])


class TestDiscard(EditorTestCase):
    def test_discard_restores_the_widgets_from_disk(self):
        at = self._app()
        original = at.number_input(key="jc:repetitions").value

        at.number_input(key="jc:repetitions").set_value(99)
        at.run()
        self.assertTrue(any("Unsaved changes" in w.value for w in at.warning))

        self._click(at, "↩️ Discard")
        at.run()

        # Reads from `value=` only because the generation counter changed the
        # widget key; a stale key would still show 99.
        self.assertEqual(at.number_input(key="jc:repetitions").value, original)
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))


class TestFiltering(EditorTestCase):
    def test_editing_a_visible_row_leaves_a_filtered_out_one_intact(self):
        (Path(self.project_dir) / "test_cases.json").write_text(
            json.dumps([
                {"id": "keepme", "prompt": "p1", "criteria": "c1"},
                {"id": "editme", "prompt": "p2", "criteria": "c2"},
            ], indent=4),
            encoding="utf-8",
        )
        at = self._app()

        at.text_input(key="test_filter").set_value("editme")
        at.run()
        visible = [w for w in at.text_input if w.label == "ID"]
        self.assertEqual([w.value for w in visible], ["editme"])

        visible[0].set_value("edited")
        at.run()
        at.text_input(key="test_filter").set_value("")
        at.run()

        ids = [w.value for w in at.text_input if w.label == "ID"]
        self.assertEqual(ids, ["keepme", "edited"])


class TestUntouchedArtifacts(EditorTestCase):
    def test_saving_never_touches_the_progress_log_or_reports(self):
        self._add_progress_log()
        report_dir = Path(self.project_dir) / "Report"
        report_dir.mkdir()
        (report_dir / "old.html").write_text("report", encoding="utf-8")
        log_before = (Path(self.project_dir) / "progress.jsonl").read_bytes()

        at = self._app()
        at.number_input(key="jc:repetitions").set_value(3)
        at.run()
        self._click(at, "💾 Save")
        at.run()
        self._click(at, "Save anyway")
        at.run()

        self.assertEqual((Path(self.project_dir) / "progress.jsonl").read_bytes(), log_before)
        self.assertEqual((report_dir / "old.html").read_text("utf-8"), "report")

    def test_only_the_changed_config_file_is_rewritten(self):
        before = self._disk()
        at = self._app()
        at.number_input(key="jc:repetitions").set_value(4)
        at.run()
        self._click(at, "💾 Save")
        at.run()

        after = self._disk()
        self.assertNotEqual(after["judge_config.json"], before["judge_config.json"])
        for name in ("candidates.json", "test_cases.json", "global_criteria.json"):
            self.assertEqual(after[name], before[name], msg=name)


class TestSeedNewProject(EditorTestCase):
    def test_new_scaffolds_a_project_without_a_secrets_file(self):
        target = Path(self.project_dir) / "Fresh"
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.text_input[0].set_value(str(target))
        at.run()
        self._click(at, "✨ New")
        at.run()

        self.assertEqual(list(at.exception), [])
        for name in HASHED_FILENAMES:
            self.assertTrue((target / name).exists(), msg=name)
        self.assertFalse((target / "secrets.json").exists())
        self.assertTrue(any("Created" in s.value for s in at.success))


if __name__ == "__main__":
    unittest.main()
