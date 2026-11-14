"""AppTest coverage for the Streamlit project editor.

The cases here exist to pin the design decisions that are easy to regress:
uid-keyed rows (so deleting a row does not shift the others' widget values), the
generation counter (so Discard actually reloads), and the save-confirmation gate
(so a hash-invalidating write never happens silently).
"""
from __future__ import annotations

import json
import re
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import prompttestenv
from prompttestenv import projectio as pio
from prompttestenv.models import Candidate
from prompttestenv.progress import HASHED_FILENAMES, append_event, calculate_config_hash

from testutils import LoggerResetTestCase, make_temp_project

APP_PATH = str(Path(prompttestenv.__file__).parent / "gui" / "editor.py")

TAB_LABELS = ["Candidates", "Test cases", "Judge config", "Global criteria",
              "System prompts", "Test files", "Group clusters"]


class EditorTestCase(LoggerResetTestCase):
    """Opens a throwaway copy of the smoke fixture in a fresh editor app.

    Every catalog listing the editor asks for is stubbed: AppTest runs the
    script in this process, so the patch reaches it and no test probes a local
    server or calls a provider. By default the provider listing fails, which
    leaves the menus exactly as they are without a catalog.
    """

    providers: dict = {"providers": [], "error": "no catalog in tests"}
    models: dict = {}

    def setUp(self):
        super().setUp()
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.catalog_calls: list = []
        patcher = patch("prompttestenv.catalog.list_catalog", side_effect=self._fake_catalog)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _model(model_id, **capabilities) -> dict:
        """A model entry shaped like catalog.list_catalog's, unknown by default."""
        return {"id": model_id, "display_name": None, "reasoning": None, "tool_use": None,
                "file_types": None, "embeddings": None, "context_length": None,
                **capabilities}

    def _list(self, provider: str, *models) -> None:
        """Make `provider` usable and list `models` (ids or _model dicts) for it."""
        usable = [p for p in self.providers.get("providers", []) if p["name"] != provider]
        usable.append({"name": provider, "local": False, "requires_api_key": True,
                       "file_types": [], "supports_unload": False,
                       "supports_embeddings": True})
        self.providers = {"providers": usable, "error": None}
        self.models = {**self.models, provider: {
            "provider": provider, "error": None,
            "models": [m if isinstance(m, dict) else self._model(m) for m in models],
        }}

    def _fake_catalog(self, provider=None, **_kwargs):
        self.catalog_calls.append(provider)
        if provider is None:
            return self.providers
        return self.models.get(provider, {"provider": provider, "models": [], "error": None})

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
    def _key(at: AppTest, name: str) -> str:
        """A project-state widget key as the editor scopes it (see editor._skey)."""
        return f"ed{at.session_state['ed']['gen']}:{name}"

    def _open(self, at: AppTest, project_dir: str) -> None:
        """Load another project into an already running editor."""
        at.sidebar.text_input[0].set_value(project_dir)
        at.run()
        self._click(at, "📂 Open")
        at.run()

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
        """Row widgets by key suffix, text inputs and menus alike.

        Not by label: the judge blocks also have a "Model" input, and st.tabs
        renders every tab body on every run, so a label match would collide.
        """
        return [w for w in [*at.text_input, *at.selectbox] if w.proto.id and
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
        self.assertEqual([t.label for t in at.tabs][:7], TAB_LABELS)

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
        at.text_area(key=self._key(at, "tj:template")).set_value("{criteria} {user_prompt}")
        at.run()
        warned = self._shape(at)

        # A stray brace -> an error instead.
        at.text_area(key=self._key(at, "tj:template")).set_value("literal { brace")
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


class TestRowExpandersSurviveHeaderEdits(EditorTestCase):
    """Each candidate / test-case row expander carries a uid-scoped `key`.

    The headers interpolate the fields being edited (name/provider/model for a
    candidate, id/group/judge_type for a test case). Without a stable key the
    frontend loses the expander's identity when the label changes and snaps it
    shut on every blur. AppTest has no frontend, so this pins the server-side
    invariants the fix depends on: no exception, a stable set of expanders, and
    unique keys (a duplicate would raise `StreamlitDuplicateElementKey`). The
    collapse itself is frontend behaviour, confirmed against the JS bundle.
    """

    _CAND_LABEL = re.compile(r"^\d+\. ")

    def _candidate_expanders(self, at):
        return [e for e in at.expander if self._CAND_LABEL.match(e.label)]

    def _testcase_expanders(self, at):
        return [e for e in at.expander if " · [" in e.label]

    def _seed_four_candidates(self):
        (Path(self.project_dir) / "candidates.json").write_text(
            json.dumps([
                {"name": n, "provider": "google", "model": f"m{i}"}
                for i, n in enumerate(("Alpha", "Beta", "Gamma", "Delta"), start=1)
            ], indent=4),
            encoding="utf-8",
        )

    def test_editing_a_candidate_header_field_keeps_the_expander_set(self):
        self._seed_four_candidates()
        at = self._app()
        before = [e.label for e in self._candidate_expanders(at)]
        self.assertEqual(len(before), 4)

        # Row 4's header is `expanded=False` (>3 candidates); pre-fix this edit
        # rebuilt it collapsed.
        self._row_inputs(at, "cand", "name")[3].set_value("Renamed")
        at.run()

        after = self._candidate_expanders(at)
        self.assertEqual(list(at.exception), [])
        self.assertEqual(len(after), 4)
        self.assertIn("Renamed", after[3].label)
        self.assertEqual(before[:3], [e.label for e in after][:3])

    def test_editing_a_test_case_id_keeps_the_expander_set(self):
        at = self._app()
        before = self._testcase_expanders(at)
        self.assertEqual(len(before), 2)

        self._row_inputs(at, "test", "id")[0].set_value("renamed_case")
        at.run()

        after = self._testcase_expanders(at)
        self.assertEqual(list(at.exception), [])
        self.assertEqual(len(after), 2)
        self.assertTrue(after[0].label.startswith("renamed_case · ["))


class TestNoOpSave(EditorTestCase):
    def test_freshly_opened_project_has_nothing_to_save(self):
        at = self._app()
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))

    def test_rerunning_without_edits_leaves_nothing_to_save(self):
        """Every interaction is a rerun, and the second one is where the harvest
        first reads widget state back. The fixture has no reasoning_judge block,
        whose bool `thinking` default the widget spells "false": that used to add
        the block, and change a hashed file, on a form nobody had touched.
        """
        self.assertNotIn("reasoning_judge", json.loads(
            (Path(self.project_dir) / "judge_config.json").read_text("utf-8")))
        at = self._app()
        at.run()
        at.run()
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))
        self.assertNotIn("reasoning_judge", at.session_state["ed"]["draft"].judge)

    def test_a_pick_enables_save_on_the_same_run(self):
        """Picker values are harvested, not mirrored during render one run late."""
        at = self._app()
        pickers = [w for w in at.selectbox if w.proto.id.endswith(":system_prompt_file:pick")]
        current = pickers[0].value
        pickers[0].set_value("(none)" if current != "(none)" else pickers[0].options[-1])
        at.run()
        self.assertTrue(any("Unsaved changes: candidates.json" in w.value
                            for w in at.sidebar.warning))

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
        self._list("google", "gemini-2.5-flash", "gemini-3-flash-preview", "brand-new")
        at = self._app()
        self._click(at, "➕ Add candidate")
        at.run()

        self._row_inputs(at, "cand", "name")[-1].set_value("Newcomer")
        at.run()
        # Picked from the listing stubbed in setUp: AppTest can only select a
        # listed option, never type a new one in.
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


class TestRowDuplication(EditorTestCase):
    """The 📋 button clones a row in place with a unique name / id."""

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
        return [b for b in at.button if f"-{prefix}:" in b.proto.id][index]

    def test_duplicating_a_candidate_inserts_a_renamed_copy_after_it(self):
        self._seed_three_candidates()
        at = self._app()

        self._row_button(at, "cdup", 0).click()
        at.run()

        self.assertEqual(list(at.exception), [])
        self.assertEqual(self._candidate_names(at),
                         ["Alpha", "Alpha (copy)", "Beta", "Gamma"])
        # Every other field rides along with the clone.
        self.assertEqual([w.value for w in self._row_inputs(at, "cand", "model")],
                         ["m1", "m1", "m2", "m3"])
        self.assertFalse(any("more than once" in e.value for e in at.error))

    def test_duplicating_twice_bumps_the_suffix(self):
        self._seed_three_candidates()
        at = self._app()

        for _ in range(2):
            self._row_button(at, "cdup", 0).click()
            at.run()

        self.assertEqual(
            self._candidate_names(at),
            ["Alpha", "Alpha (copy 2)", "Alpha (copy)", "Beta", "Gamma"],
        )
        self.assertEqual(list(at.exception), [])

    def test_duplicating_a_test_case_inserts_a_renamed_copy(self):
        at = self._app()

        self._row_button(at, "tdup", 0).click()
        at.run()

        self.assertEqual(list(at.exception), [])
        self.assertEqual(
            [w.value for w in self._row_inputs(at, "test", "id")],
            ["customer_email", "customer_email (copy)", "file_analysis"],
        )

    def test_the_duplicate_survives_a_save(self):
        self._seed_three_candidates()
        at = self._app()

        self._row_button(at, "cdup", 0).click()
        at.run()
        self._click(at, "💾 Save")
        at.run()

        written = json.loads(
            (Path(self.project_dir) / "candidates.json").read_text("utf-8")
        )
        self.assertEqual([c["name"] for c in written],
                         ["Alpha", "Alpha (copy)", "Beta", "Gamma"])
        self.assertEqual(written[1]["model"], "m1")


class TestOptionalTemperature(EditorTestCase):
    """Temperature is a checkbox+input pair: unset means none is sent."""

    def setUp(self):
        super().setUp()
        (Path(self.project_dir) / "candidates.json").write_text(
            json.dumps([
                {"name": "Unset", "provider": "google", "model": "m1"},
                {"name": "Set", "provider": "google", "model": "m2", "temperature": 0.3},
            ], indent=4),
            encoding="utf-8",
        )

    @staticmethod
    def _toggles(at: AppTest) -> list:
        return [w for w in at.checkbox if w.proto.id and ":cand:" in w.proto.id
                and w.proto.id.endswith(":temperature:on")]

    def _saved(self) -> list[dict]:
        return json.loads((Path(self.project_dir) / "candidates.json").read_text("utf-8"))

    def test_toggles_reflect_whether_the_file_sets_one(self):
        at = self._app()
        self.assertEqual([w.value for w in self._toggles(at)], [False, True])

    def test_setting_one_writes_the_seed_value_and_touches_nothing_else(self):
        at = self._app()
        self._toggles(at)[0].check()
        at.run()
        self._click(at, "💾 Save")
        at.run()

        saved = self._saved()
        self.assertEqual(saved[0]["temperature"], 1.0)
        self.assertEqual(saved[1], {"name": "Set", "provider": "google",
                                    "model": "m2", "temperature": 0.3})

    def test_unsetting_one_writes_null_and_loads_as_none(self):
        at = self._app()
        self._toggles(at)[1].uncheck()
        at.run()
        self._click(at, "💾 Save")
        at.run()

        saved = self._saved()
        self.assertNotIn("temperature", saved[0])
        self.assertIsNone(saved[1]["temperature"])
        self.assertIsNone(Candidate.load_all(self.project_dir)[1].temperature)

    def test_judge_temperature_is_nullable_too(self):
        at = self._app()
        toggle = at.checkbox(key=self._key(at, "tj:temperature:on"))
        self.assertTrue(toggle.value)
        toggle.uncheck()
        at.run()
        self._click(at, "💾 Save")
        at.run()

        judge = json.loads((Path(self.project_dir) / "judge_config.json").read_text("utf-8"))
        self.assertIsNone(judge["test_judge"]["temperature"])


class TestJsonOutput(EditorTestCase):
    """json_output is a plain checkbox, written only when it carries information."""

    def setUp(self):
        super().setUp()
        (Path(self.project_dir) / "test_cases.json").write_text(
            json.dumps([
                {"id": "plain", "prompt": "p", "criteria": "c"},
                {"id": "json", "prompt": "p", "criteria": "c", "json_output": True},
            ], indent=4),
            encoding="utf-8",
        )

    @staticmethod
    def _toggles(at: AppTest) -> list:
        return [w for w in at.checkbox if w.proto.id and ":test:" in w.proto.id
                and w.proto.id.endswith(":json_output")]

    def _saved(self) -> list[dict]:
        return json.loads((Path(self.project_dir) / "test_cases.json").read_text("utf-8"))

    def test_toggles_reflect_the_file(self):
        at = self._app()
        self.assertEqual([w.value for w in self._toggles(at)], [False, True])

    def test_checking_one_writes_true_and_touches_nothing_else(self):
        at = self._app()
        self._toggles(at)[0].check()
        at.run()
        self._click(at, "💾 Save")
        at.run()

        saved = self._saved()
        self.assertEqual(saved[0], {"id": "plain", "prompt": "p", "criteria": "c",
                                    "json_output": True})
        self.assertEqual(saved[1]["json_output"], True)


class TestModelMenus(EditorTestCase):
    """Provider and model menus come from the live catalog, cached per session."""

    def test_listed_models_are_offered_with_what_they_can_do(self):
        self._list("google",
                   self._model("gemini-2.5-flash", reasoning=True, file_types=["image"],
                               context_length=1_048_576),
                   self._model("text-embedding-004", embeddings=True))
        at = self._app()
        menu = self._row_inputs(at, "cand", "model")[0]

        self.assertEqual(menu.value, "gemini-2.5-flash")  # the raw id, not the label
        self.assertIn("gemini-2.5-flash · reasoning, image, 1048k ctx", menu.options)
        # An embedding model is no candidate.
        self.assertFalse(any(o.startswith("text-embedding-004") for o in menu.options))

    def test_the_similarity_judge_is_offered_embedding_models_only(self):
        self._list("ollama",
                   self._model("bge-m3", embeddings=True),
                   self._model("qwen3:8b", embeddings=False),
                   self._model("mystery"))  # unknown: never filtered out
        at = self._app()
        options = at.selectbox(key=self._key(at, "sj:model")).options
        self.assertEqual(options, ["(none)", "bge-m3 · embeddings", "mystery"])

    def test_an_unlisted_model_is_kept_warned_about_and_never_rewritten(self):
        self._list("google", "some-other-model")
        before = self._disk()
        at = self._app()

        self.assertEqual(self._row_inputs(at, "cand", "model")[0].value, "gemini-2.5-flash")
        self.assertTrue(any("lists no model 'gemini-2.5-flash'" in w.value for w in at.warning))
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))
        self.assertEqual(list(at.error), [])
        self.assertEqual(self._disk(), before)

    def test_an_unavailable_provider_is_marked_and_not_asked_for_models(self):
        self._list("ollama", "bge-m3")
        at = self._app()

        options = self._row_inputs(at, "cand", "provider")[0].options
        self.assertIn("google (not available)", options)
        self.assertIn("ollama", options)
        self.assertNotIn("google", self.catalog_calls)
        self.assertTrue(any("not available right now" in c.value for c in at.caption))

    def test_listings_are_fetched_once_until_refreshed(self):
        self._list("google", "gemini-2.5-flash")
        at = self._app()
        fetched = len(self.catalog_calls)
        at.run()
        self.assertEqual(len(self.catalog_calls), fetched)

        self._click(at, "🔄 Refresh model lists")
        at.run()
        self.assertGreater(len(self.catalog_calls), fetched)
        self.assertEqual(list(at.exception), [])


class TestSaveConfirmation(EditorTestCase):
    def test_reasoning_only_edit_saves_without_confirmation(self):
        self._add_progress_log()
        hash_before = calculate_config_hash(self.project_dir)
        at = self._app()

        at.radio(key=self._key(at, "jc:scope")).set_value("all")
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

        at.number_input(key=self._key(at, "jc:repetitions")).set_value(42)
        at.run()
        self._click(at, "💾 Save")
        at.run()

        self.assertTrue(at.session_state["ed"]["confirm_save"])
        self.assertEqual(self._disk(), before)
        self.assertTrue(self._has_button(at, "Save anyway"))

    def test_confirming_writes(self):
        self._add_progress_log()
        at = self._app()
        at.number_input(key=self._key(at, "jc:repetitions")).set_value(42)
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
        at.number_input(key=self._key(at, "jc:repetitions")).set_value(42)
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
        at.number_input(key=self._key(at, "jc:repetitions")).set_value(7)
        at.run()
        self._click(at, "💾 Save")
        at.run()

        self.assertFalse(at.session_state["ed"]["confirm_save"])
        written = json.loads((Path(self.project_dir) / "judge_config.json").read_text("utf-8"))
        self.assertEqual(written["repetitions"], 7)


class TestCandidateRole(EditorTestCase):
    def _role_boxes(self, at):
        return [w for w in at.selectbox if w.proto.id and w.proto.id.endswith(":role")
                and ":cand:" in w.proto.id]

    def test_default_role_is_shown_and_never_written(self):
        at = self._app()
        self.assertTrue(all(box.value == "candidate" for box in self._role_boxes(at)))
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))

    def test_marking_a_baseline_writes_only_that_key(self):
        before = json.loads((Path(self.project_dir) / "candidates.json").read_text("utf-8"))
        at = self._app()
        self._role_boxes(at)[0].set_value("baseline")
        at.run()
        self._click(at, "💾 Save")
        at.run()
        after = json.loads((Path(self.project_dir) / "candidates.json").read_text("utf-8"))
        self.assertEqual(after[0], {**before[0], "role": "baseline"})
        self.assertEqual(after[1:], before[1:])


class TestGroupClustersTab(EditorTestCase):
    FILE = "group_clusters.json"

    def _path(self) -> Path:
        return Path(self.project_dir) / self.FILE

    def _save(self, at) -> None:
        self._click(at, "💾 Save")
        at.run()

    def _row_key(self, at, index: int, field: str) -> str:
        uid = at.session_state["ed"]["clusters"][index]["uid"]
        return self._key(at, f"cl:{uid}:{field}")

    def test_opening_never_creates_the_file(self):
        at = self._app()
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))
        self.assertFalse(self._path().exists())

    def test_trying_an_option_and_taking_it_back_leaves_nothing_to_save(self):
        at = self._app()
        box = at.selectbox(key=self._key(at, "gcl:defaults:x_axis"))
        box.set_value("think_tokens")
        at.run()
        self.assertTrue(any(self.FILE in w.value for w in at.sidebar.warning))
        at.selectbox(key=self._key(at, "gcl:defaults:x_axis")).set_value("(inherit)")
        at.run()
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))
        self.assertEqual(at.session_state["ed"]["draft"].clusters, {})

    def test_a_default_saves_without_confirmation_even_with_a_run(self):
        self._add_progress_log()
        hash_before = calculate_config_hash(self.project_dir)
        at = self._app()
        at.selectbox(key=self._key(at, "gcl:defaults:x_axis")).set_value("think_tokens")
        at.run()
        self._save(at)
        self.assertFalse(at.session_state["ed"]["confirm_save"])
        self.assertEqual(json.loads(self._path().read_text("utf-8")),
                         {"defaults": {"x_axis": "think_tokens"}})
        self.assertEqual(calculate_config_hash(self.project_dir), hash_before)

    def test_adding_a_cluster(self):
        at = self._app()
        self._click(at, "➕ Add cluster")
        at.run()
        at.text_input(key=self._row_key(at, 0, "name")).set_value("Everything")
        at.multiselect(key=self._row_key(at, 0, "groups")).set_value(["Default group"])
        at.selectbox(key=self._row_key(at, 0, "opt:front")).set_value("lower")
        at.run()
        self._save(at)
        self.assertEqual(json.loads(self._path().read_text("utf-8")), {"clusters": [
            {"name": "Everything", "groups": ["Default group"], "front": "lower"},
        ]})

    def test_an_unnamed_cluster_blocks_saving(self):
        at = self._app()
        self._click(at, "➕ Add cluster")
        at.run()
        self.assertTrue(any("has no name" in e.value for e in at.error))

    def test_an_existing_file_round_trips_and_a_cluster_can_be_removed(self):
        original = json.dumps({
            "defaults": {"threshold": 6},
            "clusters": [
                {"name": "A", "groups": ["Default group"]},
                {"name": "B", "groups": ["Default group"], "x_axis": "time"},
            ],
        }, indent=4).replace("\n", "\r\n") + "\r\n"
        self._path().write_text(original, encoding="utf-8", newline="")
        at = self._app()
        # The int threshold and the CRLF survive: nothing is pending.
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))
        self._click(at, "🗑 Remove cluster")
        at.run()
        self._save(at)
        raw = self._path().read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertEqual(json.loads(raw), {
            "defaults": {"threshold": 6},
            "clusters": [{"name": "B", "groups": ["Default group"], "x_axis": "time"}],
        })

    def test_discard_reseeds_the_tab_from_disk(self):
        at = self._app()
        self._click(at, "➕ Add cluster")
        at.run()
        at.text_input(key=self._row_key(at, 0, "name")).set_value("Temp")
        at.selectbox(key=self._key(at, "gcl:defaults:front")).set_value("lower")
        at.run()
        self._click(at, "↩️ Discard")
        at.run()
        self.assertEqual(at.session_state["ed"]["clusters"], [])
        self.assertEqual(at.session_state["ed"]["draft"].clusters, {})
        self.assertEqual(at.selectbox(key=self._key(at, "gcl:defaults:front")).value, "(inherit)")

    def test_excluding_a_candidate(self):
        name = json.loads((Path(self.project_dir) / "candidates.json").read_text("utf-8"))[0]["name"]
        at = self._app()
        at.multiselect(key=self._key(at, "gcl:exclude")).set_value([name])
        at.checkbox(key=self._key(at, "gcl:include")).uncheck()
        at.run()
        self._save(at)
        self.assertEqual(json.loads(self._path().read_text("utf-8")),
                         {"include_in_verdict": False, "exclude_candidates": [name]})


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
        at.text_area(key=self._key(at, "tj:template")).set_value("literal { brace")
        at.run()
        self.assertTrue(any("evaluation_template" in e.value for e in at.error))
        save = next(b for b in at.button if b.label == "💾 Save")
        self.assertTrue(save.proto.disabled)

    def test_missing_placeholder_warns_but_still_allows_saving(self):
        at = self._app()
        at.text_area(key=self._key(at, "tj:template")).set_value("{criteria} {user_prompt}")
        at.run()
        self.assertTrue(any("candidate_response" in w.value for w in at.warning))
        save = next(b for b in at.button if b.label == "💾 Save")
        self.assertFalse(save.proto.disabled)

    def test_literal_braces_in_verdict_template_are_fine(self):
        at = self._app()
        at.text_area(key=self._key(at, "vj:template")).set_value('Answer with {"verdict": "..."}')
        at.run()
        self.assertEqual([e.value for e in at.error if "verdict_template" in e.value], [])


class TestDiscard(EditorTestCase):
    def test_discard_restores_the_widgets_from_disk(self):
        at = self._app()
        original = at.number_input(key=self._key(at, "jc:repetitions")).value

        at.number_input(key=self._key(at, "jc:repetitions")).set_value(99)
        at.run()
        self.assertTrue(any("Unsaved changes" in w.value for w in at.warning))

        self._click(at, "↩️ Discard")
        at.run()

        # Reads from `value=` only because the generation counter changed the
        # widget key; a stale key would still show 99.
        self.assertEqual(at.number_input(key=self._key(at, "jc:repetitions")).value, original)
        self.assertTrue(any("No unsaved changes" in s.value for s in at.success))


class TestLoadAnotherProject(EditorTestCase):
    """Loading a project must not carry the previous one's widget state into it."""

    def setUp(self):
        super().setUp()
        self.other_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.other_dir, ignore_errors=True)

    def test_judge_values_come_from_the_new_project(self):
        judge_path = Path(self.other_dir) / "judge_config.json"
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        judge["repetitions"] = 7
        judge_path.write_text(json.dumps(judge, indent=4), encoding="utf-8")
        at = self._app()
        at.number_input(key=self._key(at, "jc:repetitions")).set_value(99)
        at.run()

        self._open(at, self.other_dir)

        on_disk = json.loads(
            (Path(self.other_dir) / "judge_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(at.number_input(key=self._key(at, "jc:repetitions")).value,
                         on_disk["repetitions"])
        self.assertEqual(at.session_state["ed"]["draft"].judge["repetitions"],
                         on_disk["repetitions"])

    def test_system_prompt_body_comes_from_the_new_project(self):
        name = "pirate_prompt.txt"
        (Path(self.other_dir) / "system_prompts" / name).write_text(
            "the other project's prompt", encoding="utf-8",
        )
        at = self._app()
        at.selectbox(key=self._key(at, "sp:pick")).set_value(name)
        at.run()
        at.text_area(key=self._key(at, f"sp:body:{name}")).set_value("edited, unsaved")
        at.run()

        self._open(at, self.other_dir)
        at.selectbox(key=self._key(at, "sp:pick")).set_value(name)
        at.run()

        self.assertEqual(at.text_area(key=self._key(at, f"sp:body:{name}")).value,
                         "the other project's prompt")


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

        at.text_input(key=self._key(at, "test_filter")).set_value("editme")
        at.run()
        visible = [w for w in at.text_input if w.label == "ID"]
        self.assertEqual([w.value for w in visible], ["editme"])

        visible[0].set_value("edited")
        at.run()
        at.text_input(key=self._key(at, "test_filter")).set_value("")
        at.run()

        ids = [w.value for w in at.text_input if w.label == "ID"]
        self.assertEqual(ids, ["keepme", "edited"])


class TestAttachments(EditorTestCase):
    """The attachment picker is multi-select, and writes 0/1/N shapes."""

    def _picker(self, at: AppTest):
        pickers = [w for w in at.multiselect if w.proto.id.endswith(":file:pick")]
        self.assertTrue(pickers, "no attachment multiselect rendered")
        return pickers

    def test_existing_single_attachment_shows_as_selected(self):
        at = self._app()
        selected = [p.value for p in self._picker(at)]
        self.assertIn(["test_files/sample.txt"], selected)

    def test_selecting_a_second_file_writes_a_list(self):
        (Path(self.project_dir) / "test_files" / "extra.md").write_text(
            "# extra\n", encoding="utf-8",
        )
        at = self._app()
        picker = [p for p in self._picker(at) if p.value][0]
        picker.set_value(["test_files/sample.txt", "test_files/extra.md"])
        at.run()

        self._click(at, "💾 Save")
        at.run()
        written = json.loads(
            (Path(self.project_dir) / "test_cases.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            written[1]["file"], ["test_files/sample.txt", "test_files/extra.md"],
        )

    def test_clearing_the_selection_writes_no_attachment(self):
        at = self._app()
        picker = [p for p in self._picker(at) if p.value][0]
        picker.set_value([])
        at.run()

        self._click(at, "💾 Save")
        at.run()
        written = json.loads(
            (Path(self.project_dir) / "test_cases.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(written[1]["file"])


class TestUntouchedArtifacts(EditorTestCase):
    def test_saving_never_touches_the_progress_log_or_reports(self):
        self._add_progress_log()
        report_dir = Path(self.project_dir) / "Report"
        report_dir.mkdir()
        (report_dir / "old.html").write_text("report", encoding="utf-8")
        log_before = (Path(self.project_dir) / "progress.jsonl").read_bytes()

        at = self._app()
        at.number_input(key=self._key(at, "jc:repetitions")).set_value(3)
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
        at.number_input(key=self._key(at, "jc:repetitions")).set_value(4)
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
