"""Smoke tests for the Streamlit GUI (``prompttestenv/gui/app.py``).

Uses ``streamlit.testing.v1.AppTest`` — no browser, no real LLM calls. The base
class resets the logger backend because loading ``app.py`` calls
``logger.set_backend("streamlit")`` as an import-time side effect.
"""
from __future__ import annotations

import shutil

from streamlit.testing.v1 import AppTest

import prompttestenv
from testutils import LoggerResetTestCase, make_temp_project

from pathlib import Path

APP_PATH = str(Path(prompttestenv.__file__).parent / "gui" / "app.py")

_EXPECTED_BUTTONS = {
    "📁 Initialize Project",
    "▶️ Run Benchmark",
    "🧠 Analyze Reasoning",
    "📊 Render from Progress",
    "📁 Browse...",
}


class TestGuiApp(LoggerResetTestCase):
    def _fresh_app(self) -> AppTest:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        return at

    def test_app_loads_without_exception(self):
        at = self._fresh_app()
        self.assertEqual(list(at.exception), [])
        self.assertEqual({b.label for b in at.button}, _EXPECTED_BUTTONS)

    def test_empty_dir_shows_error_not_success(self):
        at = self._fresh_app()
        for b in at.button:
            if b.label == "▶️ Run Benchmark":
                b.click()
        at.run()
        self.assertTrue(
            any("Project directory is required." in e.value for e in at.error)
        )
        self.assertEqual(list(at.success), [])

    def test_render_no_progress_is_warning_not_success(self):
        project_dir = make_temp_project()  # copy of smoke_project, no progress.jsonl
        self.addCleanup(shutil.rmtree, project_dir, ignore_errors=True)

        at = self._fresh_app()
        at.text_input[0].set_value(project_dir)
        at.run()
        for b in at.button:
            if b.label == "📊 Render from Progress":
                b.click()
        at.run()

        self.assertEqual(list(at.success), [])
        self.assertTrue(
            any("No progress found" in w.value for w in at.warning),
            msg=f"warnings were: {[w.value for w in at.warning]}",
        )

    def test_result_persists_across_rerun(self):
        project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, project_dir, ignore_errors=True)

        at = self._fresh_app()
        at.text_input[0].set_value(project_dir)
        at.run()
        for b in at.button:
            if b.label == "📊 Render from Progress":
                b.click()
        at.run()
        # A rerun with no further interaction must still show the last outcome.
        at.run()

        self.assertEqual(list(at.success), [])
        self.assertTrue(any("No progress found" in w.value for w in at.warning))
