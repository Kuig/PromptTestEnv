"""Smoke tests for the Streamlit GUI (``prompttestenv/gui/app.py``).

Uses ``streamlit.testing.v1.AppTest`` — no browser, no real LLM calls. The base
class resets the logger backend because loading ``app.py`` calls
``logger.set_backend("streamlit")`` as an import-time side effect.
"""
from __future__ import annotations

import shutil
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import prompttestenv
from prompttestenv.gui.common import report_path
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


class TestReportPath(LoggerResetTestCase):
    """The success banner offers the report file instead of repeating its path."""

    def setUp(self):
        super().setUp()
        self.project_dir = make_temp_project()
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)

    def _make_report(self, name="20260101_000000_2C_3T.html"):
        report_dir = Path(self.project_dir) / "Report"
        report_dir.mkdir(exist_ok=True)
        path = report_dir / name
        path.write_text("<html></html>", encoding="utf-8")
        return path

    def test_parses_both_report_result_strings(self):
        path = self._make_report()
        self.assertEqual(
            report_path(f"Full process complete. HTML Report: {path}"), path)

        md = self._make_report("report.md")
        self.assertEqual(report_path(f"Markdown generated: {md}"), md)

        js = self._make_report("report.json")
        self.assertEqual(report_path(f"JSON report: {js}"), js)

    def test_returns_none_for_results_that_name_no_file(self):
        for result in (
            "Winner: Baseline (8.4)",
            "No progress found in Projects/Foo",
            "Partial progress: 3 generation(s) and 0 evaluation(s) found.",
            "ERROR: No results produced.",
        ):
            with self.subTest(result=result):
                self.assertIsNone(report_path(result))

    def test_returns_none_when_the_file_is_gone(self):
        path = self._make_report()
        result = f"Full process complete. HTML Report: {path}"
        path.unlink()
        self.assertIsNone(report_path(result))

    def test_success_shows_one_banner_and_an_open_button(self):
        path = self._make_report()
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.session_state["last_result"] = (
            "success", f"Full process complete. HTML Report: {path}")
        at.run()

        self.assertEqual(len(at.success), 1)
        # The path used to be repeated verbatim in an st.code block below it.
        self.assertEqual(list(at.code), [])
        self.assertTrue(any(b.label == f"🔗 Open {path.name}" for b in at.button))

    def test_a_result_without_a_file_gets_no_open_button(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.session_state["last_result"] = ("success", "Winner: Baseline (8.4)")
        at.run()

        self.assertEqual(len(at.success), 1)
        self.assertFalse(any(b.label.startswith("🔗 Open") for b in at.button))

    def test_clicking_open_launches_the_report_in_a_browser(self):
        path = self._make_report()
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.session_state["last_result"] = (
            "success", f"Full process complete. HTML Report: {path}")
        at.run()

        with patch("webbrowser.open") as mock_open:
            next(b for b in at.button if b.label.startswith("🔗 Open")).click()
            at.run()

        mock_open.assert_called_once()
        self.assertEqual(mock_open.call_args.args[0], path.resolve().as_uri())
