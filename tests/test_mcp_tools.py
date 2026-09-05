from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from prompttestenv.mcp_tools import register_tools
from prompttestenv.projectedit import EditResult


class FakeMCP:
    """Minimal stand-in for FastMCP: captures decorated tool functions by name."""

    def __init__(self):
        self.tools: dict[str, callable] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


class TestRegisterTools(unittest.TestCase):
    def setUp(self):
        self.mcp = FakeMCP()
        register_tools(self.mcp)

    def test_registers_the_six_expected_tools(self):
        self.assertEqual(
            set(self.mcp.tools.keys()),
            {
                "prompttest_init_project",
                "prompttest_read_project",
                "prompttest_edit_project",
                "prompttest_run_project",
                "prompttest_analyze_reasoning",
                "prompttest_get_results",
            },
        )

    @patch("prompttestenv.config.init_project")
    def test_init_project_delegates_and_returns_confirmation(self, mock_init):
        result = self.mcp.tools["prompttest_init_project"]("Projects/Foo")
        mock_init.assert_called_once_with("Projects/Foo")
        self.assertIn("successfully initialized", result)

    @patch("prompttestenv.config.init_project")
    def test_init_project_wraps_exceptions(self, mock_init):
        mock_init.side_effect = RuntimeError("disk full")
        result = self.mcp.tools["prompttest_init_project"]("Projects/Foo")
        self.assertEqual(result, "Error: disk full")

    @patch("prompttestenv.runner.run_project")
    def test_run_project_delegates_with_defaults(self, mock_run):
        mock_run.return_value = "Report/out.html"
        result = self.mcp.tools["prompttest_run_project"]("Projects/Foo")
        mock_run.assert_called_once_with("Projects/Foo", "html", False, False)
        self.assertEqual(result, "Report/out.html")

    @patch("prompttestenv.runner.run_project")
    def test_run_project_forwards_retry_errors(self, mock_run):
        mock_run.return_value = "Report/out.html"
        self.mcp.tools["prompttest_run_project"]("Projects/Foo", retry_errors=True)
        mock_run.assert_called_once_with("Projects/Foo", "html", False, True)

    @patch("prompttestenv.runner.run_project")
    def test_run_project_wraps_exceptions(self, mock_run):
        mock_run.side_effect = RuntimeError("boom")
        result = self.mcp.tools["prompttest_run_project"]("Projects/Foo")
        self.assertEqual(result, "Error: boom")

    @patch("prompttestenv.runner.render_from_progress")
    def test_get_results_delegates(self, mock_render):
        mock_render.return_value = "Report content"
        result = self.mcp.tools["prompttest_get_results"]("Projects/Foo")
        mock_render.assert_called_once_with("Projects/Foo", "html")
        self.assertEqual(result, "Report content")

    @patch("prompttestenv.runner.render_from_progress")
    def test_get_results_passes_the_output_mode_through(self, mock_render):
        mock_render.return_value = "JSON report: /x/y.json"
        self.mcp.tools["prompttest_get_results"]("Projects/Foo", "json")
        mock_render.assert_called_once_with("Projects/Foo", "json")

    @patch("prompttestenv.runner.render_from_progress")
    def test_get_results_wraps_exceptions(self, mock_render):
        mock_render.side_effect = RuntimeError("boom")
        result = self.mcp.tools["prompttest_get_results"]("Projects/Foo")
        self.assertEqual(result, "Error: boom")


class TestEditingTools(unittest.TestCase):
    """The two tools an agent uses to author a project without file access."""

    def setUp(self):
        self.mcp = FakeMCP()
        register_tools(self.mcp)

    @patch("prompttestenv.projectedit.read_project")
    def test_read_project_returns_json(self, mock_read):
        mock_read.return_value = {"candidates": [], "progress_valid": True}
        result = self.mcp.tools["prompttest_read_project"]("Projects/Foo")
        mock_read.assert_called_once_with("Projects/Foo")
        self.assertEqual(json.loads(result), {"candidates": [], "progress_valid": True})

    @patch("prompttestenv.projectedit.read_project")
    def test_read_project_wraps_exceptions(self, mock_read):
        mock_read.side_effect = RuntimeError("boom")
        self.assertEqual(
            self.mcp.tools["prompttest_read_project"]("Projects/Foo"), "Error: boom"
        )

    @patch("prompttestenv.projectedit.edit_project")
    def test_edit_project_passes_its_flags_through(self, mock_edit):
        mock_edit.return_value = EditResult(ok=True, written=["candidates.json"])
        patch_doc = {"candidates": [{"name": "A"}]}
        result = self.mcp.tools["prompttest_edit_project"](
            "Projects/Foo", patch_doc, dry_run=True, force=True
        )
        mock_edit.assert_called_once_with(
            "Projects/Foo", patch_doc, dry_run=True, force=True
        )
        self.assertEqual(json.loads(result)["written"], ["candidates.json"])

    @patch("prompttestenv.projectedit.edit_project")
    def test_edit_project_reports_a_refusal_as_data_not_an_exception(self, mock_edit):
        """A blocked edit is a normal result an agent has to read, not a crash."""
        mock_edit.return_value = EditResult(
            ok=False, errors=["This edit changes the config hash"], hash_changed=True
        )
        payload = json.loads(
            self.mcp.tools["prompttest_edit_project"]("Projects/Foo", {})
        )
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["hash_changed"])
        self.assertIn("config hash", payload["errors"][0])

    @patch("prompttestenv.projectedit.edit_project")
    def test_edit_project_wraps_exceptions(self, mock_edit):
        mock_edit.side_effect = RuntimeError("boom")
        self.assertEqual(
            self.mcp.tools["prompttest_edit_project"]("Projects/Foo", {}), "Error: boom"
        )


if __name__ == "__main__":
    unittest.main()
