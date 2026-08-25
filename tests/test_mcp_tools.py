from __future__ import annotations

import unittest
from unittest.mock import patch

from prompttestenv.mcp_tools import register_tools


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

    def test_registers_the_four_expected_tools(self):
        self.assertEqual(
            set(self.mcp.tools.keys()),
            {
                "prompttest_init_project",
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
        mock_run.assert_called_once_with("Projects/Foo", "html", False)
        self.assertEqual(result, "Report/out.html")

    @patch("prompttestenv.runner.run_project")
    def test_run_project_wraps_exceptions(self, mock_run):
        mock_run.side_effect = RuntimeError("boom")
        result = self.mcp.tools["prompttest_run_project"]("Projects/Foo")
        self.assertEqual(result, "Error: boom")

    @patch("prompttestenv.runner.render_from_progress")
    def test_get_results_delegates(self, mock_render):
        mock_render.return_value = "Report content"
        result = self.mcp.tools["prompttest_get_results"]("Projects/Foo")
        mock_render.assert_called_once_with("Projects/Foo")
        self.assertEqual(result, "Report content")

    @patch("prompttestenv.runner.render_from_progress")
    def test_get_results_wraps_exceptions(self, mock_render):
        mock_render.side_effect = RuntimeError("boom")
        result = self.mcp.tools["prompttest_get_results"]("Projects/Foo")
        self.assertEqual(result, "Error: boom")


if __name__ == "__main__":
    unittest.main()
