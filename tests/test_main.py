from __future__ import annotations

import unittest
from unittest.mock import patch

from pathlib import Path

from prompttestenv.__main__ import (
    build_parser, cmd_editor, cmd_init, cmd_render, cmd_run,
)
from testutils import LoggerResetTestCase


class TestBuildParser(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_init_subcommand(self):
        args = self.parser.parse_args(["init", "Projects/Foo"])
        self.assertEqual(args.command, "init")
        self.assertEqual(args.project_dir, "Projects/Foo")
        self.assertIs(args.func, cmd_init)

    def test_run_subcommand_defaults(self):
        args = self.parser.parse_args(["run", "Projects/Foo"])
        self.assertEqual(args.output_mode, "html")
        self.assertFalse(args.force_restart)

    def test_run_subcommand_flags(self):
        args = self.parser.parse_args(["run", "Projects/Foo", "--output-mode", "md", "--force-restart"])
        self.assertEqual(args.output_mode, "md")
        self.assertTrue(args.force_restart)

    def test_run_subcommand_rejects_invalid_output_mode(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "Projects/Foo", "--output-mode", "bogus"])

    def test_render_subcommand(self):
        args = self.parser.parse_args(["render", "Projects/Foo"])
        self.assertIs(args.func, cmd_render)

    def test_mcp_and_gui_subcommands_need_no_project_dir(self):
        args_mcp = self.parser.parse_args(["mcp"])
        args_gui = self.parser.parse_args(["gui"])
        self.assertEqual(args_mcp.command, "mcp")
        self.assertEqual(args_gui.command, "gui")

    def test_editor_subcommand_needs_no_project_dir(self):
        args = self.parser.parse_args(["editor"])
        self.assertEqual(args.command, "editor")
        self.assertIs(args.func, cmd_editor)
        self.assertFalse(hasattr(args, "project_dir"))

    def test_no_subcommand_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])


class TestCmdEditor(unittest.TestCase):
    @patch("prompttestenv.__main__.subprocess.run")
    def test_launches_streamlit_on_an_existing_script(self, mock_run):
        cmd_editor(build_parser().parse_args(["editor"]))

        mock_run.assert_called_once()
        argv = mock_run.call_args.args[0]
        self.assertEqual(argv[1:4], ["-m", "streamlit", "run"])
        self.assertTrue(Path(argv[4]).is_file(), f"{argv[4]} does not exist")
        self.assertEqual(Path(argv[4]).name, "editor.py")


class TestCmdHandlers(LoggerResetTestCase):
    @patch("prompttestenv.config.init_project")
    def test_cmd_init_delegates(self, mock_init):
        args = build_parser().parse_args(["init", "Projects/Foo"])
        cmd_init(args)
        mock_init.assert_called_once_with("Projects/Foo")

    @patch("prompttestenv.runner.run_project")
    def test_cmd_run_delegates_with_parsed_args(self, mock_run):
        mock_run.return_value = "done"
        args = build_parser().parse_args(["run", "Projects/Foo", "--output-mode", "md"])
        cmd_run(args)
        mock_run.assert_called_once_with("Projects/Foo", "md", False)

    @patch("prompttestenv.runner.render_from_progress")
    def test_cmd_render_delegates(self, mock_render):
        mock_render.return_value = "done"
        args = build_parser().parse_args(["render", "Projects/Foo"])
        cmd_render(args)
        mock_render.assert_called_once_with("Projects/Foo")


if __name__ == "__main__":
    unittest.main()
