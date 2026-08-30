from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pathlib import Path

from prompttestenv.projectedit import EditResult

from prompttestenv.__main__ import (
    build_parser, cmd_edit, cmd_editor, cmd_init, cmd_render, cmd_run, cmd_show,
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

    def test_show_subcommand(self):
        args = self.parser.parse_args(["show", "Projects/Foo"])
        self.assertEqual(args.command, "show")
        self.assertEqual(args.project_dir, "Projects/Foo")
        self.assertIs(args.func, cmd_show)

    def test_edit_subcommand_flags(self):
        args = self.parser.parse_args(
            ["edit", "Projects/Foo", "--patch", "-", "--dry-run", "--force"]
        )
        self.assertIs(args.func, cmd_edit)
        self.assertEqual(args.patch, "-")
        self.assertTrue(args.dry_run)
        self.assertTrue(args.force)

    def test_edit_subcommand_defaults(self):
        args = self.parser.parse_args(["edit", "Projects/Foo", "--patch", "p.json"])
        self.assertFalse(args.dry_run)
        self.assertFalse(args.force)

    def test_edit_subcommand_requires_a_patch(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["edit", "Projects/Foo"])

    def test_run_subcommand_rejects_invalid_output_mode(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "Projects/Foo", "--output-mode", "bogus"])

    def test_run_subcommand_accepts_json_output_mode(self):
        args = self.parser.parse_args(["run", "Projects/Foo", "--output-mode", "json"])
        self.assertEqual(args.output_mode, "json")

    def test_render_subcommand(self):
        args = self.parser.parse_args(["render", "Projects/Foo"])
        self.assertIs(args.func, cmd_render)
        self.assertEqual(args.output_mode, "html")

    def test_render_subcommand_takes_the_same_output_modes_as_run(self):
        """Re-rendering a finished run in another format costs no LLM call."""
        for mode in ("html", "md", "json", "winner_only"):
            with self.subTest(mode=mode):
                args = self.parser.parse_args(["render", "Projects/Foo", "--output-mode", mode])
                self.assertEqual(args.output_mode, mode)

    def test_render_subcommand_rejects_invalid_output_mode(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["render", "Projects/Foo", "--output-mode", "bogus"])

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
        mock_render.assert_called_once_with("Projects/Foo", "html")

    @patch("prompttestenv.runner.render_from_progress")
    def test_cmd_render_forwards_the_output_mode(self, mock_render):
        mock_render.return_value = "done"
        args = build_parser().parse_args(["render", "Projects/Foo", "--output-mode", "json"])
        cmd_render(args)
        mock_render.assert_called_once_with("Projects/Foo", "json")


class TestCmdEdit(LoggerResetTestCase):
    """The one subcommand that exits non-zero, because scripts drive it."""

    def _args(self, *extra):
        return build_parser().parse_args(["edit", "Projects/Foo", "--patch", "-", *extra])

    @patch("prompttestenv.projectedit.edit_project")
    def test_reads_the_patch_from_stdin_and_forwards_the_flags(self, mock_edit):
        mock_edit.return_value = EditResult(ok=True, written=["candidates.json"])
        with patch("sys.stdin", io.StringIO('{"candidates": []}')):
            cmd_edit(self._args("--dry-run", "--force"))
        mock_edit.assert_called_once_with(
            "Projects/Foo", {"candidates": []}, dry_run=True, force=True
        )

    @patch("prompttestenv.projectedit.edit_project")
    def test_reads_the_patch_from_a_file(self, mock_edit):
        mock_edit.return_value = EditResult(ok=True)
        directory = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        patch_file = Path(directory) / "p.json"
        patch_file.write_text('{"judge_config": {"repetitions": 3}}', encoding="utf-8")

        args = build_parser().parse_args(
            ["edit", "Projects/Foo", "--patch", str(patch_file)]
        )
        cmd_edit(args)
        mock_edit.assert_called_once_with(
            "Projects/Foo", {"judge_config": {"repetitions": 3}},
            dry_run=False, force=False,
        )

    @patch("prompttestenv.projectedit.edit_project")
    def test_a_refused_edit_exits_non_zero(self, mock_edit):
        mock_edit.return_value = EditResult(ok=False, errors=["nope"])
        with patch("sys.stdin", io.StringIO("{}")):
            with self.assertRaises(SystemExit) as caught:
                cmd_edit(self._args())
        self.assertEqual(caught.exception.code, 1)

    def test_a_malformed_patch_exits_non_zero_without_calling_the_editor(self):
        with patch("prompttestenv.projectedit.edit_project") as mock_edit:
            with patch("sys.stdin", io.StringIO("{{{")):
                with self.assertRaises(SystemExit) as caught:
                    cmd_edit(self._args())
        self.assertEqual(caught.exception.code, 1)
        mock_edit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
