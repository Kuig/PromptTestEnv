from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from pathlib import Path

from prompttestenv.projectedit import EditResult

from prompttestenv.__main__ import (
    build_parser, cmd_edit, cmd_editor, cmd_init, cmd_models, cmd_render, cmd_run, cmd_show,
    main,
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
        self.assertFalse(args.retry_errors)

    def test_run_subcommand_flags(self):
        args = self.parser.parse_args(["run", "Projects/Foo", "--output-mode", "md", "--force-restart"])
        self.assertEqual(args.output_mode, "md")
        self.assertTrue(args.force_restart)

    def test_run_subcommand_retry_errors_flag(self):
        args = self.parser.parse_args(["run", "Projects/Foo", "--retry-errors"])
        self.assertTrue(args.retry_errors)
        self.assertFalse(args.force_restart)

    def test_run_subcommand_rejects_both_resume_flags(self):
        """--force-restart discards the very log --retry-errors repairs."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "Projects/Foo", "--force-restart", "--retry-errors"])

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

    def test_models_subcommand_defaults(self):
        args = self.parser.parse_args(["models"])
        self.assertIs(args.func, cmd_models)
        self.assertIsNone(args.provider)
        self.assertFalse(args.no_probe)

    def test_models_subcommand_takes_a_provider_and_no_probe(self):
        args = self.parser.parse_args(["models", "ollama", "--no-probe"])
        self.assertEqual(args.provider, "ollama")
        self.assertTrue(args.no_probe)

    def test_no_subcommand_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])


class TestMain(unittest.TestCase):
    @patch("prompttestenv.__main__.build_parser")
    @patch("prompttestenv.__main__.configure_ai_logging")
    @patch("prompttestenv.__main__.silence_sdks")
    def test_configures_ai_logging_before_dispatch(self, mock_silence, mock_cfg_log, mock_parser):
        with patch("sys.argv", ["prompttestenv", "run", "Projects/Foo"]):
            main()
        mock_silence.assert_called_once()
        mock_cfg_log.assert_called_once()
        mock_parser.return_value.parse_args.return_value.func.assert_called_once()


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
        mock_run.assert_called_once_with("Projects/Foo", "md", False, False)

    @patch("prompttestenv.runner.run_project")
    def test_cmd_run_forwards_retry_errors(self, mock_run):
        mock_run.return_value = "done"
        args = build_parser().parse_args(["run", "Projects/Foo", "--retry-errors"])
        cmd_run(args)
        mock_run.assert_called_once_with("Projects/Foo", "html", False, True)

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


class TestCmdModels(LoggerResetTestCase):
    def _run(self, argv, catalog):
        args = build_parser().parse_args(argv)
        with patch("prompttestenv.catalog.list_catalog", return_value=catalog) as mock_list, \
             patch("sys.stdout", new_callable=io.StringIO) as out:
            cmd_models(args)
        return mock_list, out.getvalue()

    def test_lists_providers_without_probing_when_asked(self):
        mock_list, out = self._run(["models", "--no-probe"], {"providers": [{
            "name": "google", "local": False, "requires_api_key": True,
            "file_types": ["image"], "supports_unload": False, "supports_embeddings": True,
        }], "error": None})
        mock_list.assert_called_once_with(None, probe_local=False)
        self.assertIn("google", out)
        self.assertIn("embeddings", out)

    def test_lists_models_with_unknown_capabilities_as_question_marks(self):
        _, out = self._run(["models", "ollama"], {"provider": "ollama", "models": [{
            "id": "gemma", "display_name": None, "reasoning": None, "tool_use": True,
            "file_types": None, "embeddings": False, "context_length": None,
        }], "error": None})
        self.assertIn("gemma", out)
        self.assertIn("reasoning=?", out)
        self.assertIn("tools=yes", out)
        self.assertIn("embeddings=no", out)

    def test_an_error_is_reported_instead_of_a_listing(self):
        _, out = self._run(["models", "openai"],
                           {"provider": "openai", "models": [], "error": "no key"})
        self.assertIn("Error: no key", out)
        self.assertNotIn("lists no models", out)


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


class TestSysPathHygiene(unittest.TestCase):
    """The CLI must not publish the package's own module names globally.

    __main__ used to insert its own directory at the front of sys.path, which
    put modules called config, api, models, logger and progress ahead of
    everything installed. Every dependency imported afterwards, streamlit and
    the provider SDKs included, would have received ours for a bare
    `import config`. It was also unnecessary: this module only ever runs as
    prompttestenv.__main__, so the package is importable already.
    """

    def test_the_package_directory_is_not_on_sys_path(self):
        import prompttestenv
        package_dir = str(Path(prompttestenv.__file__).parent)
        self.assertNotIn(package_dir, sys.path)

    def test_the_streamlit_scripts_do_not_manipulate_sys_path(self):
        """A textual guard, deliberately: both scripts are executed by path in
        another process, where an assertion on sys.path would be ambiguous."""
        import prompttestenv
        for name in ("app.py", "editor.py"):
            with self.subTest(name):
                source = (Path(prompttestenv.__file__).parent / "gui" / name).read_text(encoding="utf-8")
                self.assertNotIn("sys.path.insert", source)


class TestOutputModesConstant(unittest.TestCase):
    def test_the_cli_enforces_the_one_the_runner_enforces(self):
        """A second copy would let argparse and the core disagree."""
        from prompttestenv import models
        from prompttestenv import __main__ as cli
        self.assertIs(cli.OUTPUT_MODES, models.OUTPUT_MODES)

