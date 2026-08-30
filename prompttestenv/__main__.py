"""PromptTestEnv — CLI entry point.

Usage examples:
    prompttestenv init Projects/MyBenchmark
    prompttestenv run Projects/MyBenchmark --output-mode html
    prompttestenv run Projects/MyBenchmark --force-restart
    prompttestenv analyze Projects/MyBenchmark
    prompttestenv render Projects/MyBenchmark --output-mode json
    prompttestenv show Projects/MyBenchmark
    prompttestenv edit Projects/MyBenchmark --patch patch.json
    prompttestenv mcp
    prompttestenv gui
    prompttestenv editor
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# UTF-8 console setup is handled once by prompttestenv/__init__.py, which
# always runs first when this module is imported as part of the package.
sys.path.insert(0, str(Path(__file__).parent))

from unified_ai_client import silence_sdks

# The report formats every entry point offers. "md" is a text summary of the
# verdict alone, "json" the same content the HTML carries in machine-readable
# form (see report.schema.json), "winner_only" writes no file at all.
OUTPUT_MODES = ["html", "md", "json", "winner_only"]


def cmd_init(args: argparse.Namespace) -> None:
    """Handle the ``init`` subcommand."""
    from prompttestenv.config import init_project
    import prompttestenv.logger as logger
    init_project(args.project_dir)
    logger.log_success(f"Initialized project at: {args.project_dir}")
    logger.log_warning("Edit candidates.json, judge_config.json and test_cases.json before running.")


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand."""
    from prompttestenv.runner import run_project
    import prompttestenv.logger as logger
    result = run_project(args.project_dir, args.output_mode, args.force_restart)
    logger.log_info(result)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the ``analyze`` subcommand."""
    from prompttestenv.runner import analyze_project
    import prompttestenv.logger as logger
    result = analyze_project(args.project_dir, args.force_reanalyze)
    logger.log_info(result)


def cmd_render(args: argparse.Namespace) -> None:
    """Handle the ``render`` subcommand."""
    from prompttestenv.runner import render_from_progress
    import prompttestenv.logger as logger
    result = render_from_progress(args.project_dir, args.output_mode)
    logger.log_info(result)


def cmd_show(args: argparse.Namespace) -> None:
    """Handle the ``show`` subcommand — print the project as JSON."""
    import json
    from prompttestenv.projectedit import read_project
    print(json.dumps(read_project(args.project_dir), indent=2, ensure_ascii=False))


def cmd_edit(args: argparse.Namespace) -> None:
    """Handle the ``edit`` subcommand — apply a patch to a project.

    The only subcommand that exits non-zero on failure. It is meant to be driven
    by scripts, where a silently failing edit in the middle of a pipeline is
    worse than the inconsistency with the other commands.
    """
    import prompttestenv.logger as logger
    from prompttestenv.projectedit import edit_project, parse_patch

    source = sys.stdin.read() if args.patch == "-" else Path(args.patch).read_text(
        encoding="utf-8"
    )
    try:
        patch = parse_patch(source)
    except ValueError as exc:
        logger.log_error(f"Error: {exc}")
        sys.exit(1)

    result = edit_project(
        args.project_dir, patch, dry_run=args.dry_run, force=args.force
    )
    for message in result.warnings:
        logger.log_warning(message)
    if result.ok:
        logger.log_success(result.summary())
    else:
        for message in result.errors:
            logger.log_error(message)
        sys.exit(1)


def cmd_mcp(args: argparse.Namespace) -> None:
    """Handle the ``mcp`` subcommand — start the FastMCP server on stdio."""
    import prompttestenv.logger as logger

    # Before anything can log: stdio transport carries the JSON-RPC frames on
    # stdout, so every log line has to go to stderr instead.
    logger.set_backend("mcp")
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.log_error("'mcp' library not installed. Run: pip install mcp")
        sys.exit(1)
    from prompttestenv.mcp_tools import register_tools
    mcp = FastMCP("PromptTestEnv")
    register_tools(mcp)
    mcp.run()


def cmd_gui(args: argparse.Namespace) -> None:
    """Handle the ``gui`` subcommand — launch the Streamlit web interface."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "gui" / "app.py")])


def cmd_editor(args: argparse.Namespace) -> None:
    """Handle the ``editor`` subcommand — launch the Streamlit project editor."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "gui" / "editor.py")])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prompttestenv",
        description="PromptTestEnv — LLM prompt benchmarking tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Initialize a new benchmark project.")
    p_init.add_argument("project_dir", help="Path to the project directory.")
    p_init.set_defaults(func=cmd_init)

    p_run = subparsers.add_parser("run", help="Run the benchmark on an existing project.")
    p_run.add_argument("project_dir", help="Path to the project directory.")
    p_run.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default="html",
        help="Report format (default: html).",
    )
    p_run.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore previous progress.jsonl and restart from scratch.",
    )
    p_run.set_defaults(func=cmd_run)

    p_analyze = subparsers.add_parser(
        "analyze",
        help="Run only the reasoning analysis on an existing progress.jsonl.",
    )
    p_analyze.add_argument("project_dir", help="Path to the project directory.")
    p_analyze.add_argument(
        "--force-reanalyze",
        action="store_true",
        help="Recompute reasoning analyses that already exist.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_render = subparsers.add_parser(
        "render",
        help="Regenerate reports from an existing progress.jsonl without re-running.",
    )
    p_render.add_argument("project_dir", help="Path to the project directory.")
    p_render.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default="html",
        help="Report format (default: html).",
    )
    p_render.set_defaults(func=cmd_render)

    p_show = subparsers.add_parser(
        "show",
        help="Print a project's configuration as JSON, with its validation findings.",
    )
    p_show.add_argument("project_dir", help="Path to the project directory.")
    p_show.set_defaults(func=cmd_show)

    p_edit = subparsers.add_parser(
        "edit",
        help="Apply a JSON patch to a project's configuration.",
    )
    p_edit.add_argument("project_dir", help="Path to the project directory.")
    p_edit.add_argument(
        "--patch",
        required=True,
        metavar="FILE",
        help="Patch document to apply. Use '-' to read it from stdin.",
    )
    p_edit.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    p_edit.add_argument(
        "--force",
        action="store_true",
        help="Write even when the edit invalidates an existing progress.jsonl.",
    )
    p_edit.set_defaults(func=cmd_edit)

    p_mcp = subparsers.add_parser("mcp", help="Start the MCP server on stdio.")
    p_mcp.set_defaults(func=cmd_mcp)

    p_gui = subparsers.add_parser("gui", help="Launch Streamlit web interface.")
    p_gui.set_defaults(func=cmd_gui)

    p_editor = subparsers.add_parser(
        "editor", help="Launch the Streamlit project editor (create/modify projects)."
    )
    p_editor.set_defaults(func=cmd_editor)

    return parser


def main() -> None:
    """Main entry point for the PromptTestEnv CLI."""
    silence_sdks()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
