from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_tools(mcp: "FastMCP") -> None:
    """Register all PromptTestEnv MCP tools with the given FastMCP instance.

    Args:
        mcp: A FastMCP server instance.
    """

    @mcp.tool()
    def prompttest_init_project(project_dir: str) -> str:
        """Initialize a new prompt testing project directory.

        Args:
            project_dir: Path to the project directory to initialize.

        Returns:
            Confirmation message.
        """
        try:
            from prompttestenv.config import init_project
            init_project(project_dir)
            return f"Project '{project_dir}' has been successfully initialized."
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def prompttest_run_project(
        project_dir: str,
        output_mode: str = "html",
        force_restart: bool = False,
    ) -> str:
        """Run a prompt testing benchmark project.

        Args:
            project_dir: Path to the project directory.
            output_mode: Report format, one of:
                'html' — full human-readable report under Report/: verdict,
                    per-test-case scores, best responses and reasoning traces.
                'md' — the verdict text alone, as a compact readable summary.
                    Nothing is lost by choosing it: every detail stays in
                    progress.jsonl and a later render can rebuild the HTML
                    without any LLM call. Prefer this when you need to read the
                    outcome yourself without filling your context.
                'json' — the same content as 'html' in machine-readable form,
                    plus the per-repetition raw values the page reduces to
                    mean/std. Described by report.schema.json. Prefer this when
                    a program, not a person, consumes the result.
                'winner_only' — writes no file; returns one line naming the
                    highest average task score. Cheapest: skips the verdict LLM
                    call entirely.
            force_restart: If True, ignores previous progress and restarts from scratch.

        Returns:
            Path to the generated report or an error description.
        """
        try:
            from prompttestenv.runner import run_project
            return run_project(project_dir, output_mode, force_restart)
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def prompttest_analyze_reasoning(project_dir: str, force_reanalyze: bool = False) -> str:
        """Analyze the reasoning traces stored in a project's progress log.

        Makes no generation or judging calls: it reads the thinking traces the
        benchmark already recorded, so it is safe and cheap to re-run.

        Args:
            project_dir: Path to the project directory.
            force_reanalyze: If True, recompute analyses that already exist.

        Returns:
            Summary of how many traces were analyzed, or an error description.
        """
        try:
            from prompttestenv.runner import analyze_project
            return analyze_project(project_dir, force_reanalyze)
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def prompttest_get_results(project_dir: str, output_mode: str = "html") -> str:
        """Regenerate and return the benchmark report from an existing progress.jsonl.

        Makes no LLM call of any kind when the log already holds a verdict, so
        re-rendering the same run in another format is free.

        Args:
            project_dir: Path to the project directory.
            output_mode: Report format, one of:
                'html' — full human-readable report under Report/: verdict,
                    per-test-case scores, best responses and reasoning traces.
                'md' — the verdict text alone, as a compact readable summary.
                    Nothing is lost by choosing it: every detail stays in
                    progress.jsonl and a later render can rebuild the HTML
                    without any LLM call. Prefer this when you need to read the
                    outcome yourself without filling your context.
                'json' — the same content as 'html' in machine-readable form,
                    plus the per-repetition raw values the page reduces to
                    mean/std. Described by report.schema.json. Prefer this when
                    a program, not a person, consumes the result.
                'winner_only' — writes no file; returns one line naming the
                    highest average task score. Cheapest: skips the verdict LLM
                    call entirely.

        Returns:
            Report content or error description.
        """
        try:
            from prompttestenv.runner import render_from_progress
            return render_from_progress(project_dir, output_mode)
        except Exception as exc:
            return f"Error: {exc}"
