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
            output_mode: Report format — 'html', 'md', or 'winner_only'.
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
    def prompttest_get_results(project_dir: str) -> str:
        """Regenerate and return the benchmark report from an existing progress.jsonl.

        Args:
            project_dir: Path to the project directory.

        Returns:
            Report content or error description.
        """
        try:
            from prompttestenv.runner import render_from_progress
            return render_from_progress(project_dir)
        except Exception as exc:
            return f"Error: {exc}"
