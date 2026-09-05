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
    def prompttest_read_project(project_dir: str) -> str:
        """Read a benchmark project's whole configuration, as JSON.

        Returns the four config files exactly as they sit on disk, unknown keys
        included, which is what a patch for prompttest_edit_project has to be
        written against. System prompts come back with their text; attachments
        come back as names and sizes only, since one can be a large binary.

        Also reports `errors` and `warnings` from the same validation the editor
        runs, and `progress_valid`: False means an existing progress.jsonl no
        longer matches the config on disk, so the next run would start over.

        Args:
            project_dir: Path to the project directory.

        Returns:
            A JSON object, or an error description.
        """
        try:
            import json
            from prompttestenv.projectedit import read_project
            return json.dumps(read_project(project_dir), indent=2, ensure_ascii=False)
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def prompttest_edit_project(
        project_dir: str,
        patch: dict,
        dry_run: bool = False,
        force: bool = False,
    ) -> str:
        """Create or modify a benchmark project's configuration, without writing files.

        Send only what changes. Candidates are matched by `name` and test cases
        by `id`: a matching entry is merged into, a new one is appended, and
        everything the patch does not mention is left byte for byte alone. Run
        prompttest_init_project first; this tool edits an existing directory.

        Patch keys, all optional:

            {
              "candidates":      [{"name": "Baseline", "temperature": 0.9}],
              "test_cases":      [{"id": "t1", "prompt": "...", "criteria": "..."}],
              "judge_config":    {"repetitions": 3,
                                  "test_judge": {"model": "gemini-3-flash"}},
              "global_criteria": {"mode": "none"},
              "system_prompts":  {"terse.txt": "Be terse."},
              "test_files":      {"notes.md": "text content"},
              "delete": {"candidates": ["Old"], "test_cases": ["t9"],
                         "system_prompts": ["old.txt"], "test_files": ["old.csv"]},
              "order":  {"candidates": ["Baseline", "Challenger"]}
            }

        An unknown key is an error, not a no-op, so a typo is never silent.
        Attachments must be text here; a binary one has to be placed in
        test_files/ directly.

        IMPORTANT: when the project already holds a progress.jsonl and the edit
        would change the config hash, this REFUSES and explains why, because the
        next run would discard that finished work and start over. Read the
        message, then either accept it or pass force=True deliberately. Editing
        only `reasoning_analysis` or the `reasoning_judge` block never triggers
        this. Editing system_prompts/ or test_files/ does not either, but for the
        opposite reason: those are outside the hash, so a resumed run silently
        mixes old and new responses.

        Args:
            project_dir: Path to an existing project directory.
            patch: The patch document described above.
            dry_run: Report what would change and write nothing. Never refuses
                over the hash; it tells you force would be needed.
            force: Write even when the edit invalidates progress.jsonl.

        Returns:
            A JSON object with `ok`, `written`, `deleted`, `errors`, `warnings`,
            `hash_changed`, `stored_hash` and `new_hash`.
        """
        try:
            import json
            from prompttestenv.projectedit import edit_project
            result = edit_project(project_dir, patch, dry_run=dry_run, force=force)
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    def prompttest_run_project(
        project_dir: str,
        output_mode: str = "html",
        force_restart: bool = False,
        retry_errors: bool = False,
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
            retry_errors: If True, resumes as usual but also redoes the steps
                whose stored result is a failure placeholder rather than a real
                model answer: a '[TIMEOUT EXCEEDED]' response, or a -1 score
                whose reasoning is a framework error. Use this to repair a run
                that finished with some cells failed, instead of paying for
                every response again with force_restart. A deliberate -1 from
                an assert lambda is never retried.

        Returns:
            Path to the generated report or an error description.
        """
        try:
            from prompttestenv.runner import run_project
            return run_project(project_dir, output_mode, force_restart, retry_errors)
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
