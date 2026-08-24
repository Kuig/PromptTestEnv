# prompttestenv package
from __future__ import annotations

import sys
import io

# Ensure UTF-8 output on Windows (cp1252 cannot encode emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# Public library API, exposed lazily (PEP 562) so that submodules with heavy
# dependencies (unified_ai_client, jinja2) are only imported when one of these
# names is actually accessed — CLI subcommands that don't need them (e.g. init,
# mcp) keep their current fast startup.
_CONFIG_EXPORTS = (
    "init_project",
    "load_candidates",
    "load_judge_config",
    "load_test_cases",
    "load_global_criteria",
)
_RUNNER_EXPORTS = ("run_project", "render_from_progress")

__all__ = [*_CONFIG_EXPORTS, *_RUNNER_EXPORTS]


def __getattr__(name: str):
    """Lazily resolve public API re-exports on first access.

    Args:
        name: Attribute name being accessed on the ``prompttestenv`` package.

    Returns:
        The resolved function object.

    Raises:
        AttributeError: If ``name`` is not a recognized public API export.
    """
    if name in _CONFIG_EXPORTS:
        from prompttestenv import config
        return getattr(config, name)
    if name in _RUNNER_EXPORTS:
        from prompttestenv import runner
        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


