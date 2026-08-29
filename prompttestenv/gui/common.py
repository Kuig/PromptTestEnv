"""Helpers shared by the two Streamlit apps (`app.py` and `editor.py`).

Deliberately imports no Streamlit: this module must stay importable from a
plain unit test without starting a script run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ERROR_PREFIXES = ("Error:", "ERROR:")
_INCOMPLETE_PREFIXES = (
    "No progress found",
    "No generated responses found",
    "Partial progress:",
)


def status_of(result: str) -> str:
    """Classify a runner return string into a Streamlit banner level.

    ``run_project`` / ``analyze_project`` / ``render_from_progress`` never raise:
    they return a descriptive string. Error strings start with ``Error:`` /
    ``ERROR:``; no-op or partial outcomes start with one of
    ``_INCOMPLETE_PREFIXES``; anything else is a real success (report path,
    winner summary, ...).

    Args:
        result: The string a runner entry point returned.

    Returns:
        One of ``"error"``, ``"warning"``, ``"success"``.
    """
    if result.startswith(_ERROR_PREFIXES):
        return "error"
    if result.startswith(_INCOMPLETE_PREFIXES):
        return "warning"
    return "success"


# The success strings runner._generate_output builds around a written file.
_REPORT_PREFIXES = (
    "Full process complete. HTML Report: ",
    "Markdown generated: ",
    "JSON report: ",
)


def report_path(result: str) -> Path | None:
    """The report file a runner result points at, if it points at one.

    Returns None for the results that name no file — a `winner_only` summary, a
    partial-progress note, or an error.

    Args:
        result: The string a runner entry point returned.

    Returns:
        The path, or None when there is no readable file to offer.
    """
    for prefix in _REPORT_PREFIXES:
        if result.startswith(prefix):
            path = Path(result[len(prefix):].strip())
            return path if path.is_file() else None
    return None


PICKER_SNIPPET = """
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
path = filedialog.askdirectory(initialdir=sys.argv[1] or None)
root.destroy()
sys.stdout.write(path or "")
"""


def pick_directory(initialdir: str) -> str | None:
    """Open a native folder picker and return the chosen path.

    Run in a child process: Streamlit executes its scripts on a ``ScriptRunner``
    thread, and Tkinter must own the process's main thread ("main thread is not
    in main loop"). The subprocess gets its own.

    Args:
        initialdir: Directory the dialog opens in.

    Returns:
        The selected path, or None if the user cancelled.

    Raises:
        RuntimeError: If no GUI is available (headless server, no Tk).
    """
    proc = subprocess.run(
        [sys.executable, "-c", PICKER_SNIPPET, initialdir],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"exit code {proc.returncode}")
    return proc.stdout.strip() or None
