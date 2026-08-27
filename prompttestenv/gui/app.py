from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

import prompttestenv.logger as logger
from prompttestenv.config import init_project
from prompttestenv.runner import analyze_project, run_project, render_from_progress

logger.set_backend("streamlit")

st.set_page_config(page_title="PromptTestEnv", page_icon="🧪", layout="wide")
st.title("🧪 PromptTestEnv — LLM Benchmarking")

st.session_state.setdefault("project_dir", "")
st.session_state.setdefault("pending_action", None)
st.session_state.setdefault("last_result", None)

_ERROR_PREFIXES = ("Error:", "ERROR:")
_INCOMPLETE_PREFIXES = (
    "No progress found",
    "No generated responses found",
    "Partial progress:",
)


def _status_of(result: str) -> str:
    """Classify a runner return string into a Streamlit banner level.

    ``run_project`` / ``analyze_project`` / ``render_from_progress`` never raise:
    they return a descriptive string. Error strings start with ``Error:`` /
    ``ERROR:``; no-op or partial outcomes start with one of
    ``_INCOMPLETE_PREFIXES``; anything else is a real success (report path,
    winner summary, ...).
    """
    if result.startswith(_ERROR_PREFIXES):
        return "error"
    if result.startswith(_INCOMPLETE_PREFIXES):
        return "warning"
    return "success"


_PICKER_SNIPPET = """
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
path = filedialog.askdirectory(initialdir=sys.argv[1] or None)
root.destroy()
sys.stdout.write(path or "")
"""


def _pick_directory(initialdir: str) -> str | None:
    """Open a native folder picker and return the chosen path.

    Run in a child process: Streamlit executes this script on a ``ScriptRunner``
    thread, and Tkinter must own the process's main thread ("main thread is not
    in main loop"). The subprocess gets its own. Raises if no GUI is available.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PICKER_SNIPPET, initialdir],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"exit code {proc.returncode}")
    return proc.stdout.strip() or None


with st.sidebar:
    st.header("⚙️ Project")
    project_dir = st.text_input(
        "Project directory path",
        value=st.session_state.project_dir,
        placeholder="Projects/MyBenchmark",
    )
    st.session_state.project_dir = project_dir

    if st.button("📁 Browse...", use_container_width=True):
        init_dir = st.session_state.project_dir or str(Path.cwd())
        if not Path(init_dir).is_absolute():
            init_dir = str(Path.cwd() / init_dir)

        picked = None
        try:
            picked = _pick_directory(init_dir)
        except Exception as exc:
            st.warning(f"Folder picker unavailable: {exc}. Type the path manually.")

        if picked:
            try:
                st.session_state.project_dir = Path(picked).relative_to(Path.cwd()).as_posix()
            except ValueError:
                st.session_state.project_dir = Path(picked).as_posix()
            st.rerun()

    st.divider()
    output_mode = st.selectbox("Output mode", ["html", "md", "winner_only"])
    force_restart = st.checkbox("Force restart (ignore progress)")
    force_reanalyze = st.checkbox("Force reanalyze (redo reasoning analysis)")


def _request(action: str) -> None:
    """Queue an action for the full-width execution block below the columns."""
    if not project_dir:
        st.error("Project directory is required.")
    else:
        st.session_state.pending_action = (action, project_dir)


col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📁 Initialize Project", use_container_width=True):
        _request("init")

with col2:
    if st.button("▶️ Run Benchmark", type="primary", use_container_width=True):
        _request("run")

with col3:
    if st.button("🧠 Analyze Reasoning", use_container_width=True):
        _request("analyze")

with col4:
    if st.button("📊 Render from Progress", use_container_width=True):
        _request("render")

# ── Full-width execution + result area ────────────────────────────────────────
# The work runs here, outside the narrow columns, so the logger's inline
# Streamlit output during a run is readable at full width, and the outcome is
# stored in session_state so it survives the reruns that later widget clicks
# trigger.

_SPINNERS = {
    "init": "Initializing project...",
    "run": "Running benchmark...",
    "analyze": "Analyzing reasoning traces...",
    "render": "Rendering report...",
}

pending = st.session_state.pending_action
if pending:
    action, pdir = pending
    st.session_state.pending_action = None
    with st.spinner(_SPINNERS[action]):
        try:
            if action == "init":
                init_project(pdir)
                st.session_state.last_result = ("success", f"Initialized: {pdir}")
            elif action == "run":
                result = run_project(pdir, output_mode, force_restart)
                st.session_state.last_result = (_status_of(result), result)
            elif action == "analyze":
                result = analyze_project(pdir, force_reanalyze)
                st.session_state.last_result = (_status_of(result), result)
            elif action == "render":
                result = render_from_progress(pdir)
                st.session_state.last_result = (_status_of(result), result)
        except Exception as exc:
            st.session_state.last_result = ("error", f"Error: {exc}")

last = st.session_state.last_result
if last:
    status, text = last
    if status == "error":
        st.error(text)
    elif status == "warning":
        st.warning(text)
    else:
        st.success(text)
        st.code(text)
