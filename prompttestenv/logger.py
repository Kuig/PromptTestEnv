"""Multi-backend logger: console, MCP (stderr), silent, or Streamlit.

Usage:
    from prompttestenv.logger import log_success, log_error, set_backend

    log_success("Operation complete")   # console (default), writes to stdout
    set_backend("mcp")
    log_success("Operation complete")   # same format, writes to stderr
    set_backend("streamlit")
    log_success("Operation complete")   # compact HTML via st.markdown
"""
from __future__ import annotations
import sys
from typing import Callable, NamedTuple


class _Level(NamedTuple):
    emoji: str
    color: str


_LEVELS: dict[str, _Level] = {
    "success": _Level("✅",  "#2ebd59"),
    "error":   _Level("❌",  "#ff4b4b"),
    "warning": _Level("⚠️", "#ffa500"),
    "action":  _Level("→",  "#00b0ff"),
    "info":    _Level("📌", "#00e5ff"),
    "save":    _Level("💾", "#2ebd59"),
    "ai":      _Level("🤖", "#b388ff"),
    "metric":  _Level("📊", "#888888"),
}


# ── Backend callables ──────────────────────────────────────────────────────────

def _console_emit(emoji: str, msg: str, color: str) -> None:
    print(f"{emoji} {msg}")

def _console_sep() -> None:
    print("─" * 40)

def _mcp_emit(emoji: str, msg: str, color: str) -> None:
    print(f"{emoji} {msg}", file=sys.stderr)

def _mcp_sep() -> None:
    print("─" * 40, file=sys.stderr)

def _silent_emit(emoji: str, msg: str, color: str) -> None:
    pass

def _silent_sep() -> None:
    pass

_emit: Callable[[str, str, str], None] = _console_emit
_sep:  Callable[[], None]              = _console_sep


def set_backend(backend: str) -> None:
    """Switch output backend.

    Args:
        backend: One of ``'console'`` (default, stdout), ``'mcp'`` (stderr),
            ``'silent'`` (discard), or ``'streamlit'``. Any unknown value
            falls back to ``'console'``.
    """
    global _emit, _sep
    if backend == "mcp":
        _emit, _sep = _mcp_emit, _mcp_sep
    elif backend == "silent":
        _emit, _sep = _silent_emit, _silent_sep
    elif backend == "streamlit":
        import streamlit as st

        def _st_emit(emoji: str, msg: str, color: str) -> None:
            st.markdown(
                f'<div style="color:{color};font-family:monospace;'
                f'line-height:1.4;margin:2px 0">{emoji} {msg}</div>',
                unsafe_allow_html=True,
            )

        def _st_sep() -> None:
            st.markdown(
                '<hr style="margin:5px 0;border:none;border-top:1px dashed #444">',
                unsafe_allow_html=True,
            )

        _emit, _sep = _st_emit, _st_sep
    else:
        _emit, _sep = _console_emit, _console_sep


# ── Generic dispatcher ─────────────────────────────────────────────────────────

def _log(level: str, msg: str) -> None:
    lvl = _LEVELS[level]
    _emit(lvl.emoji, msg, lvl.color)


# ── Public API ─────────────────────────────────────────────────────────────────

def log_success(msg: str) -> None:
    """Log a success message."""
    _log("success", msg)

def log_error(msg: str) -> None:
    """Log an error message."""
    _log("error", msg)

def log_warning(msg: str) -> None:
    """Log a warning message."""
    _log("warning", msg)

def log_action(msg: str) -> None:
    """Log an in-progress action."""
    _log("action", msg)

def log_info(msg: str) -> None:
    """Log an informational message."""
    _log("info", msg)

def log_save(msg: str) -> None:
    """Log a file-save event."""
    _log("save", msg)

def log_ai(msg: str) -> None:
    """Log an AI model interaction."""
    _log("ai", msg)

def log_metric(msg: str) -> None:
    """Log a numeric metric or statistics line."""
    _log("metric", msg)

def log_separator() -> None:
    """Print a visual separator line."""
    _sep()
