from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

import prompttestenv.logger as logger
from prompttestenv.config import init_project
from prompttestenv.runner import run_project, render_from_progress

logger.set_backend("streamlit")

st.set_page_config(page_title="PromptTestEnv", page_icon="🧪", layout="wide")
st.title("🧪 PromptTestEnv — LLM Benchmarking")

if "project_dir" not in st.session_state:
    st.session_state.project_dir = ""

with st.sidebar:
    st.header("⚙️ Project")
    project_dir = st.text_input(
        "Project directory path",
        value=st.session_state.project_dir,
        placeholder="Projects/MyBenchmark"
    )
    st.session_state.project_dir = project_dir

    if st.button("📁 Browse...", use_container_width=True):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        
        init_dir = st.session_state.project_dir if st.session_state.project_dir else str(Path.cwd())
        if not Path(init_dir).is_absolute():
            init_dir = str(Path.cwd() / init_dir)
            
        selected_dir = filedialog.askdirectory(initialdir=init_dir)
        if selected_dir:
            try:
                rel_path = Path(selected_dir).relative_to(Path.cwd())
                st.session_state.project_dir = rel_path.as_posix()
            except ValueError:
                st.session_state.project_dir = Path(selected_dir).as_posix()
            st.rerun()

    st.divider()
    output_mode = st.selectbox("Output mode", ["html", "md", "winner_only"])
    force_restart = st.checkbox("Force restart (ignore progress)")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📁 Initialize Project", use_container_width=True):
        if not project_dir:
            st.error("Project directory is required.")
        else:
            try:
                init_project(project_dir)
                st.success(f"Initialized: {project_dir}")
            except Exception as exc:
                st.error(f"Error: {exc}")

with col2:
    if st.button("▶️ Run Benchmark", type="primary", use_container_width=True):
        if not project_dir:
            st.error("Project directory is required.")
        else:
            with st.spinner("Running benchmark..."):
                try:
                    result = run_project(project_dir, output_mode, force_restart)
                    st.success("Benchmark complete.")
                    st.text(result)
                except Exception as exc:
                    st.error(f"Error: {exc}")

with col3:
    if st.button("📊 Render from Progress", use_container_width=True):
        if not project_dir:
            st.error("Project directory is required.")
        else:
            with st.spinner("Rendering..."):
                try:
                    result = render_from_progress(project_dir)
                    st.success("Done.")
                    st.text(result)
                except Exception as exc:
                    st.error(f"Error: {exc}")
