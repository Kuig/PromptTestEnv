"""Streamlit interfaces: `app.py` runs benchmarks, `editor.py` edits projects.

This file exists so `prompttestenv.gui` is a real package. Without it,
`packages.find` in pyproject.toml does not pick the directory up and `gui/*.py`
ships in no wheel at all — `prompttestenv gui` then works only from an editable
install, which points at the source tree.

Import only `common` and `projectio` from here. Importing `app` or `editor`
executes a Streamlit script at import time: `st.set_page_config` runs outside a
script run, and `logger.set_backend("streamlit")` leaks into whatever process
did the import (see tests/testutils.py).
"""
