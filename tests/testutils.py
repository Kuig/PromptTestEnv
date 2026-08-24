"""Shared test helpers — not a test file itself (no Test* classes here).

Deliberately no ``tests/__init__.py`` in this directory: ``unittest discover``
run as ``python -m unittest discover -s tests`` (no ``-t``) defaults the
top-level directory to ``tests`` itself, which puts ``tests/`` on ``sys.path``
and lets every file in it import this module as a bare top-level module
(``from testutils import ...``). Adding an ``__init__.py`` would instead make
unittest climb to the project root as the top-level directory and require
``from tests.testutils import ...`` everywhere — pick one convention and this
file assumes the no-``__init__.py`` one throughout the suite.

Provides:
- LoggerResetTestCase: resets prompttestenv.logger's module-global backend
  to "console" after every test, so a test that flips it to "streamlit"
  (or imports prompttestenv.gui.app, which does so as an import-time side
  effect) never leaks state into unrelated tests run afterward.
- make_temp_project(): copies the read-only tests/fixtures/smoke_project/
  fixture into a fresh temp directory and returns its path. Callers must
  never write into fixtures/smoke_project/ directly, since generation,
  evaluation and progress-tracking code all write into the project
  directory they're given (progress.jsonl, Report/, verdict_prompt_debug.txt).
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import prompttestenv.logger as logger

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMOKE_PROJECT_FIXTURE = FIXTURES_DIR / "smoke_project"


class LoggerResetTestCase(unittest.TestCase):
    """Base TestCase that resets the logger backend to 'console' after each test."""

    def tearDown(self) -> None:
        logger.set_backend("console")
        super().tearDown()


def make_temp_project() -> str:
    """Copy the smoke_project fixture into a new temp directory.

    Returns:
        Path (str) to the temp directory containing a fresh copy of the
        fixture's config files. The caller owns cleanup, e.g.:

            project_dir = make_temp_project()
            self.addCleanup(shutil.rmtree, project_dir, ignore_errors=True)
    """
    tmp_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
    shutil.copytree(SMOKE_PROJECT_FIXTURE, tmp_dir, dirs_exist_ok=True)
    return tmp_dir
