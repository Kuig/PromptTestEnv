from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

import prompttestenv.logger as logger


class TestConsoleBackend(unittest.TestCase):
    def setUp(self):
        logger.set_backend("console")

    def tearDown(self):
        logger.set_backend("console")

    def _captured_output(self, fn, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args)
        return buf.getvalue()

    def test_log_success_includes_emoji_and_message(self):
        out = self._captured_output(logger.log_success, "done")
        self.assertIn("✅", out)
        self.assertIn("done", out)

    def test_log_error_includes_emoji_and_message(self):
        out = self._captured_output(logger.log_error, "broke")
        self.assertIn("❌", out)
        self.assertIn("broke", out)

    def test_log_warning(self):
        out = self._captured_output(logger.log_warning, "careful")
        self.assertIn("⚠️", out)

    def test_log_action(self):
        out = self._captured_output(logger.log_action, "doing")
        self.assertIn("→", out)

    def test_log_info(self):
        out = self._captured_output(logger.log_info, "fyi")
        self.assertIn("📌", out)

    def test_log_save(self):
        out = self._captured_output(logger.log_save, "saved")
        self.assertIn("💾", out)

    def test_log_ai(self):
        out = self._captured_output(logger.log_ai, "thinking")
        self.assertIn("🤖", out)

    def test_log_metric(self):
        out = self._captured_output(logger.log_metric, "42")
        self.assertIn("📊", out)

    def test_log_separator(self):
        out = self._captured_output(logger.log_separator)
        self.assertIn("─", out)


class TestMcpBackend(unittest.TestCase):
    """The mcp backend must keep stdout clean: it carries the JSON-RPC frames."""

    def tearDown(self):
        logger.set_backend("console")

    def _captured(self, fn, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fn(*args)
        return out.getvalue(), err.getvalue()

    def test_log_goes_to_stderr_and_never_to_stdout(self):
        logger.set_backend("mcp")
        out, err = self._captured(logger.log_info, "fyi")
        self.assertEqual(out, "")
        self.assertIn("📌", err)
        self.assertIn("fyi", err)

    def test_separator_goes_to_stderr(self):
        logger.set_backend("mcp")
        out, err = self._captured(logger.log_separator)
        self.assertEqual(out, "")
        self.assertIn("─", err)


class TestSilentBackend(unittest.TestCase):
    def tearDown(self):
        logger.set_backend("console")

    def test_everything_is_discarded(self):
        logger.set_backend("silent")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            logger.log_error("broke")
            logger.log_separator()
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "")


class TestUnknownBackend(unittest.TestCase):
    def tearDown(self):
        logger.set_backend("console")

    def test_unknown_backend_falls_back_to_console(self):
        logger.set_backend("nonesuch")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            logger.log_success("done")
        self.assertIn("done", buf.getvalue())


class TestStreamlitBackend(unittest.TestCase):
    def tearDown(self):
        logger.set_backend("console")

    def test_set_backend_streamlit_routes_through_st_markdown(self):
        fake_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": fake_st}):
            logger.set_backend("streamlit")
            logger.log_success("hello")

        fake_st.markdown.assert_called_once()
        args, kwargs = fake_st.markdown.call_args
        self.assertIn("hello", args[0])
        self.assertTrue(kwargs.get("unsafe_allow_html"))


if __name__ == "__main__":
    unittest.main()
