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
