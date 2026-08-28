from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from prompttestenv.models import LlmResult, GlobalCriteria, JudgeConfig, TestCaseResult
from prompttestenv.test_judge import (
    _evaluate_assert,
    _evaluate_llm_judge,
    _evaluate_similarity,
    evaluate_with_judge,
)


class TestEvaluateAssert(unittest.TestCase):
    def test_valid_lambda_with_prefix(self):
        score, reasoning = _evaluate_assert("lambda s: (9, 'good')", "anything")
        self.assertEqual(score, 9)
        self.assertEqual(reasoning, "good")

    def test_valid_lambda_without_prefix(self):
        score, reasoning = _evaluate_assert("s: (7, 'ok')", "anything")
        self.assertEqual(score, 7)

    def test_malformed_syntax_returns_error_tuple(self):
        score, reasoning = _evaluate_assert("s: (", "anything")
        self.assertEqual(score, -1)
        self.assertIn("Error", reasoning)

    def test_non_two_tuple_return_falls_back(self):
        score, reasoning = _evaluate_assert("s: 42", "anything")
        self.assertEqual(score, -1)
        self.assertIn("did not return", reasoning)

    def test_score_is_returned_verbatim_without_clamping(self):
        """Criteria are the author's own unsandboxed lambda: range is their call."""
        score, _ = _evaluate_assert("s: (99, 'too high')", "anything")
        self.assertEqual(score, 99)
        score, _ = _evaluate_assert("s: (-5, 'too low')", "anything")
        self.assertEqual(score, -5)

    def test_author_can_signal_not_measured_explicitly(self):
        score, reasoning = _evaluate_assert("s: (-1, 'not applicable here')", "anything")
        self.assertEqual(score, -1)
        self.assertEqual(reasoning, "not applicable here")


class TestEvaluateLlmJudge(unittest.TestCase):
    def _judge_config(self, template="[{user_prompt}] [{candidate_response}] [{criteria}]"):
        jc = JudgeConfig()
        jc.test_judge.evaluation_template = template
        jc.test_judge.evaluation_system_prompt = "sys"
        return jc

    def test_missing_template_short_circuits(self):
        jc = self._judge_config(template="")
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            score, reasoning = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, -1)
        mock_llm.assert_not_called()

    def test_successful_json_response(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 8, "reasoning": "good job"}))
            score, reasoning = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, 8)
        self.assertEqual(reasoning, "good job")

    def test_system_note_leads_the_prompt_and_names_the_file(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 8, "reasoning": "r"}))
            _evaluate_llm_judge(
                prompt, "resp", jc, local_media_paths=["/proj/test_files/paper.pdf"],
            )
        sent = mock_llm.call_args.kwargs["user_prompt"]
        # Head position: a text attachment is inlined AHEAD of the whole prompt,
        # so a trailing note would arrive long after the file it announces.
        self.assertTrue(sent.startswith("[SYSTEM NOTE]"), sent[:80])
        self.assertIn("paper.pdf", sent)
        self.assertIn("file for the test is attached", sent)
        self.assertEqual(
            mock_llm.call_args.kwargs["local_media_paths"], ["/proj/test_files/paper.pdf"],
        )

    def test_system_note_is_plural_with_several_attachments(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 8, "reasoning": "r"}))
            _evaluate_llm_judge(
                prompt, "resp", jc,
                local_media_paths=["/proj/test_files/a.txt", "/proj/test_files/b.md"],
            )
        sent = mock_llm.call_args.kwargs["user_prompt"]
        self.assertIn("files for the test are attached", sent)
        self.assertIn("a.txt, b.md", sent)
        self.assertIn("Use them as ground truth", sent)

    def test_no_attachment_means_no_system_note(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 8, "reasoning": "r"}))
            _evaluate_llm_judge(prompt, "resp", jc)
        sent = mock_llm.call_args.kwargs["user_prompt"]
        self.assertNotIn("[SYSTEM NOTE]", sent)
        self.assertIsNone(mock_llm.call_args.kwargs["local_media_paths"])

    def test_response_as_json_list_uses_first_element(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps([{"score": 6, "reasoning": "ok"}]))
            score, reasoning = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, 6)

    def test_non_json_response_returns_sentinel_with_error(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text="not json")
            score, reasoning = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, -1)
        self.assertIn("failed", reasoning)

    def test_out_of_range_score_is_clamped_to_1_10(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 0, "reasoning": "x"}))
            score, _ = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, 1)
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 15, "reasoning": "x"}))
            score, _ = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, 10)
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": -3, "reasoning": "x"}))
            score, _ = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, 1)

    def test_missing_score_field_returns_sentinel(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"reasoning": "no score field"}))
            score, reasoning = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, -1)
        self.assertIn("missing or not numeric", reasoning)

    def test_non_numeric_score_returns_sentinel(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": "alto", "reasoning": "x"}))
            score, reasoning = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, -1)
        self.assertIn("missing or not numeric", reasoning)

    def test_numeric_string_score_is_accepted(self):
        jc = self._judge_config()
        prompt = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": "8", "reasoning": "x"}))
            score, _ = _evaluate_llm_judge(prompt, "resp", jc)
        self.assertEqual(score, 8)


class TestEvaluateSimilarity(unittest.TestCase):
    def test_empty_response_scores_minimum(self):
        jc = JudgeConfig()
        score, reasoning = _evaluate_similarity("target", "  ", jc)
        self.assertEqual(score, 1)

    def test_empty_criteria_scores_minimum(self):
        jc = JudgeConfig()
        score, reasoning = _evaluate_similarity("  ", "response", jc)
        self.assertEqual(score, 1)

    def test_identical_embeddings_score_ten(self):
        jc = JudgeConfig()
        with patch("prompttestenv.test_judge.get_text_embedding") as mock_embed:
            mock_embed.return_value = [1.0, 0.0]
            score, reasoning = _evaluate_similarity("target", "response", jc)
        self.assertEqual(score, 10)

    def test_embedding_failure_returns_sentinel(self):
        jc = JudgeConfig()
        with patch("prompttestenv.test_judge.get_text_embedding") as mock_embed:
            mock_embed.side_effect = RuntimeError("boom")
            score, reasoning = _evaluate_similarity("target", "response", jc)
        self.assertEqual(score, -1)
        self.assertIn("Error", reasoning)


class TestEvaluateWithJudge(unittest.TestCase):
    def _prompt(self, judge_type="assert", criteria="s: (10, 'ok')"):
        return TestCaseResult(test_id="t1", prompt="p", criteria=criteria, judge_type=judge_type)

    def test_assert_dispatch_with_global_disabled(self):
        jc = JudgeConfig()
        jc.global_criteria = GlobalCriteria(mode="none")
        result = evaluate_with_judge(self._prompt(), "response text", jc)
        self.assertEqual(result["score"], 10)
        self.assertEqual(result["global_score"], -1)

    def test_unknown_judge_type_falls_back(self):
        jc = JudgeConfig()
        jc.global_criteria = GlobalCriteria(mode="none")
        result = evaluate_with_judge(self._prompt(judge_type="bogus"), "response", jc)
        self.assertEqual(result["score"], -1)
        self.assertIn("Unknown judge_type", result["reasoning"])

    def test_unknown_global_mode_falls_back(self):
        jc = JudgeConfig()
        jc.global_criteria = GlobalCriteria(mode="bogus")
        result = evaluate_with_judge(self._prompt(), "response", jc)
        self.assertEqual(result["global_score"], -1)
        self.assertIn("Unknown global evaluation mode", result["global_reasoning"])

    def test_global_assert_dispatch(self):
        jc = JudgeConfig()
        jc.global_criteria = GlobalCriteria(mode="assert", assert_criteria="s: (7, 'globally fine')")
        result = evaluate_with_judge(self._prompt(), "response", jc)
        self.assertEqual(result["global_score"], 7)
        self.assertEqual(result["global_reasoning"], "globally fine")

    def test_attachments_reach_both_the_task_and_the_global_llm_judge(self):
        jc = JudgeConfig()
        jc.test_judge.evaluation_template = "[{user_prompt}] [{candidate_response}] [{criteria}]"
        jc.global_criteria = GlobalCriteria(mode="llm-judge", llm_judge_criteria="global rubric")
        paths = ["/proj/test_files/a.txt", "/proj/test_files/b.md"]

        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=json.dumps({"score": 8, "reasoning": "r"}))
            evaluate_with_judge(
                self._prompt(judge_type="llm-judge", criteria="task rubric"),
                "response", jc, local_media_paths=paths,
            )

        self.assertEqual(mock_llm.call_count, 2)
        for call in mock_llm.call_args_list:
            self.assertEqual(call.kwargs["local_media_paths"], paths)

    def test_similarity_and_assert_never_see_the_attachments(self):
        """By design: only the llm-judge evaluators take files."""
        jc = JudgeConfig()
        jc.global_criteria = GlobalCriteria(mode="none")
        with patch("prompttestenv.test_judge.get_llm_response") as mock_llm:
            result = evaluate_with_judge(
                self._prompt(), "response", jc, local_media_paths=["/proj/a.txt"],
            )
        self.assertEqual(result["score"], 10)
        mock_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
