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
    def setUp(self):
        self.jc = JudgeConfig()

    def _assert(self, criteria, response="anything"):
        return _evaluate_assert(criteria, response, self.jc)

    def test_valid_lambda_with_prefix(self):
        score, reasoning = self._assert("lambda s: (9, 'good')")
        self.assertEqual(score, 9)
        self.assertEqual(reasoning, "good")

    def test_valid_lambda_without_prefix(self):
        score, reasoning = self._assert("s: (7, 'ok')")
        self.assertEqual(score, 7)

    def test_malformed_syntax_returns_error_tuple(self):
        score, reasoning = self._assert("s: (")
        self.assertEqual(score, -1)
        self.assertIn("Error", reasoning)

    def test_non_two_tuple_return_falls_back(self):
        score, reasoning = self._assert("s: 42")
        self.assertEqual(score, -1)
        self.assertIn("did not return", reasoning)

    def test_score_is_returned_verbatim_without_clamping(self):
        """Criteria are the author's own unsandboxed lambda: range is their call."""
        score, _ = self._assert("s: (99, 'too high')")
        self.assertEqual(score, 99)
        score, _ = self._assert("s: (-5, 'too low')")
        self.assertEqual(score, -5)

    def test_author_can_signal_not_measured_explicitly(self):
        score, reasoning = self._assert("s: (-1, 'not applicable here')")
        self.assertEqual(score, -1)
        self.assertEqual(reasoning, "not applicable here")

    def test_scaffold_example_criteria_still_works(self):
        """The string init writes into a new project's test_cases.json."""
        expr = "s: (10, 'Correct comma count') if s.count(',') == 2 else (1, 'Wrong comma count')"
        score, reasoning = self._assert(expr, "a, b, c")
        self.assertEqual(score, 10)
        self.assertEqual(reasoning, "Correct comma count")
        score, _ = self._assert(expr, "no commas here")
        self.assertEqual(score, 1)

    def test_int_truncates_a_float_score_without_clamping(self):
        score, _ = self._assert("s: (9.9, 'truncated, not rounded, not clamped')")
        self.assertEqual(score, 9)

    def test_re_is_available_inside_the_lambda(self):
        expr = (
            "s: (10, 'found answer ' + re.search(r'ANSWER=(\\d+)', s).group(1)) "
            "if re.search(r'ANSWER=(\\d+)', s) else (1, 'no answer token')"
        )
        score, reasoning = self._assert(expr, "blah blah ANSWER=42 blah")
        self.assertEqual(score, 10)
        self.assertIn("42", reasoning)

    def test_math_is_available_inside_the_lambda(self):
        score, reasoning = self._assert("s: (int(math.pi * 2), 'two pi')")
        self.assertEqual(score, 6)

    def test_statistics_is_available_inside_the_lambda(self):
        expr = "s: (int(statistics.mean([2, 4, 6, 8])), 'mean of the run')"
        score, _ = self._assert(expr)
        self.assertEqual(score, 5)

    def test_json_is_available_inside_the_lambda(self):
        expr = "s: (json.loads(s)['score'], json.loads(s)['why'])"
        score, reasoning = self._assert(expr, '{"score": 8, "why": "parsed from json"}')
        self.assertEqual(score, 8)
        self.assertEqual(reasoning, "parsed from json")

    def test_datetime_is_available_inside_the_lambda(self):
        expr = (
            "s: (10, 'is a leap year') "
            "if (datetime.date(2024, 2, 29).month == 2) else (1, 'no')"
        )
        score, reasoning = self._assert(expr)
        self.assertEqual(score, 10)
        self.assertEqual(reasoning, "is a leap year")

    def test_string_is_available_inside_the_lambda(self):
        expr = "s: (10, 'all letters') if set(s) <= set(string.ascii_letters) else (1, 'has non-letters')"
        score, _ = self._assert(expr, "abcDEF")
        self.assertEqual(score, 10)
        score, _ = self._assert(expr, "abc123")
        self.assertEqual(score, 1)

    def test_similarity_helper_is_available_and_branchable(self):
        with patch("prompttestenv.test_judge.get_text_embedding") as mock_embed:
            # Both texts embed to the same vector, so the raw cosine is 1.0.
            mock_embed.return_value = [1.0, 0.0, 0.0]
            expr = "s: (10, 'close enough') if similarity(s, 'expected text') > 0.8 else (1, 'too far')"
            score, reasoning = self._assert(expr, "some response")
        self.assertEqual(score, 10)
        self.assertEqual(reasoning, "close enough")
        self.assertTrue(mock_embed.called)

    def test_similarity_helper_low_score_branch(self):
        with patch("prompttestenv.test_judge.get_text_embedding") as mock_embed:
            # Orthogonal vectors: raw cosine is 0.0, below the lambda's threshold.
            mock_embed.side_effect = [[1.0, 0.0], [0.0, 1.0]]
            expr = "s: (10, 'close') if similarity(s, 'expected') > 0.8 else (1, 'too far')"
            score, reasoning = self._assert(expr, "some response")
        self.assertEqual(score, 1)
        self.assertEqual(reasoning, "too far")

    def test_similarity_backend_failure_degrades_to_not_measured(self):
        with patch("prompttestenv.test_judge.get_text_embedding") as mock_embed:
            mock_embed.side_effect = RuntimeError("embedding backend down")
            score, reasoning = self._assert(
                "s: similarity(s, 'x') > 0.5 and (10, 'ok') or (1, 'no')", "resp",
            )
        self.assertEqual(score, -1)
        self.assertIn("Error", reasoning)

    def test_module_internal_names_are_no_longer_reachable(self):
        """Before the explicit namespace, the lambda saw this module's globals
        (get_llm_response, os, logger, ...). It must not any more."""
        for leaked in ("get_llm_response", "os", "logger", "TestCaseResult"):
            score, reasoning = self._assert(f"s: ({leaked} is not None, 'leaked')")
            self.assertEqual(score, -1, leaked)
            self.assertIn("Error", reasoning)

    def test_builtins_including_import_are_still_reachable(self):
        """The namespace is explicit but not a sandbox: full builtins stay."""
        score, _ = self._assert("s: (len(s), 'len via builtin')", "abcd")
        self.assertEqual(score, 4)
        score, reasoning = self._assert(
            "s: (10, __import__('math').floor(3.7)) if True else (1, 'no')",
        )
        self.assertEqual(score, 10)
        self.assertEqual(reasoning, "3")

    def test_argument_is_the_raw_unstripped_response(self):
        score, _ = self._assert("s: (len(s), 'raw length keeps whitespace')", "  hi  ")
        self.assertEqual(score, 6)


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
