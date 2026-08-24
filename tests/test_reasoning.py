from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from prompttestenv.models import JudgeConfig, ReasoningStats
from prompttestenv.reasoning import (
    aggregate_reasoning_stats,
    analyze_reasoning,
    compute_reasoning_percentages,
)


class TestComputeReasoningPercentages(unittest.TestCase):
    def test_normal_case(self):
        stats = ReasoningStats(
            interpretation="ab", planning="abcd", pure_reasoning="abcdefgh", output_formulation="abcdefghij",
        )
        compute_reasoning_percentages(stats)
        total_pct = stats.interpretation_pct + stats.planning_pct + stats.pure_reasoning_pct + stats.output_formulation_pct
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_zero_total_leaves_percentages_at_zero(self):
        stats = ReasoningStats()
        compute_reasoning_percentages(stats)
        self.assertEqual(stats.interpretation_pct, 0.0)
        self.assertEqual(stats.pure_reasoning_pct, 0.0)


class TestAggregateReasoningStats(unittest.TestCase):
    def test_empty_list_returns_empty_dict(self):
        self.assertEqual(aggregate_reasoning_stats([]), {})

    def test_single_item_has_zero_std(self):
        analyses = [ReasoningStats(alt_path=3, autocorrect=1, alignment_score=8).to_dict()]
        result = aggregate_reasoning_stats(analyses)
        self.assertEqual(result["avg_alt_path"], 3.0)
        self.assertEqual(result["std_alt_path"], 0.0)

    def test_multiple_items_compute_mean_and_std(self):
        analyses = [
            ReasoningStats(alt_path=2, alignment_score=6).to_dict(),
            ReasoningStats(alt_path=4, alignment_score=10).to_dict(),
        ]
        result = aggregate_reasoning_stats(analyses)
        self.assertAlmostEqual(result["avg_alt_path"], 3.0)
        self.assertGreater(result["std_alt_path"], 0.0)


class TestAnalyzeReasoning(unittest.TestCase):
    def _judge_config(self, reasoning_analysis=True, seg_template="Seg {reasoning_text}", metrics_template="Met {reasoning_text} {candidate_response}"):
        jc = JudgeConfig()
        jc.reasoning_analysis = reasoning_analysis
        jc.reasoning_judge.segmentation_template = seg_template
        jc.reasoning_judge.metrics_template = metrics_template
        return jc

    def test_disabled_analysis_returns_none_without_calling_llm(self):
        jc = self._judge_config(reasoning_analysis=False)
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            result = analyze_reasoning("some trace", jc)
        self.assertIsNone(result)
        mock_llm.assert_not_called()

    def test_empty_reasoning_text_returns_none(self):
        jc = self._judge_config()
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            result = analyze_reasoning("   ", jc)
        self.assertIsNone(result)
        mock_llm.assert_not_called()

    def test_missing_templates_returns_none_with_warning(self):
        jc = self._judge_config(seg_template="", metrics_template="")
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm, \
             patch("prompttestenv.logger.log_warning") as mock_warn:
            result = analyze_reasoning("some trace", jc)
        self.assertIsNone(result)
        mock_llm.assert_not_called()
        mock_warn.assert_called_once()

    def test_full_success_path(self):
        jc = self._judge_config()
        seg_json = json.dumps({
            "interpretation": "i", "planning": "p", "pure_reasoning": "r", "output_formulation": "o",
        })
        met_json = json.dumps({"alt_path": 2, "autocorrect": 1, "alignment_score": 9})
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.side_effect = [
                (seg_json, 0, 0, ""),
                (met_json, 0, 0, ""),
            ]
            result = analyze_reasoning("trace text", jc, candidate_response="resp")
        self.assertIsNotNone(result)
        self.assertEqual(result.interpretation, "i")
        self.assertEqual(result.alt_path, 2)
        self.assertEqual(result.alignment_score, 9)

    def test_segmentation_failure_returns_none(self):
        jc = self._judge_config()
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.side_effect = [("not valid json", 0, 0, "")]
            result = analyze_reasoning("trace text", jc)
        self.assertIsNone(result)

    def test_metrics_failure_after_successful_segmentation_returns_partial_stats(self):
        jc = self._judge_config()
        seg_json = json.dumps({
            "interpretation": "i", "planning": "p", "pure_reasoning": "r", "output_formulation": "o",
        })
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.side_effect = [
                (seg_json, 0, 0, ""),
                ("not valid json", 0, 0, ""),
            ]
            result = analyze_reasoning("trace text", jc)
        self.assertIsNotNone(result)
        self.assertEqual(result.interpretation, "i")
        self.assertEqual(result.alt_path, 0)  # metrics never populated


if __name__ == "__main__":
    unittest.main()
