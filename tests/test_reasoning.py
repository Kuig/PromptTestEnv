from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch

from prompttestenv.config import (
    AppConfig,
    ReasoningDefaults,
    ReasoningDimension,
    ReasoningSchema,
    UnitSplittingConfig,
)
from prompttestenv.models import (
    REASONING_SCOPE_ALL,
    REASONING_SCOPE_NONE,
    JudgeConfig,
    LlmResult,
    ReasoningStats,
    ReasoningUnit,
)
from prompttestenv.config import get_app_config
from prompttestenv.reasoning import (
    _example_json,
    _format_template,
    aggregate_reasoning_stats,
    analyze_reasoning,
    compute_coverages,
    compute_repetition_rate,
    split_into_units,
)

SPLITTING = UnitSplittingConfig(min_unit_chars=15, abbreviations=["e.g.", "i.e.", "Dr."])


def make_app_config() -> AppConfig:
    """Build an AppConfig with a minimal but complete reasoning schema."""
    return AppConfig(
        reasoning_schema=ReasoningSchema(
            system_prompt="analyst",
            intensity_scale=3,
            dimensions=[
                ReasoningDimension(name="framing", color="#111111", definition="frame"),
                ReasoningDimension(name="solving", color="#222222", definition="solve"),
                ReasoningDimension(name="presentation", color="#333333", definition="present"),
            ],
            dimension_template=(
                "{dimension_name}|{dimension_definition}|{intensity_scale}|"
                "{example_json}|{user_prompt}|{criteria}|{numbered_units}"
            ),
            joint_template=(
                "{dimension_count}|{dimension_definitions}|{intensity_scale}|"
                "{example_json}|{user_prompt}|{criteria}|{numbered_units}"
            ),
            metrics_template="{user_prompt}|{numbered_units}|{candidate_response}",
        ),
        unit_splitting=SPLITTING,
        reasoning_defaults=ReasoningDefaults(),
        local_providers=["ollama"],
    )


class TestSplitIntoUnits(unittest.TestCase):
    """The split is what guarantees full coverage, so its invariants are the contract."""

    def assert_invariants(self, text: str) -> list[tuple[int, int]]:
        spans = split_into_units(text, SPLITTING)
        for start, end in spans:
            self.assertLess(start, end, "spans must be non-empty")
            self.assertFalse(text[start].isspace(), "spans must start on non-whitespace")
            self.assertFalse(text[end - 1].isspace(), "spans must end on non-whitespace")
        for (_, end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, next_start, "spans must not overlap")
        covered = {i for start, end in spans for i in range(start, end)}
        dropped = set(range(len(text))) - covered
        self.assertTrue(
            all(text[i].isspace() for i in dropped),
            "only whitespace may fall outside a unit",
        )
        return spans

    def test_covers_every_non_whitespace_character(self):
        text = "First thought here. Second thought follows.\n\nA third one, in its own paragraph."
        spans = self.assert_invariants(text)
        self.assertEqual(len(spans), 3)

    def test_abbreviation_does_not_end_a_sentence(self):
        text = "The model considered options, e.g. rewriting the draft entirely, before choosing."
        spans = self.assert_invariants(text)
        self.assertEqual(len(spans), 1)

    def test_code_block_stays_atomic(self):
        text = "Here is the plan for the code.\n\n```\nx = 1. y = 2. z = 3.\n```\n\nThat should do it."
        spans = self.assert_invariants(text)
        block = next(text[a:b] for a, b in spans if "x = 1" in text[a:b])
        self.assertIn("z = 3.", block, "a fenced block must not be split on its punctuation")

    def test_heading_line_is_kept_not_dropped(self):
        text = "**My Approach**\n\nI need to answer the question that was actually asked here."
        spans = self.assert_invariants(text)
        self.assertIn("**My Approach**", text[spans[0][0]:spans[0][1]])

    def test_short_fragment_is_merged_into_its_neighbour(self):
        text = "Right. Now I will work through the problem one careful step at a time."
        spans = self.assert_invariants(text)
        self.assertEqual(len(spans), 1, "a 6-character fragment is not a scorable unit")

    def test_empty_text_yields_no_units(self):
        self.assertEqual(split_into_units("   \n\n  ", SPLITTING), [])


class TestComputeCoverages(unittest.TestCase):
    def test_coverage_is_length_weighted_and_density_is_their_sum(self):
        schema = make_app_config().reasoning_schema
        stats = ReasoningStats(units=[
            ReasoningUnit(start=0, end=10, framing=3.0, solving=0.0, presentation=0.0),
            ReasoningUnit(start=10, end=30, framing=0.0, solving=3.0, presentation=1.5),
        ])
        for dimension in ("framing", "solving", "presentation"):
            stats.set_coverage(dimension, 0.0)
        compute_coverages(stats, schema)
        # framing occupies 10 of 30 characters at full intensity.
        self.assertAlmostEqual(stats.coverage("framing"), 10 / 30, places=3)
        self.assertAlmostEqual(stats.coverage("solving"), 20 / 30, places=3)
        self.assertAlmostEqual(stats.coverage("presentation"), (20 * 0.5) / 30, places=3)
        self.assertAlmostEqual(stats.density, 10 / 30 + 20 / 30 + 10 / 30, places=3)

    def test_coverages_may_exceed_one_in_total(self):
        """Dimensions are independent, so a sentence doing three things scores three times."""
        schema = make_app_config().reasoning_schema
        stats = ReasoningStats(units=[
            ReasoningUnit(start=0, end=10, framing=3.0, solving=3.0, presentation=3.0)
        ])
        for dimension in ("framing", "solving", "presentation"):
            stats.set_coverage(dimension, 0.0)
        compute_coverages(stats, schema)
        self.assertAlmostEqual(stats.density, 3.0, places=3)

    def test_unmeasured_dimension_is_left_at_the_sentinel(self):
        schema = make_app_config().reasoning_schema
        stats = ReasoningStats(units=[ReasoningUnit(start=0, end=10, solving=3.0)])
        stats.set_coverage("framing", -1.0)
        stats.set_coverage("solving", 0.0)
        stats.set_coverage("presentation", 0.0)
        compute_coverages(stats, schema)
        self.assertEqual(stats.coverage("framing"), -1.0)
        self.assertAlmostEqual(stats.density, 1.0, places=3)


class TestComputeRepetitionRate(unittest.TestCase):
    def test_short_text_is_not_measured(self):
        self.assertEqual(compute_repetition_rate("too short to judge"), -1.0)

    def test_repeated_text_scores_higher_than_varied_text(self):
        looping = "let me check that again " * 10
        varied = " ".join(f"distinct clause number {i} here" for i in range(10))
        self.assertGreater(compute_repetition_rate(looping), compute_repetition_rate(varied))


class TestAggregateReasoningStats(unittest.TestCase):
    def test_empty_list_returns_empty_dict(self):
        self.assertEqual(aggregate_reasoning_stats([]), {})

    def test_unmeasured_values_are_excluded_rather_than_averaged_as_zero(self):
        analyses = [
            ReasoningStats(alt_path=4).to_dict(),
            ReasoningStats(alt_path=-1).to_dict(),
        ]
        result = aggregate_reasoning_stats(analyses)
        self.assertEqual(result["avg_alt_path"], 4.0, "the -1 sentinel must not drag the mean down")
        self.assertEqual(result["n"], 2)

    def test_all_unmeasured_stays_unmeasured(self):
        result = aggregate_reasoning_stats([ReasoningStats().to_dict()])
        self.assertEqual(result["avg_alignment_score"], -1.0)

    def test_summary_flag_and_schema_stamps_are_surfaced(self):
        analyses = [
            ReasoningStats(reasoning_is_summary=False, schema_stamp="a@1").to_dict(),
            ReasoningStats(reasoning_is_summary=True, schema_stamp="b@2").to_dict(),
        ]
        result = aggregate_reasoning_stats(analyses)
        self.assertTrue(result["is_summary"])
        self.assertEqual(result["schema_stamps"], ["a@1", "b@2"])


class TestAnalyzeReasoning(unittest.TestCase):
    def setUp(self):
        self.app_config = make_app_config()
        patcher = patch("prompttestenv.reasoning.get_app_config", return_value=self.app_config)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.judge_config = JudgeConfig()
        self.judge_config.reasoning_analysis = REASONING_SCOPE_ALL
        self.judge_config.reasoning_judge.provider = "google"

        drift = patch("prompttestenv.reasoning.compute_trace_response_drift", return_value=-1.0)
        drift.start()
        self.addCleanup(drift.stop)

    TRACE = (
        "I need to work out what the user is actually asking for here.\n\n"
        "The answer is forty-two, which follows from the figures given above.\n\n"
        "I will present it as a single short sentence with no preamble."
    )

    def scores_for(self, count: int, value: int) -> str:
        return json.dumps({"scores": {str(i + 1): value for i in range(count)}})

    def responder(self, by_dimension: dict[str, str], metrics: str = "{}"):
        """Answer each judge call according to which dimension it asks about.

        In split mode the three dimension calls run concurrently, so a plain
        side_effect list would be consumed in thread-completion order and bind
        answers to the wrong dimensions. Dispatching on the prompt also proves
        each dimension gets its own single-concept question.
        """
        def _respond(*_args, **kwargs):
            prompt = kwargs.get("user_prompt", "")
            for dimension, payload in by_dimension.items():
                if prompt.startswith(dimension):
                    return LlmResult(text=payload)
            return LlmResult(text=metrics)

        return _respond

    def test_disabled_analysis_returns_none_without_calling_llm(self):
        self.judge_config.reasoning_analysis = REASONING_SCOPE_NONE
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            self.assertIsNone(analyze_reasoning(self.TRACE, self.judge_config))
        mock_llm.assert_not_called()

    def test_empty_trace_returns_none_without_calling_llm(self):
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            self.assertIsNone(analyze_reasoning("   ", self.judge_config))
        mock_llm.assert_not_called()

    def test_mismatched_schema_is_refused(self):
        self.app_config.reasoning_schema.dimensions.append(
            ReasoningDimension(name="verification", definition="check")
        )
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm, \
             patch("prompttestenv.logger.log_error") as mock_error:
            self.assertIsNone(analyze_reasoning(self.TRACE, self.judge_config))
        mock_llm.assert_not_called()
        mock_error.assert_called_once()

    def test_split_mode_makes_one_call_per_dimension_plus_metrics(self):
        units = len(split_into_units(self.TRACE, SPLITTING))
        metrics = json.dumps({
            "alt_path_units": [1], "autocorrect_units": [], "alignment_score": 9,
            "alignment_evidence": [],
        })
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.side_effect = self.responder(
                {
                    "framing": self.scores_for(units, 3),
                    "solving": self.scores_for(units, 0),
                    "presentation": self.scores_for(units, 0),
                },
                metrics=metrics,
            )
            stats = analyze_reasoning(self.TRACE, self.judge_config, candidate_response="42")

        self.assertEqual(mock_llm.call_count, 4, "3 dimensions + 1 metrics call")
        self.assertAlmostEqual(stats.coverage("framing"), 1.0, places=3)
        self.assertAlmostEqual(stats.coverage("solving"), 0.0, places=3)
        self.assertEqual(stats.alt_path, 1)
        self.assertEqual(stats.alt_path_units, [1])
        self.assertEqual(stats.alignment_score, 9)
        self.assertEqual(len(stats.units), units)

    def test_joint_mode_makes_a_single_scoring_call(self):
        self.judge_config.reasoning_judge.dimension_mode = "joint"
        units = len(split_into_units(self.TRACE, SPLITTING))
        joint = json.dumps({
            "scores": {
                str(i + 1): {"framing": 3, "solving": 1, "presentation": 0}
                for i in range(units)
            }
        })
        metrics = json.dumps({"alt_path_units": [], "autocorrect_units": [], "alignment_score": 7})
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.side_effect = [LlmResult(text=joint), LlmResult(text=metrics)]
            stats = analyze_reasoning(self.TRACE, self.judge_config)

        self.assertEqual(mock_llm.call_count, 2, "1 joint call + 1 metrics call")
        self.assertAlmostEqual(stats.coverage("framing"), 1.0, places=3)
        self.assertAlmostEqual(stats.coverage("solving"), 1 / 3, places=3)

    def overlapping_calls(self) -> bool:
        """Run an analysis and report whether any two judge calls overlapped in time.

        Measured on the calls themselves rather than on ThreadPoolExecutor:
        call_with_timeout wraps every single call in a pool of its own, so
        patching the executor proves nothing about the dimension fan-out (and,
        because the patch lands on the shared concurrent.futures module, it also
        breaks that wrapper).
        """
        units = len(split_into_units(self.TRACE, SPLITTING))
        payload = self.scores_for(units, 1)
        intervals: list[tuple[float, float]] = []
        lock = threading.Lock()

        def _respond(*_args, **_kwargs):
            started = time.perf_counter()
            time.sleep(0.05)
            with lock:
                intervals.append((started, time.perf_counter()))
            return LlmResult(text=payload)

        with patch("prompttestenv.reasoning.get_llm_response", side_effect=_respond):
            analyze_reasoning(self.TRACE, self.judge_config)

        intervals.sort()
        return any(
            end > next_start
            for (_, end), (next_start, _) in zip(intervals, intervals[1:])
        )

    def test_local_judge_scores_dimensions_sequentially(self):
        """A local backend serves one model at a time, so the dimension calls must not overlap."""
        self.judge_config.reasoning_judge.provider = "ollama"
        self.assertFalse(
            self.overlapping_calls(),
            "concurrent requests to a local model queue up or force a reload",
        )

    def test_remote_judge_scores_dimensions_concurrently(self):
        self.judge_config.reasoning_judge.provider = "google"
        self.assertTrue(
            self.overlapping_calls(),
            "remote dimension calls should run in parallel",
        )

    def test_missing_ids_count_as_zero_not_as_a_failure(self):
        units = len(split_into_units(self.TRACE, SPLITTING))
        partial = json.dumps({"scores": {"1": 3}})
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=partial)
            stats = analyze_reasoning(self.TRACE, self.judge_config)
        self.assertGreaterEqual(stats.coverage("framing"), 0.0)
        self.assertEqual(len(stats.units), units)
        self.assertEqual(stats.units[-1].intensity("framing"), 0.0)

    def test_out_of_range_ids_are_ignored(self):
        payload = json.dumps({"scores": {"1": 3, "999": 3, "not-a-number": 3}})
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=payload)
            stats = analyze_reasoning(self.TRACE, self.judge_config)
        self.assertIsNotNone(stats)
        self.assertEqual(stats.units[0].intensity("framing"), 3.0)

    def test_failed_dimension_call_is_recorded_as_not_measured(self):
        units = len(split_into_units(self.TRACE, SPLITTING))
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm, \
             patch("prompttestenv.logger.log_error"):
            mock_llm.side_effect = self.responder(
                {
                    "framing": "not json",
                    "solving": self.scores_for(units, 3),
                    "presentation": self.scores_for(units, 0),
                },
                metrics=json.dumps({"alignment_score": 8}),
            )
            stats = analyze_reasoning(self.TRACE, self.judge_config)
        self.assertEqual(stats.coverage("framing"), -1.0, "a failed call is not a measured 0")
        self.assertAlmostEqual(stats.coverage("solving"), 1.0, places=3)

    def test_failed_metrics_call_keeps_the_segmentation(self):
        units = len(split_into_units(self.TRACE, SPLITTING))
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm, \
             patch("prompttestenv.logger.log_error"):
            mock_llm.side_effect = self.responder(
                {
                    "framing": self.scores_for(units, 2),
                    "solving": self.scores_for(units, 2),
                    "presentation": self.scores_for(units, 2),
                },
                metrics="not json",
            )
            stats = analyze_reasoning(self.TRACE, self.judge_config)
        self.assertGreater(stats.coverage("framing"), 0.0)
        self.assertEqual(stats.alt_path, -1)
        self.assertEqual(stats.alignment_score, -1)

    def test_judge_timeout_is_recorded_as_not_measured(self):
        with patch("prompttestenv.api.call_with_timeout", return_value=(None, True)), \
             patch("prompttestenv.logger.log_warning"):
            stats = analyze_reasoning(self.TRACE, self.judge_config)
        self.assertEqual(stats.coverage("framing"), -1.0)
        self.assertEqual(stats.alignment_score, -1)

    def test_chunking_splits_long_traces_and_merges_the_results(self):
        self.judge_config.reasoning_judge.max_units_per_call = 1
        units = len(split_into_units(self.TRACE, SPLITTING))
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=self.scores_for(units, 3))
            stats = analyze_reasoning(self.TRACE, self.judge_config)
        self.assertEqual(mock_llm.call_count, units * 3 + 1, "3 dimension calls per window, plus metrics")
        self.assertAlmostEqual(stats.coverage("framing"), 1.0, places=3)

    def test_schema_stamp_and_summary_flag_are_stored(self):
        units = len(split_into_units(self.TRACE, SPLITTING))
        with patch("prompttestenv.reasoning.get_llm_response") as mock_llm:
            mock_llm.return_value = LlmResult(text=self.scores_for(units, 1))
            stats = analyze_reasoning(self.TRACE, self.judge_config, reasoning_is_summary=True)
        self.assertTrue(stats.reasoning_is_summary)
        self.assertEqual(stats.schema_stamp, self.app_config.reasoning_schema.stamp)



class TestShippedPromptTemplates(unittest.TestCase):
    """The real templates in config.json, not the simplified ones the other tests use.

    Added after the joint template shipped with six closing braces where it needed
    seven: str.format raised ValueError, so joint mode could not run at all, and
    every test passed because they all used a stand-in template.
    """

    def setUp(self):
        self.schema = get_app_config(reload=True).reasoning_schema

    def test_dimension_template_formats(self):
        dimension = self.schema.dimensions[0]
        prompt = _format_template(
            self.schema.dimension_template,
            dimension_name=dimension.name,
            dimension_definition=dimension.definition,
            intensity_scale=self.schema.intensity_scale,
            example_json=_example_json(self.schema, joint=False),
            user_prompt="the task",
            criteria="the criteria",
            numbered_units="[1] a sentence",
        )
        self.assertTrue(prompt, "a malformed template renders as the empty string")
        self.assertIn(dimension.name, prompt)
        self.assertIn("[1] a sentence", prompt)
        self.assertIn("the task", prompt)

    def test_joint_template_formats(self):
        prompt = _format_template(
            self.schema.joint_template,
            dimension_count=len(self.schema.dimensions),
            dimension_definitions="defs",
            intensity_scale=self.schema.intensity_scale,
            example_json=_example_json(self.schema, joint=True),
            user_prompt="the task",
            criteria="the criteria",
            numbered_units="[1] a sentence",
        )
        self.assertTrue(prompt, "a malformed template renders as the empty string")
        self.assertIn("[1] a sentence", prompt)

    def test_metrics_template_formats(self):
        prompt = _format_template(
            self.schema.metrics_template,
            user_prompt="the task",
            numbered_units="[1] a sentence",
            candidate_response="the answer",
        )
        self.assertTrue(prompt, "a malformed template renders as the empty string")
        self.assertIn("the answer", prompt)

    def test_example_answers_are_valid_json_in_the_expected_shape(self):
        """The judge is asked to imitate these, so they must be parseable and correct."""
        split = json.loads(_example_json(self.schema, joint=False))
        self.assertEqual(set(split), {"scores"})
        self.assertTrue(all(isinstance(v, int) for v in split["scores"].values()))

        joint = json.loads(_example_json(self.schema, joint=True))
        self.assertEqual(set(joint), {"scores"})
        for per_unit in joint["scores"].values():
            self.assertEqual(set(per_unit), set(self.schema.dimension_names))

    def test_malformed_template_degrades_instead_of_raising(self):
        with patch("prompttestenv.logger.log_error") as mock_error:
            self.assertEqual(_format_template("unbalanced }", user_prompt="x"), "")
            self.assertEqual(_format_template("{no_such_placeholder}"), "")
        self.assertEqual(mock_error.call_count, 2)


if __name__ == "__main__":
    unittest.main()
