"""Tests for the JSON export (prompttestenv/json_report.py).

Its own file, mirroring the split of the module it covers: the exporter shares
no code with the HTML renderer, so neither do its tests.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv.models import (
    REASONING_SCOPE_ALL,
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    ReasoningStats,
    ReasoningUnit,
    TestCaseResult,
)
from prompttestenv.json_report import generate_json_report


class TestGenerateJsonReport(unittest.TestCase):
    """The JSON exporter is an alternative renderer, not a second aggregation.

    So these check two things the HTML tests cannot: that the figures it emits
    are the ones pool_by_candidate produced, and that it carries the detail the
    page drops — the per-repetition values.
    """

    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_json_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [
            Candidate(
                name="Baseline", provider="google", model="gemini",
                system_prompt_file="base.txt",
                resolved_system_instruction="THE RESOLVED PROMPT BODY",
            ),
        ]
        row = TestCaseResult(
            test_id="t1", prompt="p", criteria="c", group="G1",
            files_used=["test_files/a.txt"],
        )
        self.perf = CandidatePerformance()
        self.perf.scores.extend([8.0, 6.0])
        self.perf.global_scores.extend([7.0, 5.0])
        self.perf.times.extend([1.25, 2.5])
        self.perf.tokens.extend([150, 250])
        self.perf.reasoning_tokens.extend([200, 400])
        self.perf.best_output = "hello"
        self.perf.best_reason = "good"
        self.perf.best_global_reason = "fine"
        row.candidates_perf["Baseline"] = self.perf
        self.results = [row]
        self.judge_config = JudgeConfig()
        self.global_criteria = GlobalCriteria(mode="none")

    def _export(self, verdict: str = "Plain verdict text.", name: str = "r.json") -> dict:
        path = generate_json_report(
            self.project_dir, self.results, self.candidates, verdict,
            self.global_criteria, self.judge_config, filename=name,
        )
        self.assertTrue(Path(path).exists())
        self.assertEqual(Path(path).parent.name, "Report")
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_writes_parsable_json_under_report_dir(self):
        payload = self._export()
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertIn("Baseline", payload["aggregate"])
        self.assertEqual([t["test_id"] for t in payload["test_cases"]], ["t1"])

    def test_pooled_figures_match_the_record_they_describe(self):
        agg = self._export()["aggregate"]["Baseline"]
        self.assertAlmostEqual(agg["score"]["mean"], 7.0)
        self.assertAlmostEqual(agg["global_score"]["mean"], 6.0)
        self.assertAlmostEqual(agg["tokens"]["mean"], 200.0)
        self.assertAlmostEqual(agg["time"]["mean"], 1.875)
        self.assertAlmostEqual(agg["combined_avg"], 6.5)

    def test_every_repetition_survives_as_a_raw_value(self):
        """The one thing the HTML cannot show: the distribution behind the mean."""
        agg = self._export()["aggregate"]["Baseline"]
        self.assertEqual(agg["score"]["values"], [8.0, 6.0])
        self.assertEqual(agg["tokens"]["values"], [150, 250])
        self.assertEqual(agg["reasoning_tokens"]["values"], [200, 400])
        self.assertEqual(agg["time"]["values"], [1.25, 2.5])

    def test_both_cost_per_point_statistics_are_emitted(self):
        """They are different statistics and need not agree; the report shows both."""
        cost = self._export()["aggregate"]["Baseline"]["cost_per_point"]
        # Ratio of the means: 300 thinking tokens over a score of 7.
        self.assertAlmostEqual(cost["pooled"], 300 / 7)
        # Mean of each repetition's own ratio: (200/8 + 400/6) / 2.
        self.assertAlmostEqual(cost["mean"], (200 / 8 + 400 / 6) / 2)
        self.assertNotAlmostEqual(cost["pooled"], cost["mean"])

    def test_not_measured_stays_the_sentinel_rather_than_becoming_null(self):
        """progress.jsonl writes -1; the report must not invent a second convention."""
        self.perf.scores.clear()
        self.perf.scores.extend([-1.0, -1.0])
        agg = self._export(name="sentinel.json")["aggregate"]["Baseline"]
        self.assertEqual(agg["score"]["values"], [-1.0, -1.0])
        self.assertEqual(agg["score"]["mean"], -1.0)
        self.assertEqual(agg["combined_avg"], 6.0, "the measured global side must still count")

    def test_resolved_system_instruction_is_never_emitted(self):
        """It is derived and can be huge — candidates.json does not carry it either."""
        payload = self._export()
        cand = payload["configuration"]["candidates"][0]
        self.assertEqual(cand["system_prompt_file"], "base.txt")
        self.assertNotIn("resolved_system_instruction", cand)
        self.assertNotIn("THE RESOLVED PROMPT BODY", json.dumps(payload))

    def test_plain_verdict_is_reported_as_ungrouped(self):
        verdict = self._export()["verdict"]
        self.assertFalse(verdict["grouped"])
        self.assertEqual(verdict["groups"], [])
        self.assertEqual(verdict["global_verdict"], "Plain verdict text.")
        self.assertEqual(verdict["text"], "Plain verdict text.")

    def test_grouped_verdict_is_split_into_its_groups(self):
        grouped = json.dumps({
            "is_grouped": True,
            "groups": [{"group_name": "G1", "verdict": "Group body."}],
            "global_verdict": "Overall body.",
        })
        verdict = self._export(verdict=grouped, name="grouped.json")["verdict"]
        self.assertTrue(verdict["grouped"])
        self.assertEqual(verdict["groups"], [{"group_name": "G1", "verdict": "Group body."}])
        self.assertEqual(verdict["global_verdict"], "Overall body.")
        self.assertEqual(verdict["text"], grouped, "the raw form must survive for the log")

    def test_active_global_criteria_is_resolved_for_the_mode_in_use(self):
        self.global_criteria = GlobalCriteria(
            mode="similarity", llm_judge_criteria="unused rubric",
            similarity_criteria="the target answer",
        )
        block = self._export(name="criteria.json")["configuration"]["global_criteria"]
        self.assertEqual(block["mode"], "similarity")
        self.assertEqual(block["criteria"], "the target answer")

    def test_reasoning_analysis_is_null_when_the_trace_was_not_analysed(self):
        best = self._export()["test_cases"][0]["candidates"]["Baseline"]["best"]
        self.assertIsNone(best["reasoning_analysis"])
        self.assertEqual(best["reasoning_text"], "")

    def test_reasoning_units_are_offsets_into_the_emitted_trace(self):
        """The units are offsets, so the trace must travel with them or they are unresolvable."""
        self.judge_config.reasoning_analysis = REASONING_SCOPE_ALL
        analysis = ReasoningStats(
            units=[ReasoningUnit(start=0, end=5, framing=3.0)],
            coverage_framing=1.0, density=1.0, schema_stamp="stamp@1",
        )
        self.perf.best_reasoning_text = "first second"
        self.perf.best_reasoning_analysis = analysis
        self.perf.reasoning_analyses.append(analysis.to_dict())

        payload = self._export(name="reasoning.json")
        best = payload["test_cases"][0]["candidates"]["Baseline"]["best"]
        unit = best["reasoning_analysis"]["units"][0]
        self.assertEqual(best["reasoning_text"][unit["start"]:unit["end"]], "first")
        self.assertEqual(payload["aggregate"]["Baseline"]["reasoning_profile"]["n"], 1)

    def test_reasoning_profile_is_empty_when_nothing_was_analysed(self):
        """{} is "nothing to pool", which a consumer must not read as zero coverage."""
        self.assertEqual(self._export()["aggregate"]["Baseline"]["reasoning_profile"], {})

    def test_candidate_with_no_repetition_is_absent_rather_than_zeroed(self):
        self.candidates.append(Candidate(name="Ghost", provider="ollama", model="gemma"))
        payload = self._export(name="ghost.json")
        self.assertNotIn("Ghost", payload["test_cases"][0]["candidates"])
        self.assertIn("Ghost", payload["aggregate"], "the pooled view still lists it")
        self.assertEqual(payload["aggregate"]["Ghost"]["score"]["values"], [])


if __name__ == "__main__":
    unittest.main()
