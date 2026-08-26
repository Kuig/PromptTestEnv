from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prompttestenv.models import (
    REASONING_SCOPE_ALL,
    REASONING_SCOPE_NONE,
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    ReasoningStats,
    TestCaseResult,
)
from prompttestenv.reporting import (
    format_cost_per_point,
    format_thinking_value,
    generate_html_report,
    get_badge_class,
    md_to_html,
)


class TestGetBadgeClass(unittest.TestCase):
    def test_high_score(self):
        self.assertEqual(get_badge_class(8.0), "high-score")
        self.assertEqual(get_badge_class(10), "high-score")

    def test_mid_score(self):
        self.assertEqual(get_badge_class(5.0), "mid-score")
        self.assertEqual(get_badge_class(7.99), "mid-score")

    def test_low_score(self):
        self.assertEqual(get_badge_class(0), "low-score")
        self.assertEqual(get_badge_class(4.99), "low-score")

    def test_non_numeric_input_falls_back_to_low_score(self):
        # Documents the current bare-except behavior in get_badge_class():
        # any exception (including a bad cast here) yields "low-score".
        self.assertEqual(get_badge_class("not-a-number"), "low-score")
        self.assertEqual(get_badge_class(None), "low-score")


class TestFormatThinkingValue(unittest.TestCase):
    def test_bool_true(self):
        self.assertEqual(format_thinking_value(True), "ON")

    def test_bool_false(self):
        self.assertEqual(format_thinking_value(False), "OFF")

    def test_none(self):
        self.assertEqual(format_thinking_value(None), "OFF")

    def test_string_variants(self):
        self.assertEqual(format_thinking_value("true"), "ON")
        self.assertEqual(format_thinking_value("ON"), "ON")
        self.assertEqual(format_thinking_value("false"), "OFF")
        self.assertEqual(format_thinking_value("off"), "OFF")
        self.assertEqual(format_thinking_value("none"), "OFF")
        self.assertEqual(format_thinking_value(""), "OFF")
        self.assertEqual(format_thinking_value("default"), "DEFAULT")

    def test_other_string_uppercased(self):
        self.assertEqual(format_thinking_value("budget"), "BUDGET")

    def test_other_type_stringified_and_uppercased(self):
        self.assertEqual(format_thinking_value(42), "42")


class TestMdToHtml(unittest.TestCase):
    def test_code_fence(self):
        self.assertIn("<pre><code>print(1)</code></pre>", md_to_html("```print(1)```"))

    def test_inline_code(self):
        self.assertIn("<code>x</code>", md_to_html("`x`"))

    def test_bullet_list(self):
        html = md_to_html("- one\n- two")
        self.assertIn("<ul>", html)
        self.assertIn("<li>one</li>", html)
        self.assertIn("<li>two</li>", html)

    def test_headers(self):
        self.assertIn("<h1>Title</h1>", md_to_html("# Title"))
        self.assertIn("<h3>Sub</h3>", md_to_html("### Sub"))

    def test_image(self):
        self.assertIn('<img src="x.png" alt="alt">', md_to_html("![alt](x.png)"))

    def test_link(self):
        self.assertIn('<a href="http://x">text</a>', md_to_html("[text](http://x)"))

    def test_bold(self):
        self.assertIn("<strong>b</strong>", md_to_html("**b**"))

    def test_italic(self):
        self.assertIn("<em>i</em>", md_to_html("*i*"))

    def test_table(self):
        html = md_to_html("| A | B |\n|---|---|\n| 1 | 2 |")
        self.assertIn("<table", html)
        self.assertIn("<th>A</th>", html)
        self.assertIn("<td>1</td>", html)

    def test_html_special_chars_are_escaped(self):
        self.assertIn("&lt;script&gt;", md_to_html("<script>"))


class TestGenerateHtmlReport(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="Baseline", provider="google", model="gemini")]
        result = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        perf = CandidatePerformance()
        perf.scores.append(9.0)
        perf.times.append(1.2)
        perf.tokens.append(50)
        perf.best_output = "hello"
        perf.best_reason = "good"
        result.candidates_perf["Baseline"] = perf
        self.results = [result]
        self.judge_config = JudgeConfig()
        self.global_criteria = GlobalCriteria(mode="none")

    def test_writes_html_file_under_report_dir(self):
        html_file = generate_html_report(
            self.project_dir, self.results, self.candidates, "Plain verdict text.",
            self.global_criteria, self.judge_config, filename="test.html",
        )
        self.assertTrue(Path(html_file).exists())
        content = Path(html_file).read_text(encoding="utf-8")
        self.assertIn("Baseline", content)
        # score 9.0 -> get_badge_class() renders "high-score" as the badge's CSS class
        self.assertIn('class="score-val high-score"', content)

    def test_grouped_json_verdict_renders_details_blocks(self):
        grouped_verdict = json.dumps({
            "is_grouped": True,
            "groups": [{"group_name": "G1", "verdict": "Group verdict text"}],
            "global_verdict": "Overall verdict text",
        })
        html_file = generate_html_report(
            self.project_dir, self.results, self.candidates, grouped_verdict,
            self.global_criteria, self.judge_config, filename="grouped.html",
        )
        content = Path(html_file).read_text(encoding="utf-8")
        self.assertIn("Verdict for Group: G1", content)
        self.assertIn("Global Verdict", content)
        self.assertIn("Overall verdict text", content)



class TestFormatCostPerPoint(unittest.TestCase):
    def test_rounds_to_a_whole_number(self):
        self.assertEqual(format_cost_per_point(25.4), "25")

    def test_sentinel_renders_as_not_available(self):
        self.assertEqual(format_cost_per_point(-1.0), "N/A")

    def test_zero_is_a_real_value_not_a_sentinel(self):
        self.assertEqual(format_cost_per_point(0.0), "0")


class TestReportStatsSurviveTheTemplate(unittest.TestCase):
    """A stale key in report_template.html must not pass silently.

    Whether Jinja raises or renders nothing depends on how the value is used:
    a formatted figure blows up, a bare interpolation quietly renders empty.
    Only the second case is silent, and it is the one a "the file was written"
    assertion would miss, so these check the figures themselves.
    """

    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_test_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate(name="Baseline", provider="google", model="gemini")]
        row = TestCaseResult(test_id="t1", prompt="p", criteria="c")
        perf = CandidatePerformance()
        perf.scores.append(8.0)
        perf.global_scores.append(6.0)
        perf.times.append(1.25)
        perf.tokens.append(150)
        perf.reasoning_tokens.append(200)
        perf.best_output = "hello"
        perf.best_reason = "good"
        row.candidates_perf["Baseline"] = perf
        self.results = [row]
        self.judge_config = JudgeConfig()
        self.global_criteria = GlobalCriteria(mode="none")

    def _render(self) -> str:
        html_file = generate_html_report(
            self.project_dir, self.results, self.candidates, "Plain verdict text.",
            self.global_criteria, self.judge_config, filename="stats.html",
        )
        return Path(html_file).read_text(encoding="utf-8")

    def test_every_pooled_figure_reaches_the_html(self):
        content = self._render()
        for figure in ("8.00", "6.00", "1.25", "150", "200"):
            self.assertIn(figure, content, f"{figure} did not survive the template")

    def test_combined_average_reaches_the_header_badge(self):
        self.assertIn("7.0", self._render())

    def test_cost_per_point_is_shown_when_analysis_is_enabled(self):
        self.judge_config.reasoning_analysis = REASONING_SCOPE_ALL
        self.results[0].candidates_perf["Baseline"].reasoning_analyses.append(
            ReasoningStats(density=1.5).to_dict()
        )
        content = self._render()
        self.assertIn("lower is cheaper", content)
        self.assertIn("25/point", content, "200 thinking tokens at a score of 8 is 25 per point")

    def test_cost_per_point_is_shown_with_the_analysis_switched_off(self):
        """It needs thinking tokens, not the analysis phase.

        PaperReviewer runs with reasoning_analysis "none" and still bills over a
        thousand thinking tokens per generation; the figure used to reach the
        verdict judge but not the project's own report.
        """
        self.judge_config.reasoning_analysis = REASONING_SCOPE_NONE
        content = self._render()
        self.assertIn("25/point", content)

    def test_per_test_case_row_carries_its_own_mean_of_ratios_cost(self):
        """Distinct from the pooled STATS-row figure, and shown regardless of analysis.

        Two repetitions with the same tokens but very different scores: the
        pooled ratio-of-means (STATS row, Think/point) and the per-test-case
        mean-of-ratios (this row's own Cost figure) must diverge and both be
        visible in the same render, proving neither line is silently reusing
        the other's number.
        """
        self.judge_config.reasoning_analysis = REASONING_SCOPE_NONE
        perf = self.results[0].candidates_perf["Baseline"]
        perf.reasoning_tokens.clear()
        perf.scores.clear()
        perf.reasoning_tokens.extend([300, 300])
        perf.scores.extend([10.0, 1.0])
        content = self._render()
        self.assertIn("55/point", content, "STATS row: ratio of the pooled means (300/5.5, rounded)")
        self.assertIn("165.0", content, "per-test-case row: mean of each repetition's own ratio")

    def test_cost_per_point_is_omitted_without_thinking_tokens(self):
        self.results[0].candidates_perf["Baseline"].reasoning_tokens.clear()
        self.assertNotIn("/point", self._render())


if __name__ == "__main__":
    unittest.main()
