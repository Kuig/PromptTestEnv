"""Unit tests for prompttestenv/frontier_plot.py and the report's frontier sections."""
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from prompttestenv import frontier as fr
from prompttestenv import frontier_plot as fp
from prompttestenv.models import (
    Candidate,
    CandidatePerformance,
    GlobalCriteria,
    JudgeConfig,
    TestCaseResult,
)
from prompttestenv.reporting import generate_html_report

SVG_NS = "{http://www.w3.org/2000/svg}"


def _panel(points, roles=None, **options):
    return fr.Panel(fr.PANEL_CLUSTER, "Cluster <A&B>", ["G1", "G2"],
                    fr.ResolvedOptions(**options), points,
                    roles=roles or [fr.CANDIDATE_ROLE_CANDIDATE])


def _parse(svg: str) -> ET.Element:
    # The markup carries no xmlns: inline SVG in HTML does not need one.
    return ET.fromstring(svg.replace("<svg ", f'<svg xmlns="{SVG_NS[1:-1]}" ', 1))


class TestRenderPanelSvg(unittest.TestCase):
    def setUp(self):
        self.points = [
            fr.PanelPoint("fast <bad>", 2.0, 5.0, 2, on_front=True),
            fr.PanelPoint("slow & good", 200.0, 9.5, 2, on_front=True),
            fr.PanelPoint("dominated", 100.0, 4.0, 1),
        ]
        # The same, plus a second role whose single point is its own frontier.
        self.two_roles = [*self.points,
                          fr.PanelPoint("ref", 1.0, 9.0, 2, role="baseline", on_front=True)]

    def test_is_well_formed_and_escapes_names(self):
        svg = fp.render_panel_svg(_panel(self.points))
        root = _parse(svg)
        texts = [t.text for t in root.iter(f"{SVG_NS}text")]
        self.assertIn("fast <bad>", texts)
        self.assertIn("slow & good", texts)
        self.assertIn("dominated (1/2)", texts)

    def test_frontier_line_uses_one_gradient_per_segment(self):
        root = _parse(fp.render_panel_svg(_panel(self.points)))
        self.assertEqual(len(list(root.iter(f"{SVG_NS}linearGradient"))), 1)
        dashed = [l for l in root.iter(f"{SVG_NS}line") if l.get("stroke-dasharray")]
        self.assertEqual(len(dashed), 1)

    def test_a_second_role_gets_the_next_shape(self):
        panel = _panel(self.two_roles, roles=["candidate", "baseline"])
        root = _parse(fp.render_panel_svg(panel))
        self.assertEqual(len(list(root.iter(f"{SVG_NS}circle"))), 3)
        squares = list(root.iter(f"{SVG_NS}rect"))
        # The background is a rect too, and the "ref" marker is the other one.
        self.assertEqual(len([r for r in squares if r.get("width") != str(fp.WIDTH)]), 1)
        self.assertEqual(fp.shape_of(panel, "baseline"), "square")

    def test_shapes_cycle_after_four_roles(self):
        roles = ["a", "b", "c", "d", "e"]
        panel = _panel([], roles=roles)
        self.assertEqual([fp.shape_of(panel, r) for r in roles],
                         ["circle", "square", "triangle", "diamond", "circle"])

    def test_one_dashed_line_per_role(self):
        two = [*self.two_roles, fr.PanelPoint("ref2", 50.0, 9.9, 2, role="baseline", on_front=True)]
        root = _parse(fp.render_panel_svg(_panel(two, roles=["candidate", "baseline"])))
        dashed = [l for l in root.iter(f"{SVG_NS}line") if l.get("stroke-dasharray")]
        self.assertEqual(len(dashed), 2)
        ids = {g.get("id") for g in root.iter(f"{SVG_NS}linearGradient")}
        self.assertEqual(len(ids), 2)

    def test_the_role_reaches_the_tooltip_only_with_several_roles(self):
        single = fp.render_panel_svg(_panel(self.points))
        multi = fp.render_panel_svg(_panel(self.two_roles, roles=["candidate", "baseline"]))
        self.assertNotIn("[candidate]", single)
        self.assertIn("ref [baseline]", multi)

    def test_zero_band_for_zero_thinking(self):
        points = [fr.PanelPoint("none", 0.0, 6.0, 1, on_front=True),
                  fr.PanelPoint("some", 500.0, 8.0, 1, on_front=True)]
        root = _parse(fp.render_panel_svg(_panel(points, x_axis=fr.X_AXIS_THINK_TOKENS)))
        self.assertIn("0", [t.text for t in root.iter(f"{SVG_NS}text")])

    def test_gradient_ids_differ_between_panels(self):
        a = fp.render_panel_svg(fr.Panel(fr.PANEL_GROUP, "A", ["A"], fr.ResolvedOptions(), self.points))
        b = fp.render_panel_svg(fr.Panel(fr.PANEL_GROUP, "B", ["B"], fr.ResolvedOptions(), self.points))
        id_a = _parse(a).find(f".//{SVG_NS}linearGradient").get("id")
        id_b = _parse(b).find(f".//{SVG_NS}linearGradient").get("id")
        self.assertNotEqual(id_a, id_b)

    def test_single_point_and_empty_panel(self):
        _parse(fp.render_panel_svg(_panel([fr.PanelPoint("only", 3.0, 7.0, 1, on_front=True)])))
        self.assertIn("No measurable data", fp.render_panel_svg(_panel([])))


class TestRenderPanelHtml(unittest.TestCase):
    def test_table_lists_after_threshold(self):
        points = [fr.PanelPoint("lo", 1.0, 4.0, 2, on_front=True),
                  fr.PanelPoint("hi", 9.0, 8.0, 2, on_front=True)]
        html_text = fp.render_panel_html(_panel(points, threshold=6))
        table = html_text[html_text.index("frontier-table"):]
        self.assertIn("<td>hi</td>", table)
        self.assertNotIn("<td>lo</td>", table)
        self.assertIn("Cluster &lt;A&amp;B&gt;", html_text)

    def test_one_table_per_role_titled_only_with_several_roles(self):
        points = [fr.PanelPoint("a", 1.0, 8.0, 2, on_front=True),
                  fr.PanelPoint("b", 2.0, 9.0, 2, role="baseline", on_front=True)]
        single = fp.render_panel_html(_panel(points[:1]))
        multi = fp.render_panel_html(_panel(points, roles=["candidate", "baseline"]))
        self.assertEqual(single.count('class="frontier-table"'), 1)
        self.assertNotIn("Role:", single)
        self.assertNotIn("frontier-shape", single)
        self.assertEqual(multi.count('class="frontier-table"'), 2)
        self.assertIn("Role: candidate", multi)
        self.assertIn("Role: baseline", multi)
        self.assertEqual(multi.count('class="frontier-shape"'), 2)

    def test_nothing_passes_the_threshold(self):
        points = [fr.PanelPoint("lo", 1.0, 4.0, 2, on_front=True)]
        self.assertIn("No frontier candidate", fp.render_panel_html(_panel(points, threshold=6)))

    def test_axis_value_formats(self):
        self.assertEqual(fp.format_axis_value(12.345, fr.X_AXIS_TIME), "12.35s")
        self.assertEqual(fp.format_axis_value(12345.4, fr.X_AXIS_THINK_TOKENS), "12,345")


class TestHtmlReportSections(unittest.TestCase):
    def setUp(self):
        self.project_dir = tempfile.mkdtemp(prefix="prompttestenv_frontier_html_")
        self.addCleanup(shutil.rmtree, self.project_dir, ignore_errors=True)
        self.candidates = [Candidate("A", "ollama", "m"), Candidate("Ref", "ollama", "m", role="baseline")]
        self.rows = []
        for test_id, group in (("t1", "Alpha"), ("t2", "Beta")):
            row = TestCaseResult(test_id=test_id, prompt="p", criteria="c", group=group)
            for name in ("A", "Ref"):
                row.candidates_perf[name] = CandidatePerformance(
                    scores=[8.0], global_scores=[10.0], times=[2.0], tokens=[10], reasoning_tokens=[0],
                )
            self.rows.append(row)

    def _render(self, verdict, frontiers=None):
        path = generate_html_report(
            self.project_dir, self.rows, self.candidates, verdict,
            GlobalCriteria(mode="none"), JudgeConfig(), frontiers=frontiers,
        )
        return Path(path).read_text(encoding="utf-8")

    def test_grouped_verdict_gets_one_panel_per_group_plus_all_tests(self):
        verdict = json.dumps({
            "is_grouped": True,
            "groups": [{"group_name": "Alpha", "verdict": "a"}, {"group_name": "Beta", "verdict": "b"}],
            "global_verdict": "g",
        })
        page = self._render(verdict)
        self.assertEqual(page.count('class="frontier-figure"'), 3)
        self.assertIn(fr.OVERALL_PANEL_NAME, page)
        self.assertNotIn("Efficiency frontiers by cluster", page)
        self.assertIn('class="role-badge">baseline', page)
        # Two roles: each panel gets a table per role.
        self.assertIn("Role: baseline", page)

    def test_plain_verdict_folds_the_group_panels(self):
        page = self._render("plain")
        self.assertIn("Efficiency frontier by group", page)
        self.assertEqual(page.count('class="frontier-figure"'), 3)

    def test_clusters_section_appears_with_a_cluster(self):
        frontiers = fr.build_panels(self.candidates, self.rows, fr.GroupClusters(
            clusters=[fr.ClusterSpec("Both", ["Alpha", "Beta"])],
        ))
        page = self._render("plain", frontiers)
        self.assertIn("Efficiency frontiers by cluster", page)
        self.assertEqual(page.count('class="frontier-figure"'), 4)


if __name__ == "__main__":
    unittest.main()
