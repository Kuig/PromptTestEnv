"""Unit tests for prompttestenv/frontier.py: group_clusters.json and the panels."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompttestenv import frontier as fr
from prompttestenv.models import (
    CANDIDATE_ROLE_BASELINE,
    Candidate,
    CandidatePerformance,
    TestCaseResult,
)


def _perf(scores, times=None, tokens=None, think=None):
    n = len(scores)
    return CandidatePerformance(
        scores=list(scores),
        global_scores=[10] * n,
        times=list(times if times is not None else [1.0] * n),
        tokens=list(tokens if tokens is not None else [100] * n),
        reasoning_tokens=list(think if think is not None else [0] * n),
    )


def _row(test_id, group, perfs):
    return TestCaseResult(
        test_id=test_id, prompt="p", criteria="c", group=group, candidates_perf=perfs,
    )


def _cand(name, role="candidate"):
    return Candidate(name=name, provider="ollama", model="m", role=role)


def _names(points):
    return [p.candidate for p in points]


class TestParetoFront(unittest.TestCase):
    def _pt(self, name, x, y):
        return fr.PanelPoint(candidate=name, x=x, y=y, n=1)

    def test_upper_keeps_points_no_one_beats_on_both_axes(self):
        points = [self._pt("fast_bad", 1, 5), self._pt("slow_good", 10, 9),
                  self._pt("dominated", 5, 4), self._pt("mid", 4, 7)]
        self.assertEqual(_names(fr.pareto_front(points)), ["fast_bad", "mid", "slow_good"])

    def test_lower_keeps_the_cheap_low_end(self):
        points = [self._pt("a", 1, 5), self._pt("b", 10, 2),
                  self._pt("c", 5, 6), self._pt("d", 4, 3)]
        self.assertEqual(_names(fr.pareto_front(points, fr.FRONT_LOWER)), ["a", "d", "b"])

    def test_tie_on_x_keeps_the_better_score_only(self):
        points = [self._pt("worse", 2, 5), self._pt("better", 2, 8)]
        self.assertEqual(_names(fr.pareto_front(points)), ["better"])

    def test_exact_tie_keeps_one(self):
        points = [self._pt("a", 2, 8), self._pt("b", 2, 8)]
        self.assertEqual(len(fr.pareto_front(points)), 1)

    def test_empty(self):
        self.assertEqual(fr.pareto_front([]), [])


class TestGroupPoint(unittest.TestCase):
    def test_means_ignore_not_measured(self):
        perf = _perf([8, -1, 6], times=[10.0, -1, 20.0])
        self.assertEqual(fr.group_point(perf, fr.X_AXIS_TIME), (15.0, 7.0))

    def test_no_valid_score_means_no_point(self):
        self.assertIsNone(fr.group_point(_perf([-1, -1]), fr.X_AXIS_TIME))

    def test_no_valid_axis_value_means_no_point(self):
        self.assertIsNone(fr.group_point(_perf([5], tokens=[-1]), fr.X_AXIS_OUT_TOKENS))

    def test_zero_thinking_is_a_real_measurement(self):
        self.assertEqual(fr.group_point(_perf([5, 7], think=[0, 0]), fr.X_AXIS_THINK_TOKENS), (0.0, 6.0))


class TestBuildPanels(unittest.TestCase):
    def setUp(self):
        self.cands = [_cand("A"), _cand("B"), _cand("Base", CANDIDATE_ROLE_BASELINE)]
        # Group G1 has two tests, G2 one: the cluster mean must weigh groups equally.
        self.rows = [
            _row("t1", "G1", {"A": _perf([10], [1.0]), "B": _perf([4], [9.0]), "Base": _perf([10], [0.5])}),
            _row("t2", "G1", {"A": _perf([10], [1.0]), "B": _perf([4], [9.0]), "Base": _perf([10], [0.5])}),
            _row("t3", "G2", {"A": _perf([2], [3.0]), "B": _perf([8], [1.0]), "Base": _perf([1], [0.5])}),
        ]

    def test_one_panel_per_group_plus_overall(self):
        panels = fr.build_panels(self.cands, self.rows)
        self.assertEqual([p.name for p in panels.groups], ["G1", "G2"])
        self.assertEqual(panels.overall.name, fr.OVERALL_PANEL_NAME)
        self.assertEqual(panels.overall.groups, ["G1", "G2"])
        self.assertEqual(panels.clusters, [])

    def test_overall_is_the_mean_of_group_means(self):
        panels = fr.build_panels(self.cands, self.rows)
        a = next(p for p in panels.overall.points if p.candidate == "A")
        # Pooled over tests it would be (10+10+2)/3; per group it is (10+2)/2.
        self.assertEqual(a.y, 6.0)
        self.assertEqual(a.x, 2.0)
        self.assertEqual(a.n, 2)

    def test_baseline_is_drawn_but_never_on_the_frontier(self):
        panels = fr.build_panels(self.cands, self.rows)
        g1 = panels.group_panel("G1")
        base = next(p for p in g1.points if p.candidate == "Base")
        self.assertTrue(base.baseline)
        self.assertFalse(base.on_front)
        # Base would dominate A in G1 (faster, same score); A stays on it.
        self.assertIn("A", _names(g1.frontier))

    def test_excluded_candidate_disappears(self):
        config = fr.GroupClusters(exclude_candidates=["B"])
        panels = fr.build_panels(self.cands, self.rows, config)
        for panel in [panels.overall, *panels.groups]:
            self.assertNotIn("B", _names(panel.points))

    def test_partial_coverage_counts_groups(self):
        self.rows[2].candidates_perf["A"] = _perf([-1])
        panels = fr.build_panels(self.cands, self.rows)
        a = next(p for p in panels.overall.points if p.candidate == "A")
        self.assertEqual(a.n, 1)
        self.assertEqual(a.y, 10.0)

    def test_cluster_uses_its_own_options_and_present_groups(self):
        config = fr.GroupClusters(
            defaults=fr.PanelOptions(x_axis=fr.X_AXIS_OUT_TOKENS),
            clusters=[fr.ClusterSpec("C", ["G2", "Missing"], fr.PanelOptions(front=fr.FRONT_LOWER))],
        )
        panels = fr.build_panels(self.cands, self.rows, config)
        cluster = panels.clusters[0]
        self.assertEqual(cluster.groups, ["G2"])
        self.assertEqual(cluster.options.front, fr.FRONT_LOWER)
        self.assertEqual(cluster.options.x_axis, fr.X_AXIS_OUT_TOKENS)
        self.assertTrue(any("Missing" in w for w in panels.warnings))

    def test_cluster_options_never_reach_group_panels(self):
        config = fr.GroupClusters(
            clusters=[fr.ClusterSpec("C", ["G2"], fr.PanelOptions(front=fr.FRONT_LOWER))],
        )
        panels = fr.build_panels(self.cands, self.rows, config)
        self.assertEqual(panels.group_panel("G2").options.front, fr.FRONT_UPPER)
        self.assertEqual(panels.overall.options.front, fr.FRONT_UPPER)

    def test_cluster_with_no_present_group_gets_no_panel(self):
        config = fr.GroupClusters(clusters=[fr.ClusterSpec("C", ["Nope"])])
        self.assertEqual(fr.build_panels(self.cands, self.rows, config).clusters, [])

    def test_unknown_excluded_candidate_warns(self):
        config = fr.GroupClusters(exclude_candidates=["Ghost"])
        self.assertTrue(any("Ghost" in w for w in fr.build_panels(self.cands, self.rows, config).warnings))


class TestThreshold(unittest.TestCase):
    def _panel(self, front, threshold):
        options = fr.ResolvedOptions(front=front, threshold=threshold)
        points = [fr.PanelPoint("lo", 1, 4, 1, on_front=True), fr.PanelPoint("hi", 5, 8, 1, on_front=True)]
        return fr.Panel(fr.PANEL_CLUSTER, "C", ["G"], options, points)

    def test_upper_lists_above(self):
        self.assertEqual(_names(self._panel(fr.FRONT_UPPER, 6).listed), ["hi"])

    def test_lower_lists_below(self):
        self.assertEqual(_names(self._panel(fr.FRONT_LOWER, 6).listed), ["lo"])

    def test_no_threshold_lists_the_whole_frontier(self):
        self.assertEqual(_names(self._panel(fr.FRONT_UPPER, None).listed), ["lo", "hi"])


class TestResolve(unittest.TestCase):
    def test_built_in_defaults(self):
        self.assertEqual(fr.GroupClusters().resolve_defaults(), fr.ResolvedOptions())

    def test_cluster_inherits_from_defaults(self):
        config = fr.GroupClusters(defaults=fr.PanelOptions(x_axis=fr.X_AXIS_THINK_TOKENS, threshold=5))
        cluster = fr.ClusterSpec("C", ["G"], fr.PanelOptions(front=fr.FRONT_LOWER))
        self.assertEqual(
            config.resolve(cluster),
            fr.ResolvedOptions(x_axis=fr.X_AXIS_THINK_TOKENS, front=fr.FRONT_LOWER, threshold=5),
        )

    def test_cluster_threshold_zero_is_not_inherited_over(self):
        config = fr.GroupClusters(defaults=fr.PanelOptions(threshold=5))
        cluster = fr.ClusterSpec("C", ["G"], fr.PanelOptions(threshold=0))
        self.assertEqual(config.resolve(cluster).threshold, 0)


class TestLoadAndCheck(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="prompttestenv_frontier_"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _write(self, text):
        (self.dir / fr.GROUP_CLUSTERS_FILE).write_text(text, encoding="utf-8")

    def test_absent_file_is_the_empty_configuration(self):
        self.assertEqual(fr.GroupClusters.load(self.dir), fr.GroupClusters())

    def test_full_file(self):
        self._write(json.dumps({
            "defaults": {"x_axis": "think_tokens", "threshold": 6},
            "include_in_verdict": False,
            "exclude_candidates": ["X"],
            "clusters": [{"name": "Safety", "groups": ["HarmBench"], "front": "lower"}],
        }))
        config = fr.GroupClusters.load(self.dir)
        self.assertEqual(config.defaults, fr.PanelOptions(x_axis="think_tokens", threshold=6.0))
        self.assertFalse(config.include_in_verdict)
        self.assertEqual(config.exclude_candidates, ["X"])
        self.assertEqual(config.clusters, [
            fr.ClusterSpec("Safety", ["HarmBench"], fr.PanelOptions(front="lower")),
        ])

    def test_malformed_json_degrades_to_defaults(self):
        self._write("{not json")
        with patch.object(fr.logger, "log_warning") as warn:
            self.assertEqual(fr.GroupClusters.load(self.dir), fr.GroupClusters())
        warn.assert_called_once()

    def test_invalid_values_are_dropped_with_a_warning(self):
        with patch.object(fr.logger, "log_warning") as warn:
            config = fr.GroupClusters.from_dict({
                "defaults": {"x_axis": "cost", "front": "sideways", "threshold": "6"},
                "clusters": [{"groups": ["G"]}, "oops", {"name": "Ok", "groups": ["G"]}],
            })
        self.assertEqual(config.defaults, fr.PanelOptions())
        self.assertEqual([c.name for c in config.clusters], ["Ok"])
        self.assertGreaterEqual(warn.call_count, 5)

    def test_check_flags_structure(self):
        errors, warnings = fr.check_group_clusters({
            "extra": 1,
            "defaults": {"colour": "red"},
            "include_in_verdict": "yes",
            "exclude_candidates": "A",
            "clusters": [{"name": "C", "groups": []}, {"name": "C", "groups": "G", "threshold": True}],
        })
        joined = "\n".join(errors)
        for fragment in ("'extra'", "'colour'", "include_in_verdict", "exclude_candidates",
                         "more than once", "'groups' must be", "threshold"):
            self.assertIn(fragment, joined)
        self.assertEqual(len(warnings), 1)

    def test_check_accepts_a_valid_file(self):
        self.assertEqual(fr.check_group_clusters({
            "defaults": {"x_axis": "time", "front": "upper", "threshold": None},
            "clusters": [{"name": "C", "groups": ["G"], "threshold": 6}],
        }), ([], []))

    def test_check_rejects_a_non_object(self):
        self.assertEqual(len(fr.check_group_clusters([])[0]), 1)

    def test_save_writes_raw_dict_byte_faithfully(self):
        data = {"clusters": [{"name": "C", "groups": ["G"]}]}
        fr.GroupClusters.save(self.dir, data, newline="\r\n")
        raw = (self.dir / fr.GROUP_CLUSTERS_FILE).read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertEqual(json.loads(raw), data)


class TestCandidateRole(unittest.TestCase):
    def test_default_and_baseline(self):
        self.assertFalse(Candidate.from_dict({"name": "A"}).is_baseline)
        self.assertTrue(Candidate.from_dict({"name": "A", "role": "baseline"}).is_baseline)

    def test_unknown_role_degrades_with_a_warning(self):
        from prompttestenv import models
        with patch.object(models.logger, "log_warning") as warn:
            cand = Candidate.from_dict({"name": "A", "role": "basline"})
        self.assertEqual(cand.role, "candidate")
        warn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
