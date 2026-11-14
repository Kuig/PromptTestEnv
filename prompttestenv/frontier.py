"""Efficiency frontiers: which candidates no other candidate beats on both axes.

A panel plots one point per candidate, its mean task score against the mean of
one cost axis (time, output tokens or thinking tokens), and marks the Pareto
frontier: the candidates for which no other candidate is both cheaper and
better. Three kinds of panel exist:

- one per group, over that group's test cases;
- one per cluster declared in ``group_clusters.json``, over the groups it lists;
- one "All tests" panel, over every group of the project.

A cluster or "All tests" point is the **mean of the per-group means**, so every
group weighs the same however many test cases it holds. That is the reading a
cluster of benchmarks asks for, and it differs from pooling every test case the
moment two groups differ in size.

``group_clusters.json`` is optional and deliberately not part of the run hash
(progress.HASHED_FILENAMES): it changes how results are read, never what was
measured, so editing it costs nothing and ``render`` picks it up for free. For
the same reason nothing in it may ever stop a run: a malformed file degrades to
the defaults with a warning.

This module computes only. Drawing lives in frontier_plot.py, and the verdict
payload formats panels itself, so the three surfaces read the same Panel
objects and cannot disagree about who is on a frontier.
"""
from __future__ import annotations

import json
import numbers
from dataclasses import dataclass, field
from pathlib import Path

import prompttestenv.logger as logger
from prompttestenv.models import (
    DEFAULT_GROUP,
    Candidate,
    CandidatePerformance,
    TestCaseResult,
    _write_json_atomic,
    pool_by_candidate,
)

GROUP_CLUSTERS_FILE = "group_clusters.json"

X_AXIS_TIME = "time"
X_AXIS_OUT_TOKENS = "out_tokens"
X_AXIS_THINK_TOKENS = "think_tokens"
X_AXES = (X_AXIS_TIME, X_AXIS_OUT_TOKENS, X_AXIS_THINK_TOKENS)

# How each axis reads off a pooled record, and how it is labelled. The lists
# carry -1 for "not measured" (placeholders, refusals), which is filtered out
# exactly as calculate_stats filters it.
_AXIS_VALUES = {
    X_AXIS_TIME: lambda perf: perf.times,
    X_AXIS_OUT_TOKENS: lambda perf: perf.tokens,
    X_AXIS_THINK_TOKENS: lambda perf: perf.reasoning_tokens,
}
AXIS_LABELS = {
    X_AXIS_TIME: "mean time (s)",
    X_AXIS_OUT_TOKENS: "mean output tokens",
    X_AXIS_THINK_TOKENS: "mean thinking tokens",
}

# "upper": no other candidate is both cheaper and scores HIGHER (the usual
# reading). "lower": no other candidate is both cheaper and scores LOWER, for a
# cluster where the interesting end is the low one.
FRONT_UPPER = "upper"
FRONT_LOWER = "lower"
FRONTS = (FRONT_UPPER, FRONT_LOWER)

PANEL_GROUP = "group"
PANEL_CLUSTER = "cluster"
PANEL_OVERALL = "overall"
OVERALL_PANEL_NAME = "All tests"

_OPTION_KEYS = ("x_axis", "front", "threshold")
_TOP_LEVEL_KEYS = ("defaults", "include_in_verdict", "exclude_candidates", "clusters")
_CLUSTER_KEYS = ("name", "groups") + _OPTION_KEYS


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PanelOptions:
    """Options as written: None means "not set here, inherit"."""

    x_axis: str | None = None
    front: str | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class ResolvedOptions:
    """Options after inheritance, as a panel actually uses them."""

    x_axis: str = X_AXIS_TIME
    front: str = FRONT_UPPER
    threshold: float | None = None


@dataclass
class ClusterSpec:
    name: str
    groups: list[str] = field(default_factory=list)
    options: PanelOptions = field(default_factory=PanelOptions)


@dataclass
class GroupClusters:
    """The parsed group_clusters.json. An empty instance is the file being absent."""

    defaults: PanelOptions = field(default_factory=PanelOptions)
    include_in_verdict: bool = True
    exclude_candidates: list[str] = field(default_factory=list)
    clusters: list[ClusterSpec] = field(default_factory=list)

    def resolve_defaults(self) -> ResolvedOptions:
        """The options of the per-group and "All tests" panels.

        Those panels read `defaults` only: a cluster's options describe the
        cluster panel, never the panels of the groups it happens to contain.
        """
        return _resolve(self.defaults, PanelOptions())

    def resolve(self, cluster: ClusterSpec) -> ResolvedOptions:
        """A cluster panel's options: its own, then `defaults`, then built-ins."""
        return _resolve(cluster.options, self.defaults)

    @classmethod
    def load(cls, project_dir: str | Path) -> GroupClusters:
        """Read group_clusters.json, or the defaults when it is absent or broken.

        Never raises: the file only changes how results are read, so a typo in
        it must not stop a run that has already paid for its responses.

        Args:
            project_dir: Path to the benchmark project directory.

        Returns:
            The parsed configuration; an empty one when there is no file.
        """
        path = Path(project_dir) / GROUP_CLUSTERS_FILE
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.log_warning(f"Cannot read {path}: {exc}. Using the default frontier settings.")
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data) -> GroupClusters:
        """Build the configuration from its raw dict, tolerating mistakes.

        Every problem check_group_clusters() finds is logged as a warning, and
        the offending value is dropped (an option falls back to its inherited
        value, a malformed cluster is skipped). The editor and `edit` reject the
        same problems as errors before they can be saved.

        Args:
            data: The parsed JSON content.

        Returns:
            The configuration, with every invalid part dropped.
        """
        errors, _ = check_group_clusters(data)
        for message in errors:
            logger.log_warning(f"{GROUP_CLUSTERS_FILE}: {message}")
        if not isinstance(data, dict):
            return cls()

        include = data.get("include_in_verdict", True)
        excluded = data.get("exclude_candidates", [])
        clusters = []
        raw_clusters = data.get("clusters", [])
        for entry in raw_clusters if isinstance(raw_clusters, list) else []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            groups = entry.get("groups", [])
            clusters.append(ClusterSpec(
                name=name,
                groups=[g for g in groups if isinstance(g, str)] if isinstance(groups, list) else [],
                options=_options_from(entry),
            ))
        return cls(
            defaults=_options_from(data.get("defaults")),
            include_in_verdict=include if isinstance(include, bool) else True,
            exclude_candidates=(
                [c for c in excluded if isinstance(c, str)] if isinstance(excluded, list) else []
            ),
            clusters=clusters,
        )

    @staticmethod
    def save(
        project_dir: str | Path, data: dict, *,
        trailing_newline: bool = True, newline: str = "\n",
    ) -> None:
        """Write group_clusters.json from a raw dict.

        Takes a plain dict for the same fidelity reason as Candidate.save_all():
        round-tripping through the dataclass would drop unknown keys and spell
        out every inherited option.
        """
        _write_json_atomic(
            Path(project_dir) / GROUP_CLUSTERS_FILE, data,
            trailing_newline=trailing_newline, newline=newline,
        )


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _options_from(entry) -> PanelOptions:
    """Read the three options off a dict, dropping any invalid value."""
    if not isinstance(entry, dict):
        return PanelOptions()
    x_axis = entry.get("x_axis")
    front = entry.get("front")
    threshold = entry.get("threshold")
    return PanelOptions(
        x_axis=x_axis if x_axis in X_AXES else None,
        front=front if front in FRONTS else None,
        threshold=float(threshold) if _is_number(threshold) else None,
    )


def _resolve(own: PanelOptions, inherited: PanelOptions) -> ResolvedOptions:
    built_in = ResolvedOptions()
    return ResolvedOptions(
        x_axis=own.x_axis or inherited.x_axis or built_in.x_axis,
        front=own.front or inherited.front or built_in.front,
        threshold=own.threshold if own.threshold is not None else inherited.threshold,
    )


def _check_options(entry: dict, where: str, errors: list[str]) -> None:
    x_axis = entry.get("x_axis")
    if x_axis is not None and x_axis not in X_AXES:
        errors.append(f"{where}: unknown x_axis '{x_axis}'. Expected one of {', '.join(X_AXES)}.")
    front = entry.get("front")
    if front is not None and front not in FRONTS:
        errors.append(f"{where}: unknown front '{front}'. Expected one of {', '.join(FRONTS)}.")
    threshold = entry.get("threshold")
    if threshold is not None and not _is_number(threshold):
        errors.append(f"{where}: threshold must be a number or null, got {threshold!r}.")


def check_group_clusters(data) -> tuple[list[str], list[str]]:
    """Check the structure of group_clusters.json's content.

    The one definition of a valid file: from_dict() logs these errors as
    warnings and drops the offending parts, the editor and `edit` block on them.
    Whether the groups and candidates it names exist is a separate question,
    answered by reference_warnings(), since it needs the rest of the project.

    Args:
        data: The parsed JSON content.

    Returns:
        ``(errors, warnings)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        errors.append("the file must hold a JSON object.")
        return errors, warnings

    for key in data:
        if key not in _TOP_LEVEL_KEYS:
            errors.append(f"unknown key '{key}'. Expected one of {', '.join(_TOP_LEVEL_KEYS)}.")

    defaults = data.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, dict):
            errors.append("'defaults' must be an object.")
        else:
            for key in defaults:
                if key not in _OPTION_KEYS:
                    errors.append(f"defaults: unknown key '{key}'. Expected one of {', '.join(_OPTION_KEYS)}.")
            _check_options(defaults, "defaults", errors)

    if "include_in_verdict" in data and not isinstance(data["include_in_verdict"], bool):
        errors.append("'include_in_verdict' must be true or false.")

    excluded = data.get("exclude_candidates")
    if excluded is not None and (
        not isinstance(excluded, list) or not all(isinstance(c, str) for c in excluded)
    ):
        errors.append("'exclude_candidates' must be a list of candidate names.")

    clusters = data.get("clusters")
    if clusters is None:
        return errors, warnings
    if not isinstance(clusters, list):
        errors.append("'clusters' must be a list.")
        return errors, warnings

    seen: set[str] = set()
    for index, entry in enumerate(clusters, start=1):
        if not isinstance(entry, dict):
            errors.append(f"cluster #{index} is not an object.")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"cluster #{index} has no name.")
            label = f"cluster #{index}"
        else:
            label = f"cluster '{name}'"
            if name in seen:
                errors.append(f"{label} is declared more than once.")
            seen.add(name)
        for key in entry:
            if key not in _CLUSTER_KEYS:
                errors.append(f"{label}: unknown key '{key}'. Expected one of {', '.join(_CLUSTER_KEYS)}.")
        groups = entry.get("groups", [])
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            errors.append(f"{label}: 'groups' must be a list of group names.")
        elif not groups:
            warnings.append(f"{label} lists no groups, so it gets no panel.")
        _check_options(entry, label, errors)
    return errors, warnings


def reference_warnings(
    config: GroupClusters, group_names: list[str], candidate_names: list[str]
) -> list[str]:
    """Names in the configuration that the project does not have.

    Warnings, never errors: a cluster naming a group that a test-case edit is
    about to add is a legitimate intermediate state, and the panel simply
    covers the groups that exist.
    """
    warnings = []
    known_groups = set(group_names)
    for cluster in config.clusters:
        for group in cluster.groups:
            if group not in known_groups:
                warnings.append(
                    f"{GROUP_CLUSTERS_FILE}: cluster '{cluster.name}' names group '{group}', "
                    "which no test case uses."
                )
    known_candidates = set(candidate_names)
    for name in config.exclude_candidates:
        if name not in known_candidates:
            warnings.append(
                f"{GROUP_CLUSTERS_FILE}: excluded candidate '{name}' is not in candidates.json."
            )
    return warnings


# ── Computation ───────────────────────────────────────────────────────────────


@dataclass
class PanelPoint:
    candidate: str
    x: float
    y: float
    # Groups this point averages over; below the panel's group count when the
    # candidate has no usable measurement in some of them.
    n: int
    baseline: bool = False
    on_front: bool = False


@dataclass
class Panel:
    kind: str
    name: str
    groups: list[str]
    options: ResolvedOptions
    points: list[PanelPoint] = field(default_factory=list)

    @property
    def frontier(self) -> list[PanelPoint]:
        """The frontier, cheapest first."""
        return sorted((p for p in self.points if p.on_front), key=lambda p: p.x)

    @property
    def listed(self) -> list[PanelPoint]:
        """The frontier after the threshold, which only filters what is listed.

        Upper front keeps scores above the threshold, lower front keeps those
        below: the end of the frontier worth naming, in either orientation.
        """
        threshold = self.options.threshold
        if threshold is None:
            return self.frontier
        if self.options.front == FRONT_LOWER:
            return [p for p in self.frontier if p.y < threshold]
        return [p for p in self.frontier if p.y > threshold]


@dataclass
class FrontierPanels:
    overall: Panel
    groups: list[Panel] = field(default_factory=list)
    clusters: list[Panel] = field(default_factory=list)
    include_in_verdict: bool = True
    warnings: list[str] = field(default_factory=list)

    def group_panel(self, name: str) -> Panel | None:
        return next((p for p in self.groups if p.name == name), None)


def _valid(values: list) -> list:
    return [v for v in values if v >= 0]


def group_point(perf: CandidatePerformance, x_axis: str) -> tuple[float, float] | None:
    """One candidate's (x, y) over one group, or None when it has nothing usable.

    Nothing usable means no valid score, or no valid value on the axis: a point
    with a made-up coordinate would be worse than a missing one. A thinking
    count of 0 IS a valid value (the model does not think), so it is kept.
    """
    scores = _valid(perf.scores)
    xs = _valid(_AXIS_VALUES[x_axis](perf))
    if not scores or not xs:
        return None
    return sum(xs) / len(xs), sum(scores) / len(scores)


def pareto_front(points: list[PanelPoint], front: str = FRONT_UPPER) -> list[PanelPoint]:
    """The points no other point beats on both axes, cheapest first.

    Ported from OmnibusAnalysis's plot_clusters.py: sort by x (ties broken by
    the better score), then keep each point that strictly improves on the best
    score seen so far. A tie on both axes keeps only the first point.

    Args:
        points: Candidate points; baselines must already be filtered out.
        front: FRONT_UPPER (higher is better) or FRONT_LOWER.

    Returns:
        The frontier points, ordered by x.
    """
    sign = -1 if front == FRONT_LOWER else 1
    result: list[PanelPoint] = []
    best = float("-inf")
    for point in sorted(points, key=lambda p: (p.x, -sign * p.y)):
        if sign * point.y > best:
            result.append(point)
            best = sign * point.y
    return result


def _panel(
    kind: str,
    name: str,
    groups: list[str],
    options: ResolvedOptions,
    candidates: list[Candidate],
    pooled_by_group: dict[str, dict[str, CandidatePerformance]],
    excluded: set[str],
) -> Panel:
    points = []
    for cand in candidates:
        if cand.name in excluded:
            continue
        per_group = [
            point
            for group in groups
            if (point := group_point(pooled_by_group[group][cand.name], options.x_axis)) is not None
        ]
        if not per_group:
            continue
        points.append(PanelPoint(
            candidate=cand.name,
            x=sum(x for x, _ in per_group) / len(per_group),
            y=sum(y for _, y in per_group) / len(per_group),
            n=len(per_group),
            baseline=cand.is_baseline,
        ))
    for point in pareto_front([p for p in points if not p.baseline], options.front):
        point.on_front = True
    return Panel(kind=kind, name=name, groups=list(groups), options=options, points=points)


def build_panels(
    candidates: list[Candidate],
    results: list[TestCaseResult],
    config: GroupClusters | None = None,
) -> FrontierPanels:
    """Compute every frontier panel of a project.

    Args:
        candidates: Resolved candidate configurations, in report order.
        results: Test case results.
        config: The project's group_clusters.json; None means the file is absent.

    Returns:
        The panels, plus warnings about names the configuration uses but the
        project does not have. Nothing is logged here: the caller logs once.
    """
    config = config or GroupClusters()
    rows_by_group: dict[str, list[TestCaseResult]] = {}
    for row in results:
        rows_by_group.setdefault(row.group or DEFAULT_GROUP, []).append(row)
    group_names = list(rows_by_group)
    pooled_by_group = {
        group: pool_by_candidate(candidates, rows) for group, rows in rows_by_group.items()
    }
    excluded = set(config.exclude_candidates)
    defaults = config.resolve_defaults()

    def panel(kind, name, groups, options):
        return _panel(kind, name, groups, options, candidates, pooled_by_group, excluded)

    clusters = []
    for cluster in config.clusters:
        present = [g for g in cluster.groups if g in rows_by_group]
        if present:
            clusters.append(panel(PANEL_CLUSTER, cluster.name, present, config.resolve(cluster)))

    return FrontierPanels(
        overall=panel(PANEL_OVERALL, OVERALL_PANEL_NAME, group_names, defaults),
        groups=[panel(PANEL_GROUP, g, [g], defaults) for g in group_names],
        clusters=clusters,
        include_in_verdict=config.include_in_verdict,
        warnings=reference_warnings(config, group_names, [c.name for c in candidates]),
    )
