"""Draw frontier panels (frontier.Panel) as inline SVG for the HTML report.

Inline SVG rather than a plotting library: the report is a single
self-contained file, and matplotlib would be a heavy dependency for six
scatterplots. The layout follows OmnibusAnalysis's plot_clusters.py: log x
axis, score on y, the frontier as a dashed line whose colour fades from one
point's score colour to the next, labels placed greedily to avoid overlaps.

Two things differ from that script on purpose:

- **A "0" band.** A thinking-token count of 0 is a real measurement, and a log
  axis cannot show it, so such points sit on a band left of the axis.
- **Baselines are references.** They are drawn as hollow grey markers with an
  italic label, and frontier.build_panels never puts them on the frontier.

Every name reaching the markup is author-written free text and goes through
html.escape, since the result is inserted into the page unescaped.
"""
from __future__ import annotations

import hashlib
import html
import math

from prompttestenv.frontier import (
    AXIS_LABELS,
    FRONT_LOWER,
    X_AXIS_TIME,
    Panel,
    PanelPoint,
)

WIDTH = 560
HEIGHT = 360
_MARGIN_LEFT = 46
_MARGIN_RIGHT = 14
_MARGIN_TOP = 14
_MARGIN_BOTTOM = 40
_ZERO_BAND = 30  # px reserved left of the log axis when some x is 0
_FONT = 10

SURFACE = "#ffffff"
INK = "#1d1d1b"
INK_2 = "#5a5955"
GRID = "#e4e3df"
BLUE = "#2a78d6"    # score >= 9
GREEN = "#1baf7a"   # 8 <= score < 9
YELLOW = "#e9bc08"  # 6 <= score < 8
RED = "#d03b3b"     # score < 6
MUTED = "#a5a49d"   # dominated
BASELINE = "#7d7c76"

_LEGEND = (
    (BLUE, "score &ge; 9"), (GREEN, "8 &le; score &lt; 9"), (YELLOW, "6 &le; score &lt; 8"),
    (RED, "score &lt; 6"), (MUTED, "dominated"),
)

# Tried in order for each label; the first without collisions wins. Same set
# as plot_clusters.place_labels, in SVG units (y grows downwards).
_OFFSETS = (
    (7, -5, "start"), (7, 13, "start"), (-7, -5, "end"), (-7, 13, "end"),
    (0, -10, "middle"), (0, 19, "middle"), (12, 4, "start"), (-12, 4, "end"),
    (9, -17, "start"), (9, 25, "start"), (-9, -17, "end"), (-9, 25, "end"),
)


def score_color(score: float) -> str:
    return RED if score < 6 else YELLOW if score < 8 else GREEN if score < 9 else BLUE


def format_axis_value(value: float, x_axis: str) -> str:
    """A cost value as the report prints it: seconds, or a token count."""
    if x_axis == X_AXIS_TIME:
        return f"{value:.2f}s"
    return f"{value:,.0f}"


def _tick_label(value: float, x_axis: str) -> str:
    if x_axis == X_AXIS_TIME:
        return f"{value:g}s"
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _log_ticks(lo: float, hi: float) -> list[float]:
    """1-3-10 ticks inside [lo, hi], thinned to powers of ten when crowded."""
    ticks = []
    for exponent in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for mantissa in (1, 3):
            value = mantissa * 10 ** exponent
            if lo <= value <= hi:
                ticks.append(value)
    if len(ticks) > 7:
        ticks = [t for t in ticks if math.isclose(math.log10(t) % 1, 0, abs_tol=1e-9)]
    return ticks


class _Scale:
    """Data to SVG coordinates for one panel."""

    def __init__(self, points: list[PanelPoint]):
        positive = [p.x for p in points if p.x > 0]
        self.has_zero = any(p.x <= 0 for p in points)
        if positive:
            lo, hi = min(positive), max(positive)
            if math.isclose(lo, hi):
                lo, hi = lo / 3, hi * 3
            self.x_lo, self.x_hi = lo / 1.5, hi * 1.5
        else:
            self.x_lo, self.x_hi = 1.0, 10.0
        ys = [p.y for p in points] or [5.0]
        pad = 0.15 * (max(ys) - min(ys)) + 0.3
        self.y_lo, self.y_hi = min(ys) - pad, min(max(ys) + pad, 10.5)

        self.left = _MARGIN_LEFT + (_ZERO_BAND if self.has_zero else 0)
        self.right = WIDTH - _MARGIN_RIGHT
        self.top = _MARGIN_TOP
        self.bottom = HEIGHT - _MARGIN_BOTTOM

    def x(self, value: float) -> float:
        if value <= 0:
            return _MARGIN_LEFT + _ZERO_BAND / 2
        span = math.log10(self.x_hi) - math.log10(self.x_lo)
        return self.left + (math.log10(value) - math.log10(self.x_lo)) / span * (self.right - self.left)

    def y(self, value: float) -> float:
        return self.bottom - (value - self.y_lo) / (self.y_hi - self.y_lo) * (self.bottom - self.top)


def _y_ticks(lo: float, hi: float) -> list[int]:
    step = 2 if hi - lo > 6 else 1
    first = math.ceil(lo)
    return [v for v in range(first, math.floor(hi) + 1) if (v - first) % step == 0]


def _label_text(point: PanelPoint, n_groups: int) -> str:
    return point.candidate if point.n == n_groups else f"{point.candidate} ({point.n}/{n_groups})"


def _place_labels(items, scale: _Scale) -> list[tuple[float, float, str]]:
    """Greedy label placement, as plot_clusters.place_labels with estimated text boxes.

    Args:
        items: ``(text, px, py, bold)`` per point, in drawing order.
        scale: The panel scale, for the plot area bounds.

    Returns:
        ``(x, y, anchor)`` per item.
    """
    placed: list[tuple[float, float, float, float]] = []
    anchors = [(px, py) for _, px, py, _ in items]
    result = []
    for text, px, py, bold in items:
        width = len(text) * _FONT * (0.6 if bold else 0.55)
        height = _FONT * 1.2
        best = None
        for dx, dy, anchor in _OFFSETS:
            tx, ty = px + dx, py + dy
            x0 = tx - (width if anchor == "end" else width / 2 if anchor == "middle" else 0)
            box = (x0 - 2, ty - height, x0 + width + 2, ty + 3)
            hits = sum(
                not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1])
                for b in placed
            )
            hits += sum(
                box[0] - 5 <= qx <= box[2] + 5 and box[1] - 5 <= qy <= box[3] + 5
                for qx, qy in anchors if (qx, qy) != (px, py)
            )
            outside = not (0 <= box[0] and box[2] <= WIDTH and 0 <= box[1] and box[3] <= scale.bottom)
            cost = hits * 10 + outside * 5
            if best is None or cost < best[0]:
                best = (cost, tx, ty, anchor, box)
            if cost == 0:
                break
        _, tx, ty, anchor, box = best
        placed.append(box)
        result.append((tx, ty, anchor))
    return result


def _frontier_line(panel: Panel, scale: _Scale, id_prefix: str) -> list[str]:
    """The dashed frontier, one gradient per segment, dash phase carried across."""
    front = panel.frontier
    defs, lines, run = [], [], 0.0
    for index, (a, b) in enumerate(zip(front, front[1:])):
        ax, ay, bx, by = scale.x(a.x), scale.y(a.y), scale.x(b.x), scale.y(b.y)
        length = math.hypot(bx - ax, by - ay)
        if length < 0.5:
            continue
        gid = f"{id_prefix}g{index}"
        defs.append(
            f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
            f'x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}">'
            f'<stop offset="0" stop-color="{score_color(a.y)}"/>'
            f'<stop offset="1" stop-color="{score_color(b.y)}"/></linearGradient>'
        )
        lines.append(
            f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="url(#{gid})" stroke-width="1.6" stroke-dasharray="8 6.4" '
            f'stroke-dashoffset="{-run:.1f}"/>'
        )
        run += length
    if not lines:
        return []
    return [f"<defs>{''.join(defs)}</defs>", *lines]


def render_panel_svg(panel: Panel) -> str:
    """Draw one panel as an inline SVG element.

    Args:
        panel: The panel to draw, from frontier.build_panels.

    Returns:
        The ``<svg>`` markup; a short placeholder when the panel has no points.
    """
    if not panel.points:
        return (
            f'<svg viewBox="0 0 {WIDTH} 60" class="frontier-svg" role="img">'
            f'<text x="{WIDTH / 2}" y="34" text-anchor="middle" font-size="12" fill="{INK_2}">'
            "No measurable data for this panel.</text></svg>"
        )

    id_prefix = "f" + hashlib.md5(f"{panel.kind}:{panel.name}".encode("utf-8")).hexdigest()[:8]
    scale = _Scale(panel.points)
    x_axis = panel.options.x_axis
    out = [
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="frontier-svg" role="img" '
        f'font-family="Segoe UI, Tahoma, sans-serif">',
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
    ]

    for tick in _log_ticks(scale.x_lo, scale.x_hi):
        x = scale.x(tick)
        out.append(f'<line x1="{x:.1f}" y1="{scale.top}" x2="{x:.1f}" y2="{scale.bottom}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x:.1f}" y="{scale.bottom + 14}" text-anchor="middle" font-size="9" '
            f'fill="{INK_2}">{_tick_label(tick, x_axis)}</text>'
        )
    if scale.has_zero:
        x = scale.x(0)
        out.append(
            f'<rect x="{_MARGIN_LEFT}" y="{scale.top}" width="{_ZERO_BAND - 4}" '
            f'height="{scale.bottom - scale.top}" fill="#f4f3ef"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{scale.bottom + 14}" text-anchor="middle" font-size="9" '
            f'fill="{INK_2}">0</text>'
        )
    for tick in _y_ticks(scale.y_lo, scale.y_hi):
        y = scale.y(tick)
        out.append(f'<line x1="{_MARGIN_LEFT}" y1="{y:.1f}" x2="{scale.right}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{_MARGIN_LEFT - 6}" y="{y + 3:.1f}" text-anchor="end" font-size="9" '
            f'fill="{INK_2}">{tick}</text>'
        )
    out.append(
        f'<text x="{(scale.left + scale.right) / 2:.1f}" y="{HEIGHT - 8}" text-anchor="middle" '
        f'font-size="10" fill="{INK_2}">{html.escape(AXIS_LABELS[x_axis])}, log scale</text>'
    )
    out.append(
        f'<text x="12" y="{(scale.top + scale.bottom) / 2:.1f}" text-anchor="middle" font-size="10" '
        f'fill="{INK_2}" transform="rotate(-90 12 {(scale.top + scale.bottom) / 2:.1f})">'
        "mean task score</text>"
    )

    out.extend(_frontier_line(panel, scale, id_prefix))

    n_groups = len(panel.groups)
    # Dominated first, then baselines, then the frontier on top.
    ordered = (
        [p for p in panel.points if not p.on_front and not p.baseline]
        + [p for p in panel.points if p.baseline]
        + [p for p in panel.points if p.on_front]
    )
    for point in ordered:
        px, py = scale.x(point.x), scale.y(point.y)
        title = html.escape(
            f"{point.candidate}{' (baseline)' if point.baseline else ''}: "
            f"score {point.y:.2f}, {format_axis_value(point.x, x_axis)}, "
            f"groups {point.n}/{n_groups}"
        )
        if point.baseline:
            shape = (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{SURFACE}" '
                     f'stroke="{BASELINE}" stroke-width="2"/>')
        elif point.on_front:
            shape = (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{score_color(point.y)}" '
                     f'stroke="{SURFACE}" stroke-width="2"/>')
        else:
            shape = (f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{MUTED}" '
                     f'stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f"<g>{shape}<title>{title}</title></g>")

    items = [(_label_text(p, n_groups), scale.x(p.x), scale.y(p.y), p.on_front) for p in ordered]
    for point, (text, *_), (tx, ty, anchor) in zip(ordered, items, _place_labels(items, scale)):
        style = ' font-weight="bold"' if point.on_front else ""
        style += ' font-style="italic"' if point.baseline else ""
        color = INK if point.on_front else INK_2
        out.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="{_FONT}" '
            f'fill="{color}"{style}>{html.escape(text)}</text>'
        )

    out.append("</svg>")
    return "".join(out)


def _caption(panel: Panel) -> str:
    groups = len(panel.groups)
    front = "lower front" if panel.options.front == FRONT_LOWER else "upper front"
    parts = [f"<b>{html.escape(panel.name)}</b>", f"{groups} group{'s' if groups != 1 else ''}", front]
    if panel.options.threshold is not None:
        sign = "&lt;" if panel.options.front == FRONT_LOWER else "&gt;"
        parts.append(f"listed: score {sign} {panel.options.threshold:g}")
    return " &middot; ".join(parts)


def _legend() -> str:
    dots = "".join(
        f'<span><i style="background:{color};"></i>{label}</span>' for color, label in _LEGEND
    )
    return (
        f'<div class="frontier-legend">{dots}'
        f'<span><i class="hollow" style="border-color:{BASELINE};"></i>baseline</span></div>'
    )


def _table(panel: Panel) -> str:
    listed = panel.listed
    if not listed:
        return '<div class="frontier-note">No frontier candidate passes the threshold.</div>'
    rows = "".join(
        f"<tr><td>{html.escape(p.candidate)}</td><td>{p.y:.2f}</td>"
        f"<td>{format_axis_value(p.x, panel.options.x_axis)}</td>"
        f"<td>{p.n}/{len(panel.groups)}</td></tr>"
        for p in listed
    )
    return (
        '<table class="frontier-table"><thead><tr><th>Frontier candidate</th>'
        f"<th>Task score</th><th>{html.escape(AXIS_LABELS[panel.options.x_axis])}</th>"
        f"<th>Groups</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def render_panel_html(panel: Panel) -> str:
    """One panel as a figure: caption, plot, legend and the listed frontier."""
    return (
        '<figure class="frontier-figure">'
        f'<figcaption>{_caption(panel)}</figcaption>'
        f"{render_panel_svg(panel)}{_legend()}{_table(panel)}</figure>"
    )


def render_panels_grid(panels: list[Panel]) -> str:
    """Several panels side by side, wrapping on narrow screens."""
    return '<div class="frontier-grid">' + "".join(render_panel_html(p) for p in panels) + "</div>"
