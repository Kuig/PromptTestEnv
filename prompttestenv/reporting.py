from __future__ import annotations

import html
import importlib.resources
import os
import re

import jinja2

from prompttestenv.config import get_app_config
from prompttestenv.models import (
    REASONING_DIMENSIONS,
    Candidate,
    CandidatePerformance,
    pool_by_candidate,
    JudgeConfig,
    TestCaseResult,
    GlobalCriteria,
)
from prompttestenv.reasoning import aggregate_reasoning_stats
from prompttestenv.verdict import parse_grouped_verdict


def get_badge_class(score):
    try:
        val = float(score)
        if val >= 8.0: return "high-score"
        if val >= 5.0: return "mid-score"
        return "low-score"
    except Exception:
        return "low-score"


def format_thinking_value(val: bool | str | None) -> str:
    if val is True:
        return "ON"
    if val is False or val is None:
        return "OFF"
    if isinstance(val, str):
        v = val.lower().strip()
        if v in ("true", "on"):
            return "ON"
        if v in ("false", "off", "none", ""):
            return "OFF"
        if v == "default":
            return "DEFAULT"
        return val.upper()
    return str(val).upper()


NEUTRAL_UNIT_COLOR = "#9e9e9e"


def _tint(hex_color: str, ratio: float) -> str:
    """Build a translucent background from a dimension colour and an intensity.

    Tinting the background rather than fading the whole span keeps the trace
    readable at every intensity, which matters because the point of the view is
    that the reader can check the analysis against the actual words.

    Args:
        hex_color: Dimension colour as ``#rrggbb``.
        ratio: Intensity in [0, 1].

    Returns:
        An ``rgba(...)`` CSS colour, or ``transparent`` when the intensity is 0.
    """
    if ratio <= 0:
        return "transparent"
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return "transparent"
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    alpha = round(0.15 + 0.55 * min(1.0, ratio), 3)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def build_trace_segments(perf: CandidatePerformance) -> list[dict]:
    """Turn a scored trace into renderable, colour-coded spans.

    The report shows the whole trace tinted in place rather than four buckets of
    copied text: since the units are character offsets the reader can see that
    every word is accounted for, which is the property the old rewrite-based
    segmentation could not offer.

    Args:
        perf: Candidate performance holding best_reasoning_text and
            best_reasoning_analysis.

    Returns:
        Ordered list of span dicts. Gap spans (whitespace between units) carry
        ``unit_id`` 0 and no colour.
    """
    analysis = perf.best_reasoning_analysis
    text = perf.best_reasoning_text
    if analysis is None or not text or not analysis.units:
        return []

    colors = {d.name: d.color for d in get_app_config().reasoning_schema.dimensions}
    scale = max(1, get_app_config().reasoning_schema.intensity_scale)
    segments: list[dict] = []
    cursor = 0
    for index, unit in enumerate(analysis.units, start=1):
        if unit.start > cursor:
            segments.append({"text": text[cursor:unit.start], "unit_id": 0})
        intensities = {d: unit.intensity(d) for d in REASONING_DIMENSIONS}
        top = max(intensities, key=lambda d: intensities[d])
        peak = intensities[top]
        markers = ""
        if index in analysis.alt_path_units:
            markers += "⑂"
        if index in analysis.autocorrect_units:
            markers += "↺"
        if index in analysis.alignment_evidence:
            markers += "⚠"
        segments.append({
            "text": text[unit.start:unit.end],
            "unit_id": index,
            "background": _tint(colors.get(top, NEUTRAL_UNIT_COLOR), peak / scale),
            "title": " | ".join(f"{d}: {intensities[d]:.1f}/{scale}" for d in REASONING_DIMENSIONS),
            "markers": markers,
        })
        cursor = unit.end
    if cursor < len(text):
        segments.append({"text": text[cursor:], "unit_id": 0})
    return segments


def format_cost_per_point(cost: float) -> str:
    """Format CandidatePerformance.reasoning_cost_per_point for display.

    Formatting only: the arithmetic and the -1 "not measured" sentinel belong
    to the record, so the verdict payload and the report cannot disagree about
    what a candidate cost.

    Args:
        cost: The ratio, or -1.0 when it was not measurable.

    Returns:
        The rounded figure, or "N/A".
    """
    return "N/A" if cost < 0 else f"{cost:.0f}"


def md_to_html(md_text):
    def _process_table(match):
        block = match.group(0).strip()
        rows = block.split('\n')
        if len(rows) < 2: return block
        headers = [h.strip() for h in rows[0].strip('|').split('|')]
        html_parts = ['<table border="1">', '<thead>', '<tr>']
        for h in headers: html_parts.append(f'<th>{h}</th>')
        html_parts.append('</tr></thead><tbody>')
        for row in rows[2:]:
            html_parts.append('<tr>')
            cells = [c.strip() for c in row.strip('|').split('|')]
            if len(cells) < len(headers): cells += [''] * (len(headers) - len(cells))
            for c in cells: html_parts.append(f'<td>{c}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody></table>\n')
        return "".join(html_parts)

    output = html.escape(md_text, quote=False)
    output = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', output, flags=re.DOTALL)
    table_pattern = r'(?:^\s*\|.*?\|\s*\n)(?:^\s*\|[\s\-\:\|]+\|\s*\n)(?:^\s*\|.*?\|\s*(?:\n|$))*'
    output = re.sub(table_pattern, _process_table, output, flags=re.MULTILINE)
    output = re.sub(r'`(.*?)`', r'<code>\1</code>', output)
    output = re.sub(r'^\s*[\-\*]\s+(.*?)$', r'<li>\1</li>', output, flags=re.MULTILINE)
    output = re.sub(r'(<li>.*?</li>(\n\s*<li>.*?</li>)*)', r'<ul>\n\1\n</ul>', output, flags=re.MULTILINE)
    for i in range(6, 0, -1):
        pattern = r'^#{' + str(i) + r'}\s+(.*?)$'
        output = re.sub(pattern, r'<h' + str(i) + r'>\1</h' + str(i) + r'>', output, flags=re.MULTILINE)
    output = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1">', output)
    output = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', output)
    output = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', output)
    output = re.sub(r'__(.*?)__', r'<strong>\1</strong>', output)
    output = re.sub(r'\*(.*?)\*', r'<em>\1</em>', output)
    output = re.sub(r'_(.*?)_', r'<em>\1</em>', output)
    output = output.replace('\n\n', '<br>')
    return output


def generate_html_report(project_dir: str, results: list[TestCaseResult], candidates: list[Candidate], verdict_text: str, global_criteria: GlobalCriteria, judge_config: JudgeConfig, filename: str = "report_benchmark.html"):
    verdict_data = parse_grouped_verdict(verdict_text)
    is_grouped = verdict_data is not None

    if is_grouped:
        verdict_html = ""
        for g in verdict_data["groups"]:
            verdict_html += f"<details>\n<summary><strong>Verdict for Group: {g['group_name']}</strong></summary>\n<div style='padding-top: 10px;'>\n"
            verdict_html += md_to_html(g["verdict"])
            verdict_html += "\n</div>\n</details>\n<br>\n"
        verdict_html += "<details open>\n<summary><strong>Global Verdict</strong></summary>\n<div style='padding-top: 10px;'>\n"
        verdict_html += md_to_html(verdict_data["global_verdict"])
        verdict_html += "\n</div>\n</details>"
    else:
        verdict_html = md_to_html(verdict_text)
    
    # One aggregation for every surface: the same pooled records the verdict
    # payload reads. This used to be a second, hand-rolled copy returning raw
    # dicts, which is why the cost-per-point figure could only live in the
    # template instead of on the record it describes.
    stats = pool_by_candidate(candidates, results)
    reasoning_stats = {
        name: aggregate_reasoning_stats(perf.reasoning_analyses)
        for name, perf in stats.items()
    }

    template_text = importlib.resources.files("prompttestenv").joinpath("templates/report_template.html").read_text(encoding="utf-8")
    template = jinja2.Template(template_text, autoescape=True)
    html_content = template.render(
        project_name=os.path.basename(project_dir),
        verdict_html=verdict_html,
        candidates=candidates,
        results=results,
        stats=stats,
        global_criteria=global_criteria,
        judge_config=judge_config,
        reasoning_stats=reasoning_stats,
        reasoning_analysis_enabled=judge_config.reasoning_enabled,
        reasoning_dimensions=get_app_config().reasoning_schema.dimensions,
        get_badge_class=get_badge_class,
        format_thinking_value=format_thinking_value,
        build_trace_segments=build_trace_segments,
        format_cost_per_point=format_cost_per_point,
    )
    report_dir = os.path.join(project_dir, "Report")
    os.makedirs(report_dir, exist_ok=True)
    html_file = os.path.join(report_dir, filename)
    with open(html_file, "w", encoding="utf-8") as f: 
        f.write(html_content)
    return html_file
