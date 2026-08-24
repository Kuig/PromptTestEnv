from __future__ import annotations

import html
import importlib.resources
import os
import re

import jinja2

from prompttestenv.models import calculate_stats, Candidate, JudgeConfig, TestCaseResult, GlobalCriteria
from prompttestenv.reasoning import aggregate_reasoning_stats


def get_badge_class(score):
    try:
        val = float(score)
        if val >= 8.0: return "high-score"
        if val >= 5.0: return "mid-score"
        return "low-score"
    except:
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
    is_grouped = False
    verdict_data = None
    if verdict_text.strip().startswith("{"):
        try:
            import json
            verdict_data = json.loads(verdict_text)
            if verdict_data.get("is_grouped"):
                is_grouped = True
        except Exception:
            pass

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
    
    stats = {}
    for cand in candidates:
        cand_id = cand.name
        task_scores = []
        global_scores = []
        token_counts = []
        reasoning_counts = []
        time_counts = []
        for row in results:
            perf = row.candidates_perf.get(cand_id)
            if perf:
                task_scores.extend(perf.scores)
                global_scores.extend(perf.global_scores)
                token_counts.extend(perf.tokens)
                reasoning_counts.extend(perf.reasoning_tokens)
                time_counts.extend(perf.times)
        
        t_mean, t_std = calculate_stats(task_scores)
        g_mean, g_std = calculate_stats(global_scores, default_val=-1.0)
        tok_mean, tok_std = calculate_stats(token_counts)
        r_mean, r_std = calculate_stats(reasoning_counts)
        time_mean, time_std = calculate_stats(time_counts)
        
        if g_mean < 0:
            combined_avg = t_mean
        else:
            combined_avg = (t_mean + g_mean) / 2 if (t_mean or g_mean) else 0.0
        
        stats[cand_id] = {
            "task_mean": t_mean, "task_std": t_std,
            "global_mean": g_mean, "global_std": g_std,
            "token_mean": tok_mean, "token_std": tok_std,
            "reasoning_mean": r_mean, "reasoning_std": r_std,
            "time_mean": time_mean, "time_std": time_std,
            "combined_avg": combined_avg
        }

    reasoning_stats: dict[str, dict] = {}
    for cand in candidates:
        cand_id = cand.name
        all_analyses: list[dict] = []
        for row in results:
            perf = row.candidates_perf.get(cand_id)
            if perf:
                all_analyses.extend(perf.reasoning_analyses)
        reasoning_stats[cand_id] = aggregate_reasoning_stats(all_analyses)

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
        reasoning_analysis_enabled=judge_config.reasoning_analysis,
        get_badge_class=get_badge_class,
        format_thinking_value=format_thinking_value,
    )
    report_dir = os.path.join(project_dir, "Report")
    os.makedirs(report_dir, exist_ok=True)
    html_file = os.path.join(report_dir, filename)
    with open(html_file, "w", encoding="utf-8") as f: 
        f.write(html_content)
    return html_file
