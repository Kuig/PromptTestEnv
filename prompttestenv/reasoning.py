"""Reasoning trace analysis for thinking-enabled candidates.

The trace is split into sentence-sized units **procedurally**, and the judge is
only ever asked for numbers against those unit ids. The text itself never passes
through the LLM, so full coverage and zero overlap are structural properties of
the split rather than instructions the judge may ignore; the previous
"rewrite the whole trace verbatim into 4 JSON fields" approach lost up to 40% of
a trace and double-counted spans.

Each unit is scored **independently on every dimension** (see
``config.json`` -> ``reasoning_schema``). The dimensions are not a partition: a
sentence that weighs an approach while producing a step of the answer scores on
both, which is exactly the co-occurrence a mutually-exclusive taxonomy destroys.
Coverages therefore do not sum to 1, and their sum is itself a metric (density).

Usage:
    from prompttestenv.reasoning import analyze_reasoning, aggregate_reasoning_stats
"""
from __future__ import annotations

import concurrent.futures
import json
import re

import prompttestenv.logger as logger
from prompttestenv.api import (
    cosine_similarity,
    get_llm_response,
    get_text_embedding,
    is_local_provider,
)
from prompttestenv.config import ReasoningSchema, UnitSplittingConfig, get_app_config
from prompttestenv.models import (
    REASONING_DIMENSIONS,
    JudgeConfig,
    ReasoningStats,
    ReasoningUnit,
    calculate_stats,
)

_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*].*\*\*\s*$)")
_SENTENCE_END_RE = re.compile(r"""[.!?…]+["'”’)\]]*(\s+)""")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_WORDS_FOR_REPETITION = 30


# ---------------------------------------------------------------------------
# Procedural segmentation (no LLM involved)
# ---------------------------------------------------------------------------

def _iter_blocks(text: str) -> list[tuple[int, int, bool]]:
    """Group the trace into blocks, flagging the ones that must stay atomic.

    Fenced code blocks and heading lines are never split further: sentence
    punctuation inside them does not mark a thought boundary.

    Args:
        text: The raw reasoning trace.

    Returns:
        List of (start, end, atomic) character spans, in order.
    """
    lines: list[tuple[int, int]] = []
    pos = 0
    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.rstrip("\r\n")
        lines.append((pos, pos + len(stripped)))
        pos += len(raw_line)

    blocks: list[tuple[int, int, bool]] = []
    i = 0
    while i < len(lines):
        start, end = lines[i]
        content = text[start:end]
        if not content.strip():
            i += 1
            continue
        if _FENCE_RE.match(content):
            j = i + 1
            while j < len(lines) and not _FENCE_RE.match(text[lines[j][0]:lines[j][1]]):
                j += 1
            close = j if j < len(lines) else len(lines) - 1
            blocks.append((start, lines[close][1], True))
            i = close + 1
            continue
        if _HEADING_RE.match(content):
            blocks.append((start, end, True))
            i += 1
            continue
        j = i
        last_end = end
        while j + 1 < len(lines):
            nxt_start, nxt_end = lines[j + 1]
            nxt = text[nxt_start:nxt_end]
            if not nxt.strip() or _FENCE_RE.match(nxt) or _HEADING_RE.match(nxt):
                break
            last_end = nxt_end
            j += 1
        blocks.append((start, last_end, False))
        i = j + 1
    return blocks


def _split_sentences(
    text: str,
    start: int,
    end: int,
    abbreviations: list[str],
) -> list[tuple[int, int]]:
    """Split one paragraph into sentence spans.

    Args:
        text: The full reasoning trace.
        start: Paragraph start offset.
        end: Paragraph end offset.
        abbreviations: Tokens whose trailing dot never ends a sentence.

    Returns:
        List of (start, end) spans covering the paragraph, whitespace trimmed.
    """
    segment = text[start:end]
    lowered = [a.lower() for a in abbreviations]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END_RE.finditer(segment):
        cut = match.start(1)
        head = segment[cursor:cut]
        if any(head.rstrip().lower().endswith(a) for a in lowered):
            continue
        tail = segment[match.end(1):match.end(1) + 1]
        if tail and not (tail.isupper() or tail.isdigit() or tail in "\"'([*_#-“"):
            continue
        if head.strip():
            spans.append((start + cursor, start + cut))
        cursor = match.end(1)
    if segment[cursor:].strip():
        spans.append((start + cursor, end))
    return [_trim(text, a, b) for a, b in spans]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span so it starts and ends on non-whitespace.

    Args:
        text: The full reasoning trace.
        start: Span start offset.
        end: Span end offset.

    Returns:
        The trimmed (start, end) span.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _merge_short(
    text: str,
    spans: list[tuple[int, int]],
    min_chars: int,
) -> list[tuple[int, int]]:
    """Absorb spans too short to score into their neighbour.

    A three-word fragment carries no scorable signal on its own but does inflate
    the unit count, so it is folded into the preceding unit (or the following
    one, when it is the very first span). Titles such as a trace's opening bold
    line are absorbed this way instead of being dropped, which is what the old
    verbatim segmentation did to them.

    Args:
        text: The full reasoning trace.
        spans: Ordered, non-overlapping spans.
        min_chars: Minimum length a standalone unit must reach.

    Returns:
        The merged span list.
    """
    if not spans:
        return []
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if end - start < min_chars or merged[-1][1] - merged[-1][0] < min_chars:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def split_into_units(
    text: str,
    splitting: UnitSplittingConfig | None = None,
) -> list[tuple[int, int]]:
    """Split a reasoning trace into sentence-sized units.

    Pure and deterministic: no LLM call, no network. The returned spans are
    ordered, never overlap, and together cover the whole trace apart from
    whitespace between units, which is the invariant the judge cannot break
    because it never sees the text.

    Args:
        text: The raw reasoning trace.
        splitting: Splitter parameters. Defaults to the global config.json.

    Returns:
        Ordered list of (start, end) character offsets into ``text``.
    """
    if splitting is None:
        splitting = get_app_config().unit_splitting
    spans: list[tuple[int, int]] = []
    for start, end, atomic in _iter_blocks(text):
        if atomic:
            spans.append(_trim(text, start, end))
        else:
            spans.extend(_split_sentences(text, start, end, splitting.abbreviations))
    spans = [(a, b) for a, b in spans if b > a]
    return _merge_short(text, spans, splitting.min_unit_chars)


def _render_units(
    text: str,
    spans: list[tuple[int, int]],
    window: tuple[int, int],
) -> str:
    """Render a numbered listing of one window of units for the judge.

    Ids are global, so a judge's answer maps straight back onto ``spans`` no
    matter how the trace was windowed. The two units preceding the window are
    included as context and explicitly marked unscorable.

    Args:
        text: The full reasoning trace.
        spans: All unit spans.
        window: Half-open (first, last) index range to score.

    Returns:
        One line per unit, ``[id] sentence``.
    """
    first, last = window
    lines = []
    for idx in range(max(0, first - 2), last):
        start, end = spans[idx]
        body = re.sub(r"\s+", " ", text[start:end]).strip()
        marker = " (context, do not score)" if idx < first else ""
        lines.append(f"[{idx + 1}]{marker} {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge calls
# ---------------------------------------------------------------------------

def _resolve(override, fallback):
    """Return an explicit per-project override, or the global default.

    Args:
        override: Value from judge_config.json, or None when unset.
        fallback: Value from config.json's reasoning_defaults.

    Returns:
        The effective value.
    """
    return fallback if override is None else override


def _example_json(schema: ReasoningSchema, joint: bool) -> str:
    """Render the answer shape the judge is asked to imitate.

    Built here rather than hand-escaped in the template: the nested braces a
    literal example needs are easy to miscount, and a miscount makes str.format
    raise instead of producing a prompt.

    Args:
        schema: The reasoning schema, for the dimension names.
        joint: True for the all-dimensions-at-once shape.

    Returns:
        A compact JSON string.
    """
    if joint:
        per_unit = {d.name: 0 for d in schema.dimensions}
        example = {"scores": {"1": per_unit, "2": per_unit}}
    else:
        example = {"scores": {"1": 0, "2": schema.intensity_scale}}
    return json.dumps(example)


def _format_template(template: str, **values) -> str:
    """Fill a schema prompt template, reporting a bad template instead of raising.

    A template is authored data, not code, so a stray placeholder or an unbalanced
    brace must degrade to "this analysis was not measured" rather than take down
    the run (CONVENTIONS 7).

    Args:
        template: The template from config.json.
        **values: Placeholder values.

    Returns:
        The formatted prompt, or an empty string if the template is malformed.
    """
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        logger.log_error(
            f"Malformed reasoning prompt template in config.json ({exc}). Skipping this call."
        )
        return ""


def _ask_judge(prompt: str, judge_config: JudgeConfig) -> dict | None:
    """Send one prompt to the reasoning judge and parse its JSON answer.

    Args:
        prompt: The fully formatted user prompt.
        judge_config: JudgeConfig carrying the reasoning_judge settings.

    Returns:
        The parsed JSON object, or None if the call or the parse failed.
    """
    from prompttestenv.api import call_with_timeout

    rj = judge_config.reasoning_judge
    schema = get_app_config().reasoning_schema
    result, timed_out = call_with_timeout(
        get_llm_response,
        fn_kwargs=dict(
            provider=rj.provider,
            model_name=rj.model,
            system_instruction=schema.system_prompt or None,
            user_prompt=prompt,
            temp=rj.temperature,
            thinking=rj.thinking,
            response_mime_type="application/json",
        ),
        timeout=judge_config.max_response_timeout_seconds,
        provider=rj.provider,
        model=rj.model,
    )
    if timed_out:
        logger.log_warning(
            f"Reasoning judge timeout ({judge_config.max_response_timeout_seconds}s)."
        )
        return None
    try:
        data = json.loads(result.text)
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.log_error(f"Reasoning judge returned unparseable JSON: {exc}")
        return None
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else None
    return data if isinstance(data, dict) else None


def _parse_scores(
    payload: dict,
    window: tuple[int, int],
    scale: int,
    dimension: str | None,
) -> dict[int, float]:
    """Extract per-unit intensities from a judge answer.

    Ids outside the scored window are dropped (the context lines), ids the judge
    omitted default to 0, and out-of-range values are clamped: a judge that
    skips a sentence is saying "not this dimension", which is a real 0, whereas
    a failed call is handled one level up and yields -1 instead.

    Args:
        payload: Parsed judge JSON.
        window: Half-open (first, last) index range that was scored.
        scale: Maximum intensity declared by the schema.
        dimension: Dimension name for joint mode, or None for split mode.

    Returns:
        Mapping of unit index to intensity.
    """
    raw = payload.get("scores")
    if not isinstance(raw, dict):
        return {}
    first, last = window
    scores: dict[int, float] = {}
    for key, value in raw.items():
        try:
            idx = int(key) - 1
        except (TypeError, ValueError):
            continue
        if not first <= idx < last:
            continue
        if dimension is not None:
            if not isinstance(value, dict):
                continue
            value = value.get(dimension, 0)
        try:
            scores[idx] = max(0.0, min(float(scale), float(value)))
        except (TypeError, ValueError):
            continue
    return scores


def _score_window(
    text: str,
    spans: list[tuple[int, int]],
    window: tuple[int, int],
    schema: ReasoningSchema,
    judge_config: JudgeConfig,
    user_prompt: str,
    criteria: str,
    joint: bool,
) -> dict[str, dict[int, float] | None]:
    """Score one window of units on every dimension.

    In split mode each dimension is a separate call asking a single-concept
    question, which is what keeps the task tractable for a small local judge.
    The calls only run concurrently against a remote provider: a local backend
    serves one model at a time, so parallel requests queue or force a reload.

    Args:
        text: The full reasoning trace.
        spans: All unit spans.
        window: Half-open (first, last) index range to score.
        schema: The reasoning schema from config.json.
        judge_config: JudgeConfig carrying the reasoning_judge settings.
        user_prompt: The task the candidate was given.
        criteria: The criteria the task was judged against.
        joint: If True, ask for all dimensions in a single call.

    Returns:
        Mapping of dimension name to per-unit intensities, or to None when that
        dimension's call failed.
    """
    numbered = _render_units(text, spans, window)

    if joint:
        prompt = _format_template(
            schema.joint_template,
            dimension_count=len(schema.dimensions),
            dimension_definitions="\n\n".join(
                f"DIMENSION: {d.name}\n{d.definition}" for d in schema.dimensions
            ),
            intensity_scale=schema.intensity_scale,
            example_json=_example_json(schema, joint=True),
            user_prompt=user_prompt,
            criteria=criteria,
            numbered_units=numbered,
        )
        payload = _ask_judge(prompt, judge_config) if prompt else None
        if payload is None:
            return {d.name: None for d in schema.dimensions}
        return {
            d.name: _parse_scores(payload, window, schema.intensity_scale, d.name)
            for d in schema.dimensions
        }

    def score_one(dimension) -> dict[int, float] | None:
        prompt = _format_template(
            schema.dimension_template,
            dimension_name=dimension.name,
            dimension_definition=dimension.definition,
            intensity_scale=schema.intensity_scale,
            example_json=_example_json(schema, joint=False),
            user_prompt=user_prompt,
            criteria=criteria,
            numbered_units=numbered,
        )
        payload = _ask_judge(prompt, judge_config) if prompt else None
        if payload is None:
            return None
        return _parse_scores(payload, window, schema.intensity_scale, None)

    if is_local_provider(judge_config.reasoning_judge.provider):
        return {d.name: score_one(d) for d in schema.dimensions}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(schema.dimensions)) as pool:
        futures = {d.name: pool.submit(score_one, d) for d in schema.dimensions}
        return {name: future.result() for name, future in futures.items()}


def _extract_metrics(
    text: str,
    spans: list[tuple[int, int]],
    schema: ReasoningSchema,
    judge_config: JudgeConfig,
    user_prompt: str,
    candidate_response: str,
    stats: ReasoningStats,
) -> None:
    """Run the evidence-anchored metrics call and store its results.

    Counts are derived from the cited unit ids rather than reported directly, so
    every alternative and every self-correction is traceable to a sentence in
    the report instead of being an unverifiable number.

    Args:
        text: The full reasoning trace.
        spans: All unit spans.
        schema: The reasoning schema from config.json.
        judge_config: JudgeConfig carrying the reasoning_judge settings.
        user_prompt: The task the candidate was given.
        candidate_response: The candidate's final response.
        stats: The ReasoningStats to populate in place. Left at its -1
            sentinels when the call fails.
    """
    prompt = _format_template(
        schema.metrics_template,
        user_prompt=user_prompt,
        numbered_units=_render_units(text, spans, (0, len(spans))),
        candidate_response=candidate_response,
    )
    payload = _ask_judge(prompt, judge_config) if prompt else None
    if payload is None:
        logger.log_warning("Reasoning metrics unavailable; recorded as not measured.")
        return

    def unit_ids(key: str) -> list[int]:
        raw = payload.get(key) or []
        if not isinstance(raw, list):
            return []
        seen = []
        for value in raw:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(spans) and idx not in seen:
                seen.append(idx)
        return seen

    stats.alt_path_units = unit_ids("alt_path_units")
    stats.autocorrect_units = unit_ids("autocorrect_units")
    stats.alignment_evidence = unit_ids("alignment_evidence")
    stats.alt_path = len(stats.alt_path_units)
    stats.autocorrect = len(stats.autocorrect_units)
    try:
        stats.alignment_score = max(1, min(10, int(payload.get("alignment_score", 5))))
    except (TypeError, ValueError):
        stats.alignment_score = -1


# ---------------------------------------------------------------------------
# Procedural metrics (no LLM involved)
# ---------------------------------------------------------------------------

def compute_repetition_rate(text: str) -> float:
    """Measure how much of a trace is repeated word trigrams.

    A raw local reasoning trace that loops on itself scores high here. Gemini's
    thought summaries almost never do, which is one more reason not to compare
    a summarised trace against a raw one.

    Args:
        text: The raw reasoning trace.

    Returns:
        Share of non-distinct trigrams in [0, 1], or -1.0 when the trace is too
        short for the figure to mean anything.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < _MIN_WORDS_FOR_REPETITION:
        return -1.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return round(1.0 - len(set(trigrams)) / len(trigrams), 4)


def compute_trace_response_drift(
    reasoning_text: str,
    candidate_response: str,
    judge_config: JudgeConfig,
) -> float:
    """Cosine similarity between the trace and the response it produced.

    An objective companion to the judge's alignment_score, which saturates near
    10. Only comparable between candidates on the same test case.

    Args:
        reasoning_text: The raw reasoning trace.
        candidate_response: The candidate's final response.
        judge_config: JudgeConfig carrying the similarity_judge settings.

    Returns:
        Cosine similarity, or -1.0 if embeddings were unavailable.
    """
    if not reasoning_text.strip() or not candidate_response.strip():
        return -1.0
    sj = judge_config.similarity_judge
    try:
        trace_vec = get_text_embedding(sj.provider, sj.model, reasoning_text)
        response_vec = get_text_embedding(sj.provider, sj.model, candidate_response)
    except Exception as exc:
        logger.log_warning(f"Trace/response drift unavailable: {exc}")
        return -1.0
    return round(cosine_similarity(trace_vec, response_vec), 4)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_coverages(stats: ReasoningStats, schema: ReasoningSchema) -> None:
    """Turn per-unit intensities into per-dimension coverages, in place.

    Each coverage is a length-weighted mean intensity in [0, 1]. They are
    deliberately not normalised to sum to 1: their sum is ``density``, which
    says how many concerns the trace carries at once, and normalising would
    throw exactly that away.

    Args:
        stats: Stats whose ``units`` are already scored.
        schema: The reasoning schema, for the intensity scale.
    """
    total_weight = sum(u.length for u in stats.units)
    if not total_weight:
        return
    measured = []
    for dimension in REASONING_DIMENSIONS:
        if stats.coverage(dimension) < 0:
            continue
        weighted = sum(u.length * u.intensity(dimension) for u in stats.units)
        coverage = weighted / (total_weight * schema.intensity_scale)
        stats.set_coverage(dimension, round(coverage, 4))
        measured.append(coverage)
    stats.density = round(sum(measured), 4) if measured else -1.0


def analyze_reasoning(
    reasoning_text: str,
    judge_config: JudgeConfig,
    candidate_response: str = "",
    user_prompt: str = "",
    criteria: str = "",
    reasoning_is_summary: bool | None = None,
) -> ReasoningStats | None:
    """Analyze a candidate's reasoning trace.

    Splits the trace procedurally, has the judge score every unit on every
    dimension, then runs one evidence-anchored metrics call and two procedural
    metrics that cost nothing.

    Returns None only when there is nothing to analyse (analysis disabled, empty
    trace, unusable schema). A judge call that fails degrades to the -1
    "not measured" sentinel rather than discarding the whole analysis.

    Args:
        reasoning_text: The thinking transcript from the candidate model.
        judge_config: JudgeConfig carrying the reasoning_judge settings.
        candidate_response: The candidate's final response, used by the metrics call.
        user_prompt: The task the candidate was given. Without it the judge
            cannot tell restating the request apart from reasoning about it.
        criteria: The criteria the task was judged against.
        reasoning_is_summary: Whether the trace is a provider-written summary
            rather than the raw chain of thought.

    Returns:
        A populated ReasoningStats, or None if analysis was skipped.
    """
    if not judge_config.reasoning_analysis:
        return None
    if not reasoning_text.strip():
        return None

    app_config = get_app_config()
    schema = app_config.reasoning_schema
    if list(schema.dimension_names) != list(REASONING_DIMENSIONS):
        logger.log_error(
            f"config.json declares dimensions {schema.dimension_names}, but this build "
            f"stores {list(REASONING_DIMENSIONS)}. Skipping reasoning analysis."
        )
        return None
    if not schema.dimension_template or not schema.metrics_template:
        logger.log_warning("Reasoning schema has no prompt templates. Skipping analysis.")
        return None

    rj = judge_config.reasoning_judge
    defaults = app_config.reasoning_defaults
    mode = _resolve(rj.dimension_mode, defaults.dimension_mode)
    passes = max(1, int(_resolve(rj.reliability_k, defaults.reliability_k)))
    chunk = max(1, int(_resolve(rj.max_units_per_call, defaults.max_units_per_call)))

    spans = split_into_units(reasoning_text, app_config.unit_splitting)
    if not spans:
        return None

    stats = ReasoningStats(
        units=[ReasoningUnit(start=a, end=b) for a, b in spans],
        reasoning_is_summary=reasoning_is_summary,
        schema_stamp=schema.stamp,
    )

    windows = [(i, min(i + chunk, len(spans))) for i in range(0, len(spans), chunk)]
    totals: dict[str, dict[int, float]] = {d: {} for d in REASONING_DIMENSIONS}
    failed: set[str] = set()
    for window in windows:
        for _ in range(passes):
            scored = _score_window(
                reasoning_text, spans, window, schema, judge_config,
                user_prompt, criteria, joint=(mode == "joint"),
            )
            for dimension, values in scored.items():
                if values is None:
                    failed.add(dimension)
                    continue
                bucket = totals[dimension]
                for idx in range(*window):
                    bucket[idx] = bucket.get(idx, 0.0) + values.get(idx, 0.0)

    for dimension in REASONING_DIMENSIONS:
        if dimension in failed:
            stats.set_coverage(dimension, -1.0)
            continue
        stats.set_coverage(dimension, 0.0)
        for idx, total in totals[dimension].items():
            stats.units[idx].set_intensity(dimension, total / passes)

    compute_coverages(stats, schema)
    _extract_metrics(
        reasoning_text, spans, schema, judge_config,
        user_prompt, candidate_response, stats,
    )
    stats.repetition_rate = compute_repetition_rate(reasoning_text)
    stats.trace_response_drift = compute_trace_response_drift(
        reasoning_text, candidate_response, judge_config
    )
    return stats


def aggregate_reasoning_stats(analyses: list[dict]) -> dict:
    """Aggregate per-repetition reasoning analyses into mean and std.

    Values recorded as -1 ("not measured") are dropped by ``calculate_stats``
    rather than averaged in as zeros, so one failed judge call no longer drags a
    candidate's profile down.

    Args:
        analyses: Per-repetition dicts, each produced by ReasoningStats.to_dict().

    Returns:
        Dict with avg_/std_ keys for every coverage and metric, plus ``n``
        (number of analyses), ``is_summary`` (True when any trace was a
        provider-written summary) and ``schema_stamps``. Empty if there is
        nothing to aggregate.
    """
    if not analyses:
        return {}

    result: dict = {"n": len(analyses)}
    numeric_fields = (
        [f"coverage_{d}" for d in REASONING_DIMENSIONS]
        + ["density", "alt_path", "autocorrect", "alignment_score",
           "repetition_rate", "trace_response_drift"]
    )
    for name in numeric_fields:
        values = [a[name] for a in analyses if name in a and a[name] is not None]
        mean, std = calculate_stats(values, default_val=-1.0)
        result[f"avg_{name}"] = round(mean, 4)
        result[f"std_{name}"] = round(std, 4)

    result["is_summary"] = any(a.get("reasoning_is_summary") for a in analyses)
    result["schema_stamps"] = sorted({a.get("schema_stamp", "") for a in analyses} - {""})
    return result
