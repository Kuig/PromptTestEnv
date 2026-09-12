"""The verdict payload's metadata block: repetition count, metric semantics,
judge-type legend.

Split out of verdict.py, which otherwise mixes this with Markdown table
rendering and LLM-call orchestration. This module is deliberately light —
config and models only, no api/reasoning import — so it stays independent of
the LLM-orchestration weight the rest of verdict.py carries.
"""
from __future__ import annotations

from prompttestenv.config import get_app_config
from prompttestenv.models import (
    CandidatePerformance,
    GlobalCriteria,
    JUDGE_TYPE_ASSERT,
    JUDGE_TYPE_LLM,
    JUDGE_TYPE_SIMILARITY,
    JudgeConfig,
    TestCaseResult,
)


class MetadataError(Exception):
    """A metadata section the payload needs is missing or will not format.

    Raised rather than degraded: an absent section does not remove a figure from
    the payload, only the caveat that stops the judge over-claiming about it.
    """


def _section(metadata, name: str, **values) -> str:
    """Return one metadata section, formatted.

    Args:
        metadata: The VerdictMetadata carrying the section texts.
        name: Field name on metadata.
        **values: Placeholder values for str.format.

    Returns:
        The formatted section text.

    Raises:
        MetadataError: If the section is empty or its placeholders do not resolve.
    """
    template = getattr(metadata, name, "")
    if not template.strip():
        raise MetadataError(
            f"verdict_metadata.{name} is empty in config.json, but this payload needs it."
        )
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise MetadataError(
            f"verdict_metadata.{name} in config.json could not be formatted ({exc})."
        ) from exc


def _build_metadata(
    rows: list[TestCaseResult],
    pooled: dict[str, CandidatePerformance],
    judge_config: JudgeConfig,
    profile_shown: bool,
) -> str:
    """Assemble the payload header from the sections this payload actually needs.

    Emitting the whole thing unconditionally meant a project with no reasoning
    analysis read several paragraphs on how to interpret a reasoning profile it
    would never see, and a project using a single judge type was told its scores
    came from different judges.

    Args:
        rows: Test case results, read for the judge types in use.
        pooled: Per-candidate pooled records, read for whether any cost exists.
        judge_config: JudgeConfig, for the repetition count.
        profile_shown: Whether the reasoning profile table is being emitted.

    Returns:
        The metadata block, sections separated by blank lines.

    Raises:
        MetadataError: If a needed section is missing or malformed.
    """
    metadata = get_app_config().verdict_metadata
    parts = [
        _section(metadata, "header"),
        _section(metadata, "figures", repetitions=judge_config.repetitions),
        _section(metadata, "score_scales"),
    ]

    if any(perf.reasoning_cost_per_point >= 0 for perf in pooled.values()):
        parts.append(_section(metadata, "cost_per_point"))

    if profile_shown:
        parts.append(_section(metadata, "reasoning_profile"))

    judge_types = {row.judge_type for row in rows}
    legend = [_section(metadata, "judge_types_intro")]
    for judge_type, field_name in (
        (JUDGE_TYPE_LLM, "judge_llm"),
        (JUDGE_TYPE_SIMILARITY, "judge_similarity"),
        (JUDGE_TYPE_ASSERT, "judge_assert"),
    ):
        if judge_type in judge_types:
            legend.append(_section(metadata, field_name))
    if len(judge_types) > 1:
        legend.append(_section(metadata, "judge_types_mixed"))
    parts.append("\n".join(legend))

    parts.append(_build_global_criteria_legend(metadata, judge_config.global_criteria))

    return "\n\n".join(parts) + "\n"


_GLOBAL_CRITERIA_FIELDS = {
    JUDGE_TYPE_LLM: ("global_criteria_llm", "llm_judge_criteria"),
    JUDGE_TYPE_SIMILARITY: ("global_criteria_similarity", "similarity_criteria"),
    JUDGE_TYPE_ASSERT: ("global_criteria_assert", "assert_criteria"),
}


def _build_global_criteria_legend(metadata, gc: GlobalCriteria) -> str:
    """Render how the global score was produced, plus the criteria text itself.

    Styled after the per-test judge_type legend just above it (judge_types_intro
    / judge_llm / ...): a code-owned "how this figure is produced" line from
    config.json, immediately followed by the project's own rubric text via
    {criteria_text}. It is the last thing _build_metadata emits, and the whole
    metadata block it belongs to is appended to the verdict judge's SYSTEM
    prompt (see verdict._run_verdict_call), not placed ahead of the data in the
    user payload — which is what let it read as an instruction that outranks
    the data, when it is exactly as much "the rubric" as the per-test-case
    EVALUATION CRITERIA already is.

    Args:
        metadata: VerdictMetadata carrying the section texts.
        gc: The project's resolved GlobalCriteria.

    Returns:
        The intro line plus the mode-specific body.

    Raises:
        MetadataError: If the needed section is missing or malformed.
    """
    if gc.mode not in _GLOBAL_CRITERIA_FIELDS:
        body = _section(metadata, "global_criteria_none")
    else:
        field_name, criteria_attr = _GLOBAL_CRITERIA_FIELDS[gc.mode]
        criteria_text = getattr(gc, criteria_attr).strip() or "(none set)"
        # Every global_criteria_* section puts "  " ahead of {criteria_text}, so a
        # multi-line criteria (e.g. a numbered list) needs its own continuation
        # lines re-indented to match — otherwise only the first line lines up.
        criteria_text = criteria_text.replace("\n", "\n  ")
        body = _section(metadata, field_name, criteria_text=criteria_text)
    return _section(metadata, "global_criteria_intro") + "\n" + body
