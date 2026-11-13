from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import prompttestenv.logger as logger


# Keys of judge_config.json that do NOT invalidate a run when they change.
# Reasoning analysis is a post-hoc pass over traces already stored in
# progress.jsonl, so retuning it must not force a re-generation: hashing the
# whole file made every prompt tweak cost a full re-run of every candidate.
_UNHASHED_JUDGE_KEYS = ("reasoning_analysis", "reasoning_judge")

# The config files that make up the run hash, in the order they are fed to the
# hasher. Changing this order changes every existing project's hash.
HASHED_FILENAMES = (
    "candidates.json",
    "judge_config.json",
    "test_cases.json",
    "global_criteria.json",
)

# What a file that does not exist contributes to the hash.
MISSING_FILE_BYTES = b"missing"

# The distribution whose version the meta line records. Not hashed: a client
# upgrade can change what is measured (defaults it stops sending, options it
# starts honouring) without any project file changing, so resuming across one
# is allowed but announced.
CLIENT_DISTRIBUTION = "unified-ai-client"

# The placeholder payloads a run stores in place of a real model answer. They
# are not error messages written for a human to read: they are the values
# --retry-errors matches on, they are what the judge ends up scoring, and they
# are what the report renders, so each has exactly one definition.
#
# GEN_ERROR_TEXT is distinct from GEN_TIMEOUT_TEXT rather than reusing it,
# because the two say different things about the model: a timeout is a fact
# about how long it took, a provider error is not a fact about the model at
# all. Reusing one placeholder would make the log claim "timeout" for a 429.
# The exception's own message goes in the event's "error" field, never in the
# payload, so `output` stays comparable by equality.
GEN_TIMEOUT_TEXT = "⛔ [TIMEOUT EXCEEDED]"
GEN_ERROR_TEXT = "⛔ [GENERATION FAILED]"
JUDGE_TIMEOUT_TEXT = "⛔ [JUDGE TIMEOUT EXCEEDED]"

# Every -1 reason the framework writes on its own initiative starts with one of
# these. "Error" is deliberately not suffixed with a colon: it has to cover
# "Error: ...", the dispatcher's bare "Error", and "Error in evaluation
# template: ...", which an "Error:" prefix would miss.
_EVAL_FAILURE_PREFIXES = ("Error", "LLM evaluation failed:", JUDGE_TIMEOUT_TEXT)


def _side_failed(score: object, text: object) -> bool:
    """Report whether one side of an eval event is a framework failure.

    Args:
        score: The stored score for this side ("score" or "global_score").
        text: The reasoning stored alongside it.

    Returns:
        True when the score is the -1 sentinel AND the reasoning is one the
        framework wrote about its own failure.
    """
    return score == -1 and str(text or "").startswith(_EVAL_FAILURE_PREFIXES)


def is_failed_gen(event: dict) -> bool:
    """Report whether a "gen" event holds a placeholder instead of an answer.

    Args:
        event: A "gen" event read back from progress.jsonl.

    Returns:
        True for either framework placeholder, the timeout and the provider
        error. An empty response is a real measurement about the model, not a
        failure to obtain one, so it is never matched here.
    """
    return event.get("output") in (GEN_TIMEOUT_TEXT, GEN_ERROR_TEXT)


def is_failed_eval(event: dict) -> bool:
    """Report whether an "eval" event holds a framework failure.

    A -1 score alone is not enough to tell. ``_evaluate_llm_judge`` and
    ``_evaluate_similarity`` both clamp their result into 1-10 and only reach
    -1 through their own error paths, but ``_evaluate_assert`` deliberately
    does NOT clamp: an assert lambda returning -1 is the project author saying
    "not measured/not applicable" for that response, and re-running it would
    both ignore their intent and never converge. The prefix list exists to
    spare exactly that case, and to leave a project with
    ``global_criteria.mode: "none"`` alone, since it stores -1 with "N/A"
    forever by design.

    Either side failing makes the event retryable: without that, a project
    whose global judge hit a transient error could only be repaired with
    --force-restart, which means re-buying every candidate response. The cost
    is that ``evaluate_with_judge`` runs both sides, so retrying a global-only
    failure re-judges the task too and an already-good task score can move.

    Args:
        event: An "eval" event read back from progress.jsonl.

    Returns:
        True when either the task or the global side is a framework failure.
    """
    return (
        _side_failed(event.get("score"), event.get("reason"))
        or _side_failed(event.get("global_score"), event.get("g_reason"))
    )


def failed_gen_keys(gen_events: dict) -> set[RunKey]:
    """Select the generation keys whose stored answer is a placeholder.

    Takes the plain dict rather than a ProgressState so this signature works
    from a literal too, since this module has no dependency on models.py to
    make ProgressState a natural argument type here.

    Args:
        gen_events: ProgressState.gen_events, keyed by (candidate, test, rep).

    Returns:
        The subset of keys worth generating again.
    """
    return {key for key, event in gen_events.items() if is_failed_gen(event)}


def failed_eval_keys(eval_events: dict) -> set[RunKey]:
    """Select the evaluation keys whose stored score is a framework failure.

    Args:
        eval_events: ProgressState.eval_events, keyed by (candidate, test, rep).

    Returns:
        The subset of keys worth judging again.
    """
    return {key for key, event in eval_events.items() if is_failed_eval(event)}


def hashable_bytes(filename: str, raw: bytes | None) -> bytes:
    """Return how one config file's content contributes to the run hash.

    Pure: takes content rather than a path, so a caller holding unsaved edits
    can ask what the hash *would* become. See config_hash_from_bytes().

    Only judge_config.json gets special treatment: its reasoning-analysis
    settings are stripped and the remainder is re-serialised canonically, so
    that formatting-only edits behave the same as they do for the other files
    (which are hashed raw).

    Args:
        filename: Bare filename, which selects the judge_config.json rule.
        raw: The file's bytes, or None when the file does not exist.

    Returns:
        The byte string to feed the hasher.
    """
    if raw is None:
        return MISSING_FILE_BYTES
    if filename != "judge_config.json":
        return raw
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    if not isinstance(data, dict):
        return raw
    stripped = {k: v for k, v in data.items() if k not in _UNHASHED_JUDGE_KEYS}
    return json.dumps(stripped, sort_keys=True, ensure_ascii=False).encode("utf-8")


def config_hash_from_bytes(contents: dict[str, bytes | None]) -> str:
    """Compute the run hash from in-memory config content.

    Lets an editor predict whether a pending save would invalidate an existing
    progress.jsonl, without writing anything first.

    Args:
        contents: Maps each name in HASHED_FILENAMES to its bytes, or to None
            for "this file does not exist". Missing keys are treated as None.

    Returns:
        The MD5 hex digest, directly comparable with calculate_config_hash().
    """
    hasher = hashlib.md5()
    for name in HASHED_FILENAMES:
        hasher.update(hashable_bytes(name, contents.get(name)))
    return hasher.hexdigest()


def _read_bytes_or_none(file_path: str) -> bytes | None:
    """Read a file's bytes, or return None when it does not exist."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return f.read()


def calculate_config_hash(project_dir: str) -> str:
    """Calculate the MD5 hash of configuration files to detect changes between runs."""
    return config_hash_from_bytes({
        name: _read_bytes_or_none(os.path.join(project_dir, name))
        for name in HASHED_FILENAMES
    })


def read_stored_hash(project_dir: str) -> str | None:
    """Read the config hash recorded on progress.jsonl's first line.

    Deliberately does not go through ProgressState.load(): that renames the log
    to .bak on a mismatch and creates it when absent, neither of which a caller
    merely *asking* about the stored hash should ever trigger.

    Args:
        project_dir: Path to the benchmark project directory.

    Returns:
        The stored hash, or None when there is no log or no readable meta line.
    """
    progress_file = os.path.join(project_dir, "progress.jsonl")
    if not os.path.exists(progress_file):
        return None
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
        meta = json.loads(first_line)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or meta.get("type") != "meta":
        return None
    return meta.get("config_hash")


def client_version() -> str | None:
    """The installed UnifiedAiClient version, or None if it is not installed."""
    try:
        return importlib.metadata.version(CLIENT_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _warn_on_client_change(stored: str | None) -> None:
    """Announce a log written under a different UnifiedAiClient version.

    A log from before the version was recorded carries none, and says nothing
    either way, so it is left alone.
    """
    current = client_version()
    if stored is not None and stored != current:
        logger.log_warning(
            f"progress.jsonl was started with UnifiedAiClient {stored}, now "
            f"{current}: results already logged may not be comparable with new "
            f"ones. Use --force-restart to measure everything under {current}."
        )


def append_event(project_dir: str, event: dict) -> None:
    """Append a new JSON event to the progress.jsonl file."""
    progress_file = os.path.join(project_dir, "progress.jsonl")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


class RunKey(NamedTuple):
    """Identifies one repetition: a candidate x test_case x repetition triple.

    Indexes every event in progress.jsonl and every set/dict built from it
    (completed_gen, gen_events, stale_eval_keys, ...). A NamedTuple rather
    than a dataclass because it must behave exactly like the plain 3-tuple
    it replaces: equality, hashing, positional unpacking and use as a dict
    key are all structural, so a literal (cand, test_id, rep) still matches
    a RunKey holding the same three values, and nothing written to
    progress.jsonl changes (the key itself is never serialised, only the
    fields that compose it).
    """

    candidate: str
    test_id: str
    repetition: int


@dataclass(frozen=True)
class ProgressState:
    """Read-only snapshot of a project's resume state, built once from progress.jsonl.

    ``events`` intentionally stays a list of raw dicts rather than typed event
    dataclasses: it is a direct in-memory mirror of heterogeneous JSONL log
    lines (differently shaped per ``type``), which is exactly the "dynamic/opaque
    data" case dict is meant for.

    ``gen_events``/``eval_events``/``reasoning_events`` index the "gen"/"eval"/"reasoning" event dicts
    already in ``events`` by their ``(cand_id, test_id, rep)`` key — built in
    the same pass as ``completed_gen``/``completed_eval`` — so a resumed key
    can be looked up in O(1) instead of scanning ``events`` linearly.

    The three ``stale_*`` fields answer "does the log contradict itself?", and
    they are computed here because this is the only place that knows the ORDER
    of the lines, which is the whole evidence. The log is append-only, so a
    retry supersedes a failed step by appending a corrected event: an "eval"
    (or "reasoning") whose key has a NEWER "gen" line after it was therefore
    computed against a response that is no longer the stored one, and a
    "verdict" with "gen"/"eval" lines after it describes results that have
    since been replaced. Both states are what an interrupted --retry-errors
    run leaves behind, and neither is visible to the failure predicates above,
    which only see the surviving values and not their positions.

    The property is self-healing: once the phase re-appends its event, that
    event's line follows the "gen" again and the key stops being stale on the
    next load, so nothing has to be cleared.
    """

    hash_match: bool = True
    completed_gen: set[RunKey] = field(default_factory=set)
    completed_eval: set[RunKey] = field(default_factory=set)
    events: list[dict] = field(default_factory=list)
    verdict: str | None = None
    last_hash: str | None = None
    gen_events: dict[RunKey, dict] = field(default_factory=dict)
    eval_events: dict[RunKey, dict] = field(default_factory=dict)
    reasoning_events: dict[RunKey, dict] = field(default_factory=dict)
    stale_eval_keys: frozenset[RunKey] = frozenset()
    stale_reasoning_keys: frozenset[RunKey] = frozenset()
    stale_verdict: bool = False

    @classmethod
    def load(
        cls,
        project_dir: str | Path,
        force_restart: bool = False,
        readonly: bool = False,
    ) -> ProgressState:
        """Initialize or read a project's progress.jsonl resume log.

        If force_restart is True, or the stored config hash no longer matches
        the current config files, the existing progress.jsonl is renamed to
        progress.jsonl.bak (or, if force_restart, simply removed) instead of
        being read.

        ``readonly`` suppresses all of that: nothing is renamed, removed or
        created, and a hash mismatch is reported through ``hash_match`` with
        the events still loaded. Commands that only consume the log (rendering
        a report, analysing stored traces) must use it, since they do not
        depend on the config still matching and must not destroy the log they
        were asked to read.

        Args:
            project_dir: Path to the benchmark project directory.
            force_restart: If True, discard any existing progress.jsonl.
            readonly: If True, never write to or rename the log.

        Returns:
            A ProgressState snapshot. Outside readonly mode, a stored config
            hash that does not match the current one leaves ``hash_match``
            False and every other field at its default (empty/None), so
            callers must check ``hash_match`` before trusting the snapshot.
        """
        project_dir = str(project_dir)
        progress_file = os.path.join(project_dir, "progress.jsonl")
        current_hash = calculate_config_hash(project_dir)

        if force_restart and not readonly and os.path.exists(progress_file):
            os.remove(progress_file)

        hash_match = True
        completed_gen: set[RunKey] = set()
        completed_eval: set[RunKey] = set()
        events: list[dict] = []
        verdict: str | None = None
        last_hash: str | None = None
        gen_events: dict[RunKey, dict] = {}
        eval_events: dict[RunKey, dict] = {}
        reasoning_events: dict[RunKey, dict] = {}
        # Line ordinals of the NEWEST line of each kind, per key. Only ever
        # compared with each other, so the gaps an unparseable line leaves in
        # the numbering are harmless. See the class docstring for what they
        # are evidence of.
        gen_line: dict[RunKey, int] = {}
        eval_line: dict[RunKey, int] = {}
        reasoning_line: dict[RunKey, int] = {}
        verdict_line: int | None = None
        last_result_line = -1

        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if lines:
                try:
                    meta = json.loads(lines[0])
                    if meta.get("type") == "meta":
                        last_hash = meta.get("config_hash")
                        _warn_on_client_change(meta.get("client_version"))
                        if last_hash != current_hash:
                            hash_match = False
                            if not readonly:
                                logger.log_warning(
                                    "Configuration hash mismatch! "
                                    "Renaming invalid progress file to: progress.jsonl.bak"
                                )
                                os.replace(progress_file, progress_file + ".bak")
                                return cls(hash_match=False, last_hash=last_hash)
                except json.JSONDecodeError:
                    if readonly:
                        return cls(hash_match=False)
                    logger.log_warning(
                        "Corrupted progress file detected! Renaming to: progress.jsonl.bak"
                    )
                    os.replace(progress_file, progress_file + ".bak")
                    return cls(hash_match=False)

                for position, line in enumerate(lines[1:]):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict) or "type" not in event:
                            # Valid JSON of the wrong shape. Without this the
                            # indexing below raised TypeError out of load(),
                            # and since every command goes through here a
                            # single odd line made the project unreadable by
                            # all of them, render included.
                            continue
                        events.append(event)
                        if event["type"] == "gen":
                            key = RunKey(event["cand_id"], event["test_id"], event["rep"])
                            completed_gen.add(key)
                            gen_events[key] = event
                            gen_line[key] = position
                            last_result_line = position
                        elif event["type"] == "eval":
                            key = RunKey(event["cand_id"], event["test_id"], event["rep"])
                            completed_eval.add(key)
                            eval_events[key] = event
                            eval_line[key] = position
                            last_result_line = position
                        elif event["type"] == "reasoning":
                            key = RunKey(event["cand_id"], event["test_id"], event["rep"])
                            reasoning_events[key] = event
                            reasoning_line[key] = position
                        elif event["type"] == "verdict":
                            verdict = event["content"]
                            verdict_line = position
                    except (json.JSONDecodeError, TypeError, KeyError):
                        # Broken lines at the end of file (crash mid-write),
                        # and events missing a key this reader needs. Skipping
                        # one line costs at most one repetition, which resume
                        # then redoes; raising would cost the whole log.
                        pass

        if not readonly and not os.path.exists(progress_file):
            with open(progress_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "meta",
                    "config_hash": current_hash,
                    "client_version": client_version(),
                }) + "\n")

        # A key with a "gen" but no "eval" is not stale, it is simply unjudged,
        # and the evaluation phase already picks it up: staleness is only about
        # an event that exists and describes superseded text.
        stale_eval_keys = frozenset(
            key for key, line in eval_line.items() if gen_line.get(key, -1) > line
        )
        stale_reasoning_keys = frozenset(
            key for key, line in reasoning_line.items() if gen_line.get(key, -1) > line
        )
        # "reasoning" lines deliberately do not count towards last_result_line:
        # `analyze` is the cheap pass meant to be re-run over a finished project
        # whenever the schema is retuned, so letting it invalidate the verdict
        # would make retuning the measurement cost a verdict call every time.
        stale_verdict = verdict_line is not None and last_result_line > verdict_line

        return cls(
            hash_match=hash_match,
            completed_gen=completed_gen,
            completed_eval=completed_eval,
            events=events,
            verdict=verdict,
            last_hash=last_hash,
            gen_events=gen_events,
            eval_events=eval_events,
            reasoning_events=reasoning_events,
            stale_eval_keys=stale_eval_keys,
            stale_reasoning_keys=stale_reasoning_keys,
            stale_verdict=stale_verdict,
        )
