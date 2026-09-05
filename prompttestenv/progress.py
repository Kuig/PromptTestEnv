from __future__ import annotations

import hashlib
import json
import os


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

# The two placeholder payloads a run stores in place of a real model answer.
# They are not error messages written for a human to read: they are the values
# --retry-errors matches on, they are what the judge ends up scoring, and they
# are what the report renders, so each has exactly one definition.
GEN_TIMEOUT_TEXT = "⛔ [TIMEOUT EXCEEDED]"
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
        True only for the timeout placeholder. An empty response is a real
        measurement about the model, not a failure to obtain one.
    """
    return event.get("output") == GEN_TIMEOUT_TEXT


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


def failed_gen_keys(gen_events: dict) -> set[tuple[str, str, int]]:
    """Select the generation keys whose stored answer is a placeholder.

    Takes the plain dict rather than a ProgressState so this module keeps its
    zero dependencies (models.py imports this one, not the other way round).

    Args:
        gen_events: ProgressState.gen_events, keyed by (candidate, test, rep).

    Returns:
        The subset of keys worth generating again.
    """
    return {key for key, event in gen_events.items() if is_failed_gen(event)}


def failed_eval_keys(eval_events: dict) -> set[tuple[str, str, int]]:
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


def append_event(project_dir: str, event: dict) -> None:
    """Append a new JSON event to the progress.jsonl file."""
    progress_file = os.path.join(project_dir, "progress.jsonl")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
