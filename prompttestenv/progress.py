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
