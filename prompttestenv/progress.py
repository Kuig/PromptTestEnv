from __future__ import annotations

import hashlib
import json
import os


# Keys of judge_config.json that do NOT invalidate a run when they change.
# Reasoning analysis is a post-hoc pass over traces already stored in
# progress.jsonl, so retuning it must not force a re-generation: hashing the
# whole file made every prompt tweak cost a full re-run of every candidate.
_UNHASHED_JUDGE_KEYS = ("reasoning_analysis", "reasoning_judge")


def _hashable_bytes(file_path: str) -> bytes:
    """Return the bytes of one config file as they contribute to the run hash.

    Only judge_config.json gets special treatment: its reasoning-analysis
    settings are stripped and the remainder is re-serialised canonically, so
    that formatting-only edits behave the same as they do for the other files
    (which are hashed raw).

    Args:
        file_path: Absolute path to the config file.

    Returns:
        The byte string to feed the hasher.
    """
    if not os.path.exists(file_path):
        return b"missing"
    with open(file_path, "rb") as f:
        raw = f.read()
    if os.path.basename(file_path) != "judge_config.json":
        return raw
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw
    if not isinstance(data, dict):
        return raw
    stripped = {k: v for k, v in data.items() if k not in _UNHASHED_JUDGE_KEYS}
    return json.dumps(stripped, sort_keys=True, ensure_ascii=False).encode("utf-8")


def calculate_config_hash(project_dir: str) -> str:
    """Calculate the MD5 hash of configuration files to detect changes between runs."""
    files_to_hash = [
        os.path.join(project_dir, "candidates.json"),
        os.path.join(project_dir, "judge_config.json"),
        os.path.join(project_dir, "test_cases.json"),
        os.path.join(project_dir, "global_criteria.json")
    ]
    hasher = hashlib.md5()
    for fp in files_to_hash:
        hasher.update(_hashable_bytes(fp))
    return hasher.hexdigest()


def append_event(project_dir: str, event: dict) -> None:
    """Append a new JSON event to the progress.jsonl file."""
    progress_file = os.path.join(project_dir, "progress.jsonl")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
