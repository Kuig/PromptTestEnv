from __future__ import annotations

import hashlib
import json
import os


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
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                hasher.update(f.read())
        else:
            hasher.update(b"missing")
    return hasher.hexdigest()


def append_event(project_dir: str, event: dict) -> None:
    """Append a new JSON event to the progress.jsonl file."""
    progress_file = os.path.join(project_dir, "progress.jsonl")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
