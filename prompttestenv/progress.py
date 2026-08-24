from __future__ import annotations

import hashlib
import json
import os

import prompttestenv.logger as logger

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


def init_progress(project_dir: str, force_restart: bool = False) -> dict:
    """
    Initializes the progress.jsonl file or reads it if it exists.
    If force_restart is True or the hash doesn't match, it deletes the file (or returns the error).
    Returns a dictionary with events grouped to facilitate in-memory lookups.
    """
    progress_file = os.path.join(project_dir, "progress.jsonl")
    current_hash = calculate_config_hash(project_dir)
    
    state = {
        "hash_match": True,
        "completed_gen": set(),  # set of tuples (cand_id, test_id, rep)
        "completed_eval": set(), # set of tuples (cand_id, test_id, rep)
        "events": [],            # list of all raw logs
        "verdict": None,
        "last_hash": None
    }
    
    if force_restart and os.path.exists(progress_file):
        os.remove(progress_file)
        
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        if lines:
            try:
                meta = json.loads(lines[0])
                if meta.get("type") == "meta":
                    saved_hash = meta.get("config_hash")
                    state["last_hash"] = saved_hash
                    if saved_hash != current_hash:
                        logger.log_warning(
                            "Configuration hash mismatch! "
                            "Renaming invalid progress file to: progress.jsonl.bak"
                        )
                        os.replace(progress_file, progress_file + ".bak")
                        state["hash_match"] = False
                        return state
            except json.JSONDecodeError:
                logger.log_warning(
                    "Corrupted progress file detected! Renaming to: progress.jsonl.bak"
                )
                os.replace(progress_file, progress_file + ".bak")
                state["hash_match"] = False
                return state

            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                try:
                    event = json.loads(line)
                    state["events"].append(event)
                    if event["type"] == "gen":
                        key = (event["cand_id"], event["test_id"], event["rep"])
                        state["completed_gen"].add(key)
                    elif event["type"] == "eval":
                        key = (event["cand_id"], event["test_id"], event["rep"])
                        state["completed_eval"].add(key)
                    elif event["type"] == "verdict":
                        state["verdict"] = event["content"]
                except json.JSONDecodeError:
                    pass # ignore broken lines at the end of file (crash mid-write)

    if not os.path.exists(progress_file):
        with open(progress_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", "config_hash": current_hash}) + "\n")

    return state


def append_event(project_dir: str, event: dict) -> None:
    """Append a new JSON event to the progress.jsonl file."""
    progress_file = os.path.join(project_dir, "progress.jsonl")
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
