from __future__ import annotations

import importlib.resources
import json
import os
from pathlib import Path

from unified_ai_client import load_secrets

_PROJECT_ROOT = Path(__file__).parent.parent

# CONFIG_FILE kept for backwards compatibility — still points to secrets.json
CONFIG_FILE = str(_PROJECT_ROOT / "secrets.json")


def _load_template(filename: str) -> str:
    """Read a default-content resource file bundled under prompttestenv/templates/.

    Args:
        filename: Resource file name (e.g. "default_judge_config.json").

    Returns:
        The file's raw text content.
    """
    return importlib.resources.files("prompttestenv").joinpath(f"templates/{filename}").read_text(encoding="utf-8")


def get_api_key() -> str:
    """Load the Google AI API key from the project root secrets.json.

    Returns:
        The API key string.

    Raises:
        ValueError: If the key is missing or still has the placeholder value.
    """
    secrets = load_secrets(str(_PROJECT_ROOT))
    key = secrets.get("google_api_key")
    if not key or key == "INSERT_YOUR_API_KEY_HERE":
        raise ValueError(
            "Please provide a valid 'google_api_key' in secrets.json at the project root."
        )
    return key


def init_project(project_dir: str, custom_candidates: list[dict] | None = None) -> None:
    """Initialize the project directory structure for a new benchmark.

    Creates the project folder, subdirectories, and template configuration
    files (candidates.json, judge_config.json, test_cases.json,
    global_criteria.json). Skips files that already exist.

    Args:
        project_dir: Path to the benchmark project directory to create.
        custom_candidates: Optional list of candidate dicts. If provided,
            overrides the default candidates.json content. Candidates may
            include a 'system_prompt_text' key which is auto-saved to a file.
    """
    import prompttestenv.logger as logger

    os.makedirs(project_dir, exist_ok=True)
    sys_dir = os.path.join(project_dir, "system_prompts")
    os.makedirs(sys_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "test_files"), exist_ok=True)

    # 1. Root secrets.json — only create if missing; uses google_api_key key
    secrets_path = _PROJECT_ROOT / "secrets.json"
    if not secrets_path.exists():
        secrets_data = {"google_api_key": "INSERT_YOUR_API_KEY_HERE"}
        secrets_path.write_text(json.dumps(secrets_data, indent=4), encoding="utf-8")
        logger.log_warning(f"Created secrets.json at {secrets_path}. Please insert your API key.")

    # 2. Judge Configuration
    judge_file = os.path.join(project_dir, "judge_config.json")
    if not os.path.exists(judge_file):
        with open(judge_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_judge_config.json"))

    # 3. Candidates Configuration
    cand_file = os.path.join(project_dir, "candidates.json")
    if custom_candidates is not None:
        processed_candidates = []
        for i, cand in enumerate(custom_candidates):
            if "system_prompt_text" in cand:
                filename = f"custom_prompt_{i}.txt"
                filepath = os.path.join(sys_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cand["system_prompt_text"])
                new_cand = {k: v for k, v in cand.items() if k != "system_prompt_text"}
                new_cand["system_prompt_file"] = filename
                processed_candidates.append(new_cand)
            else:
                processed_candidates.append(cand)
        with open(cand_file, "w", encoding="utf-8") as f:
            json.dump(processed_candidates, f, indent=4)
    elif not os.path.exists(cand_file):
        default_prompt_filename = "pirate_prompt.txt"
        default_prompt_filepath = os.path.join(sys_dir, default_prompt_filename)
        with open(default_prompt_filepath, "w", encoding="utf-8") as f:
            f.write(_load_template("default_pirate_prompt.txt"))
        with open(cand_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_candidates.json"))

    # 4. Test Cases
    test_file = os.path.join(project_dir, "test_cases.json")
    if not os.path.exists(test_file):
        sample_txt_path = os.path.join(project_dir, "test_files", "sample.txt")
        if not os.path.exists(sample_txt_path):
            with open(sample_txt_path, "w", encoding="utf-8") as f:
                f.write(_load_template("default_sample.txt"))
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_test_cases.json"))

    # 5. Global Criteria (JSON)
    global_file = os.path.join(project_dir, "global_criteria.json")
    if not os.path.exists(global_file):
        with open(global_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_global_criteria.json"))

