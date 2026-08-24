from __future__ import annotations

import json
import os
from pathlib import Path

from unified_ai_client import load_secrets

_PROJECT_ROOT = Path(__file__).parent.parent

# CONFIG_FILE kept for backwards compatibility — still points to secrets.json
CONFIG_FILE = str(_PROJECT_ROOT / "secrets.json")


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
        judge_data = {
            "repetitions": 5,
            "repetition_delay_seconds": 2,
            "evaluation_delay_seconds": 2,
            "max_response_timeout_seconds": 240,
            "pass_media_to_judge": False,
            "group_verdicts": False,
            "reasoning_analysis": True,
            "test_judge": {
                "provider": "google",
                "model": "gemini-3-flash-preview",
                "temperature": 0.2,
                "disable_safety": True,
                "thinking": "default",
                "evaluation_system_prompt": (
                    "You are an impartial AI judge. Evaluate the candidate's response "
                    "against the specified criteria. Output MUST be JSON with fields: "
                    "score (1-10), reasoning."
                ),
                "evaluation_template": (
                    "[USER INPUT]\n{user_prompt}\n\n"
                    "[CANDIDATE RESPONSE]\n{candidate_response}\n\n"
                    "[CRITERIA]\n{criteria}\n\n---\n\n"
                    "Analyze the response and provide the JSON output. Score: 1-10."
                ),
            },
            "similarity_judge": {
                "provider": "ollama",
                "model": "bge-m3"
            },
            "reasoning_judge": {
                "provider": "google",
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "thinking": False,
                "reasoning_system_prompt": (
                    "You are a cognitive process analyst. Your task is to analyze the internal "
                    "reasoning traces of AI models and extract structured information from them. "
                    "Always respond with valid JSON only — no prose, no markdown code blocks."
                ),
                "segmentation_template": (
                    "You will receive the internal reasoning trace of an AI model. "
                    "Segment this trace verbatim into exactly 4 categories. "
                    "Every single word of the reasoning MUST appear in exactly one "
                    "category — do not omit, paraphrase, or summarize anything.\n\n"
                    "Categories:\n"
                    "- interpretation: The part where the model analyzes and rephrases the user's "
                    "request to ensure it understood the problem correctly.\n"
                    "- planning: The part where the model evaluates its approach, tools, or "
                    "strategy before solving the problem.\n"
                    "- pure_reasoning: The actual problem-solving: recall of facts, logical "
                    "deduction, alternative options, self-correction, and computation.\n"
                    "- output_formulation: The part dedicated exclusively to deciding how to "
                    "present the final answer (format, markdown, structure, tone).\n\n"
                    "Output MUST be a JSON object with exactly these 4 string fields: "
                    "interpretation, planning, pure_reasoning, output_formulation.\n\n"
                    "REASONING TRACE:\n{reasoning_text}"
                ),
                "metrics_template": (
                    "Analyze the following AI reasoning trace and the final response it produced, "
                    "then extract exactly 3 numeric metrics.\n\n"
                    "Metrics to extract:\n"
                    "- alt_path (integer): The number of distinct alternative solution paths or "
                    "approaches the model explicitly considered before choosing one.\n"
                    "- autocorrect (integer): The number of times the model explicitly corrected "
                    "itself (e.g., 'Wait, that\\'s wrong', 'Let me reconsider', 'Actually...').\n"
                    "- alignment_score (integer 1-10): How well the FINAL RESPONSE reflects and "
                    "builds upon the reasoning process. 10 = the response directly implements the "
                    "conclusions of the reasoning; 1 = the response ignores or contradicts the reasoning.\n\n"
                    "Output MUST be a JSON object with exactly these 3 fields: "
                    "alt_path, autocorrect, alignment_score.\n\n"
                    "REASONING TRACE:\n{reasoning_text}\n\n"
                    "FINAL RESPONSE:\n{candidate_response}"
                ),
            },
            "verdict_judge": {
                "provider": "google",
                "model": "gemini-3-flash-preview",
                "temperature": 0.2,
                "disable_safety": True,
                "thinking": "default",
                "verdict_system_prompt": (
                    "You are a Senior AI Analyst. Your task is to analyze the results of a comparative benchmark "
                    "across different models given a set of criteria, metrics and test results. "
                    "Be direct and concise. Do not use pleasantries, only provide the requested report."
                ),
                "verdict_template": (
                    "# GLOBAL EVALUATION CRITERIA:\n"
                    "{global_criteria}\n\n"
                    "# TESTS RESULTS:\n"
                    "[Note: For each test case, 'Notes' are evaluations from the single best execution (repetition) of the candidate. "
                    "Scores, execution time and token counts are averaged across all repetitions.]\n\n"
                    "{summary_data}\n\n"
                    "---\n\n"
                    "# ANALYSIS INSTRUCTIONS:\n"
                    "Analyze the results and create a Markdown report:\n"
                    "1. Identify the winning model.\n"
                    "2. Briefly describe each model's behavior.\n"
                    "3. Highlight strengths and weaknesses vs the Baseline.\n"
                    "4. Suggest use cases for each model.\n"
                    "5. Note any trends or special cases.\n"
                    "6. Write a concluding verdict."
                ),
                "global_verdict_template": (
                    "# GROUP VERDICTS:\n"
                    "{group_verdicts_data}\n\n"
                    "---\n\n"
                    "# ANALYSIS INSTRUCTIONS:\n"
                    "Analyze the verdicts from all groups and provide a final overall conclusion.\n"
                    "1. Identify the winning model.\n"
                    "2. Briefly describe each model's behavior.\n"
                    "3. Highlight strengths and weaknesses vs the Baseline.\n"
                    "4. Suggest use cases for each model.\n"
                    "5. Note any trends or special cases.\n"
                    "6. Write a concluding verdict."
                ),
            },
        }
        with open(judge_file, "w", encoding="utf-8") as f:
            json.dump(judge_data, f, indent=4)

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
            f.write(
                "You are a grumpy but helpful pirate working in a call center. "
                "Use typical pirate idioms like 'Arrrr' and 'Ahoy matey'."
            )
        candidates_data = [
            {
                "name": "Baseline (Flash 2.5)",
                "provider": "google",
                "model": "gemini-2.5-flash",
                "temperature": 0.5,
                "disable_safety": False,
            },
            {
                "name": "Thinking (Flash 2.5 Budget)",
                "provider": "google",
                "model": "gemini-2.5-flash",
                "temperature": 0.7,
                "disable_safety": True,
                "thinking": True,
            },
            {
                "name": "Pirate Call Center (Flash 3.0 Thinking)",
                "provider": "google",
                "model": "gemini-3-flash-preview",
                "temperature": 0.8,
                "disable_safety": True,
                "thinking": True,
                "system_prompt_file": default_prompt_filename,
            },
            {
                "name": "Local Gemma (Ollama)",
                "provider": "ollama",
                "model": "gemma4:e2b",
                "temperature": 0.5,
            },
        ]
        with open(cand_file, "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, indent=4)

    # 4. Test Cases
    test_file = os.path.join(project_dir, "test_cases.json")
    if not os.path.exists(test_file):
        sample_txt_path = os.path.join(project_dir, "test_files", "sample.txt")
        if not os.path.exists(sample_txt_path):
            with open(sample_txt_path, "w", encoding="utf-8") as f:
                f.write(
                    "This is a confidential document.\n\n"
                    "The Q3 revenue was 150,000 euros, with a 15% increase compared to the previous quarter."
                )
        test_data = [
            {
                "id": "customer_email",
                "prompt": "Write a short apology email.",
                "criteria": "Empathetic tone. Do not promise refunds.",
                "group": "Default group",
                "judge_type": "llm-judge"
            },
            {
                "id": "file_analysis",
                "prompt": "What was the Q3 revenue and its percentage change?",
                "criteria": "The Q3 revenue was 150,000 euros, with a 15% increase.",
                "file": "test_files/sample.txt",
                "group": "Default group",
                "judge_type": "similarity"
            },
            {
                "id": "comma_count",
                "prompt": "List exactly three European capitals, separated by commas.",
                "criteria": "s: (10, 'Correct comma count') if s.count(',') == 2 else (1, f'Expected 2 commas, found {s.count(\",\")}')",
                "group": "Default group",
                "judge_type": "assert"
            }
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4)

    # 5. Global Criteria (JSON)
    global_file = os.path.join(project_dir, "global_criteria.json")
    if not os.path.exists(global_file):
        global_data = {
            "mode": "llm-judge",
            "llm_judge_criteria": "1. Courtesy: The language must always be polite.\n2. Safety: Do not generate harmful content.",
            "similarity_criteria": "The response should be a clear, concise and informative answer.",
            "assert_criteria": "s: (10, 'OK') if len(s) > 10 else (1, 'Too short')"
        }
        with open(global_file, "w", encoding="utf-8") as f:
            json.dump(global_data, f, indent=4)

