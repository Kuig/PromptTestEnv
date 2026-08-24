# PromptTestEnv

LLM prompt benchmarking tool — part of the `_UnifyTools` suite.

Given a **project directory** containing candidate configurations and test cases, PromptTestEnv runs each prompt candidate against one or more LLM models (with optional multimodal attachments), evaluates responses with a judge LLM, and generates comparative HTML or Markdown reports. Resume is supported via `progress.jsonl`.

> 💡 **Tip:** For a comprehensive guide on creating tests, configuring options, and using the framework for alternative use cases (like bulk document processing), see the [HOWTO Guide](HOWTO.md).

---

## Installation

```powershell
cd D:\Progetti\IA\PromptTestEnv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e D:\Progetti\IA\UnifiedAiClient
pip install mcp streamlit
```

Configure your API key in `secrets.json` at the project root:

```json
{
    "google_api_key": "YOUR_KEY_HERE"
}
```

---

## Usage

### CLI

```powershell
# Initialize a new benchmark project
prompttestenv init Projects/MyBenchmark

# Run the benchmark
prompttestenv run Projects/MyBenchmark --output-mode html

# Re-run from scratch (ignore progress)
prompttestenv run Projects/MyBenchmark --force-restart

# Regenerate report from existing progress without re-running
prompttestenv render Projects/MyBenchmark

# Start the MCP server (stdio)
prompttestenv mcp

# Launch the Streamlit GUI
prompttestenv gui
```

### Streamlit GUI

```powershell
prompttestenv gui
```

### As a Library

```python
from prompttestenv.config import init_project, load_candidates
from prompttestenv.runner import run_project

init_project("Projects/MyBenchmark")
result = run_project("Projects/MyBenchmark", output_mode="html")
print(result)
```

### MCP (Claude Desktop / AI agents)

Tools exposed:

| Tool | Description |
|---|---|
| `prompttest_init_project` | Initialize a new benchmark project directory |
| `prompttest_run_project` | Run a benchmark project |
| `prompttest_get_results` | Regenerate report from existing progress |

---

## Project Structure

```
Projects/<benchmark>/
├── candidates.json         ← LLM candidates to compare
├── judge_config.json       ← Judge LLM configuration + templates
├── test_cases.json         ← Prompts, evaluation criteria and evaluation modes
├── global_criteria.json    ← Structured global criteria JSON (with mode selector)
├── system_prompts/         ← Optional system prompt files
├── test_files/             ← Attachments (images, text files)
└── progress.jsonl          ← Resume-safe run log
```

### `candidates.json` example

```json
[
    {
        "name": "Baseline (Flash 2.5)",
        "provider": "google",
        "model": "gemini-2.5-flash",
        "temperature": 0.5
    },
    {
        "name": "Local Gemma (Ollama)",
        "provider": "ollama",
        "model": "gemma4:e2b",
        "temperature": 0.5
    }
]
```

### `test_cases.json` example

```json
[
    {
        "id": "email",
        "group": "Writing tasks",
        "prompt": "Write a short apology email.",
        "criteria": "Empathetic tone. Do not promise refunds.",
        "judge_type": "llm-judge"
    },
    {
        "id": "file_analysis",
        "group": "Data extraction",
        "prompt": "What was the Q3 revenue?",
        "criteria": "The Q3 revenue was 150,000 euros, with a 15% increase.",
        "file": "test_files/report.txt",
        "judge_type": "similarity"
    },
    {
        "id": "comma_count",
        "group": "Syntax verification",
        "prompt": "List exactly three European capitals, separated by commas.",
        "criteria": "s: (10, 'Correct comma count') if s.count(',') == 2 else (1, f'Expected 2 commas, found {s.count(\",\")}')",
        "judge_type": "assert"
    }
]
```

---

## Configuration

| File | Purpose |
|---|---|
| `secrets.json` | `google_api_key` for Google AI. Copy `secrets.json.example` to get started. |

> **Note:** PromptTestEnv intentionally has no root-level `config.json`. Unlike other `_UnifyTools` projects, every benchmark under `Projects/<name>/` is fully self-contained (`candidates.json`, `judge_config.json`, `test_cases.json`, `global_criteria.json`) — there is no cross-project provider/connection setting that would belong in a shared root config.

### Resume Policy

Each run writes an append-only `progress.jsonl` inside the project directory. On the next `run`, PromptTestEnv computes an MD5 hash of `candidates.json`, `judge_config.json`, and `test_cases.json` and compares it against the hash stored in the first line of `progress.jsonl`:

- **Hash matches**: already-completed generation/evaluation steps (keyed by `candidate × test × repetition`) are skipped and restored from the log.
- **Hash mismatch**: the existing `progress.jsonl` is renamed to `progress.jsonl.bak` and the run refuses to resume silently — pass `--force-restart` to discard it and start clean, or restore the original config files to resume as before.

## Codebase Layout

The python package `prompttestenv` is organized as follows:

- `runner.py`: High-level orchestration for executing benchmarks (`run_project`) and rendering reports (`render_from_progress`).
- `generation.py`: Logic for Phase 1 (candidate response generation).
- `evaluator.py`: Orchestration logic for Phase 2 (judge evaluation progress and concurrent calls).
- `test_judge.py`: Logic for calling the judge LLM to evaluate individual test cases.
- `verdict.py`: Logic for generating aggregate group verdicts and global comparative conclusions.
- `reporting.py`: HTML and Markdown report compilation.
- `models.py`: Domain dataclasses (`Candidate`, `TestCase`, `JudgeConfig`).
- `config.py`: File loaders and scaffolding logic.
- `progress.py`: Safe JSONL-based progress log management.
- `api.py`: Routing LLM calls to `UnifiedAiClient`.
- `mcp_tools.py`: MCP tool registration.
- `gui/`: Streamlit web interface.

---

## Dependencies

- `unified-ai-client` (editable install from `D:\Progetti\IA\UnifiedAiClient`)
- `mcp` — MCP server support
- `streamlit` — web GUI
