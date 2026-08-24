# PromptTestEnv

LLM prompt benchmarking tool.

## Overview

Given a **project directory** containing candidate configurations and test cases, PromptTestEnv runs each prompt candidate against one or more LLM models (with optional multimodal attachments), evaluates responses with a judge LLM, and generates comparative HTML or Markdown reports. Resume is supported via `progress.jsonl`.

> 💡 **Tip:** For a comprehensive guide on creating tests, configuring options, and using the framework for alternative use cases (like bulk document processing), see the [HOWTO Guide](HOWTO.md).

---

## Installation

```powershell
cd D:\Progetti\IA\PromptTestEnv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e D:\Progetti\IA\UnifiedAiClient
pip install -e .
```

`pip install -e .` reads `pyproject.toml` and installs `mcp`, `streamlit`, `jinja2`, and `unified-ai-client`, plus the `prompttestenv` console script, in one step.

Configure your API key in `secrets.json` at the project root (copy `secrets.json.example` to get started):

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

### MCP (Claude Desktop / AI agents)

Tools exposed:

| Tool | Description |
|---|---|
| `prompttest_init_project` | Initialize a new benchmark project directory |
| `prompttest_run_project` | Run a benchmark project |
| `prompttest_get_results` | Regenerate report from existing progress |

### As a Library

```python
from prompttestenv import init_project, run_project, Candidate

init_project("Projects/MyBenchmark")
candidates = Candidate.load_all("Projects/MyBenchmark")
result = run_project("Projects/MyBenchmark", output_mode="html")
print(result)
```

See [API Reference](#api-reference) below for the full public surface.

### Streamlit GUI

```powershell
prompttestenv gui
```

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

> **Note:** PromptTestEnv intentionally has no root-level `config.json`. Unlike other sibling projects, every benchmark under `Projects/<name>/` is fully self-contained (`candidates.json`, `judge_config.json`, `test_cases.json`, `global_criteria.json`): there is no cross-project provider/connection setting that would belong in a shared root config.

`candidates.json`, `judge_config.json`, and `test_cases.json` are mandatory: a `run`/`render` fails with a clear error if any is missing. `global_criteria.json` is the one exception: if it's missing (or unreadable), PromptTestEnv logs a warning and silently falls back to `mode: "none"` (global criteria scoring disabled) instead of failing, since `"none"` is itself a valid, explicit way to opt out of global scoring; a missing file is treated the same as that explicit choice.

### Resume Policy

Each run writes an append-only `progress.jsonl` inside the project directory. On the next `run`, PromptTestEnv computes an MD5 hash of `candidates.json`, `judge_config.json`, `test_cases.json`, and `global_criteria.json`, and compares it against the hash stored in the first line of `progress.jsonl`:

- **Hash matches**: already-completed generation/evaluation steps (keyed by `candidate × test × repetition`) are skipped and restored from the log.
- **Hash mismatch**: the existing `progress.jsonl` is renamed to `progress.jsonl.bak` and the run refuses to resume silently: pass `--force-restart` to discard it and start clean, or restore the original config files to resume as before.

---

## API Reference

Public surface exposed by `from prompttestenv import ...` (see `prompttestenv/__init__.py`):

| Name | Kind | Signature | Description |
|---|---|---|---|
| `init_project` | function | `init_project(project_dir: str) -> None` | Scaffold a new benchmark project directory with default config files. |
| `run_project` | function | `run_project(project_dir: str, output_mode: str = "html", force_restart: bool = False) -> str` | Run the full benchmark and return the report path (or an error string, never raises). |
| `render_from_progress` | function | `render_from_progress(project_dir: str) -> str` | Regenerate the report from an existing `progress.jsonl`, with no LLM calls. |
| `Candidate` | dataclass | `Candidate.load_all(project_dir) -> list[Candidate]` | Loads and resolves `candidates.json` (including `system_prompt_file` content). Raises `FileNotFoundError` if the file is missing. |
| `TestCase` | dataclass | `TestCase.load_all(project_dir) -> list[TestCase]` | Loads `test_cases.json`. Raises `FileNotFoundError` if the file is missing. |
| `JudgeConfig` | dataclass | `JudgeConfig.load(project_dir) -> JudgeConfig` | Loads `judge_config.json`. Raises `FileNotFoundError` if the file is missing. |
| `GlobalCriteria` | dataclass | `GlobalCriteria.load(project_dir) -> GlobalCriteria` | Loads `global_criteria.json`. Falls back to `mode="none"` if the file is missing or unreadable, does not raise. |

All names are re-exported lazily (only imported on first access), so `import prompttestenv` alone stays lightweight even when the underlying modules pull in `unified_ai_client` or `jinja2`.

---

## Architecture

PromptTestEnv runs each benchmark through a two-phase pipeline, orchestrated by `runner.py`:

1. **Generation** (`generation.py`): for every candidate × test case × repetition, calls the candidate LLM and records the response, token counts, and timing to `progress.jsonl`.
2. **Evaluation** (`evaluator.py`): for every generated response, calls the judge (per the test case's `judge_type`, plus an independent global-criteria score) and, if `reasoning_analysis` is enabled, an additional reasoning-trace analysis. Results are appended to `progress.jsonl`.
3. **Verdict** (`verdict.py`): a verdict LLM writes the final comparative report body (per-group verdicts plus a global verdict if `group_verdicts` is enabled, otherwise a single verdict).
4. **Report** (`reporting.py`): renders the verdict and aggregated statistics into an HTML or Markdown report under `Report/`.

Both phases resume safely: `progress.jsonl` is an append-only JSONL log, and each run starts by comparing an MD5 hash of the project's config files against the hash stored on the first line (see [Resume Policy](#resume-policy)).

### Codebase Layout

The Python package `prompttestenv` is organized as follows:

- `runner.py`: high-level orchestration for executing benchmarks (`run_project`) and rendering reports (`render_from_progress`).
- `generation.py`: Phase 1, candidate response generation.
- `evaluator.py`: Phase 2, judge evaluation orchestration and concurrency.
- `test_judge.py`: judge dispatch logic for a single test case (`llm-judge`, `similarity`, `assert`).
- `verdict.py`: aggregate group verdicts and the global comparative conclusion.
- `reporting.py`: HTML and Markdown report compilation.
- `models.py`: domain dataclasses (`Candidate`, `TestCase`, `JudgeConfig`, `GlobalCriteria`, `ProgressState`, ...), each owning its own `.load()`/`.load_all()`.
- `config.py`: project scaffolding (`init_project`) and secrets loading.
- `progress.py`: low-level `progress.jsonl` primitives (config hashing, event appending).
- `api.py`: routing LLM calls to `UnifiedAiClient`.
- `reasoning.py`: optional reasoning-trace analysis.
- `mcp_tools.py`: MCP tool registration.
- `gui/`: Streamlit web interface.

---

## Dependencies

- `unified-ai-client` (editable install from `D:\Progetti\IA\UnifiedAiClient`)
- `mcp`: MCP server support
- `streamlit`: web GUI
- `jinja2`: HTML report templating
