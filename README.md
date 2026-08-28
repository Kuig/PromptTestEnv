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

Configure your API keys in `secrets.json`, in the directory you run `prompttestenv` from (copy `secrets.json.example` to get started). Fill in only the providers you actually use, and leave the rest empty:

```json
{
    "google_api_key": "YOUR_KEY_HERE",
    "anthropic_api_key": "",
    "openai_api_key": ""
}
```

Credentials are read by `unified_ai_client`, not by this project, so the same file also accepts `mistral_api_key`, `cohere_api_key`, `meta_api_key`, `groq_api_key` and `xai_api_key`. Each one can be supplied as an environment variable instead (`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, and so on), which takes priority over the file. Local providers such as Ollama need no key at all.

---

## Testing

Run the unit test suite (no API key or network access required, nothing under `unified_ai_client` is called for real):

```powershell
python -m unittest discover -s tests -v
```

`tests/fixtures/smoke_project` is not something you run directly: it's a minimal, git-tracked fixture that `tests/testutils.py::make_temp_project()` copies into a fresh temp directory for `test_runner.py` and `test_verdict.py`, which need a real project directory on disk to exercise `progress.jsonl`/`Report/` writing while the LLM calls themselves stay mocked. Never write into `tests/fixtures/smoke_project/` directly; treat it as read-only.

### Manual smoke-test fixtures

`tests/fixtures/` also holds four small benchmark projects meant to be *run for real*, each exercising a different corner of the pipeline:

| Project | Role |
|---|---|
| `RemoteLLMTest` | Remote-only candidates (Google), costly in API calls. Also the only fixture with `pass_media_to_judge: true` and with `reasoning_analysis: "all"` + `dimension_mode: "joint"`, so it's the most expensive one to run in full. |
| `LocalLLMTest` | Local-only candidates (Ollama), slow (local inference) but free of candidate API cost. The only fixture exercising `thinking: true` on a candidate. |
| `FeaturesTest` | Kitchen sink: mixes local and remote candidates, all three `judge_type`s, two groups with `group_verdicts: true`, and `reasoning_analysis: "best"`. Covers the most features in one run. |
| `QuickTest` | Minimal feature set and the shortest timeout: fast, but not free (it still calls remote candidates and judges). Good for a quick sanity check after a change. |

Because they live under `Projects/` in spirit but not in location, they're safe to run repeatedly without any risk to your own benchmarks in the gitignored `Projects/` folder, and unlike a personal `Projects/_SmokeTest`, they're committed, so every contributor validates against the same fixtures:

```powershell
# Cheap: generation + evaluation only, skips verdict/report generation
prompttestenv run tests/fixtures/QuickTest --output-mode winner_only

# Full run: also produces the judge-written verdict/report
prompttestenv run tests/fixtures/QuickTest --output-mode html
```

Either form leaves the fixture's own JSON config untouched: a real run only adds `progress.jsonl`, `Report/`, and one or more `verdict_prompt_debug*.txt` files (one per group plus one for the global verdict, when `group_verdicts` is enabled) inside the project directory, and those are gitignored, so running one for a local check never shows up in your diff.

---

## Configuration

| File | Purpose |
|---|---|
| `secrets.json` | Provider API keys, read by `unified_ai_client` from the working directory. Copy `secrets.json.example` to get started. |
| `config.json` | Cross-project settings: the reasoning-analysis taxonomy and its judge prompts, the sentence-splitting parameters, the list of locally served providers, and the metadata block appended to the verdict judge's system prompt. |

Everything that describes a single benchmark lives in `Projects/<name>/` and nothing else needs to. `config.json` holds the opposite kind of setting: the definition of the *measurement instrument*. The reasoning dimensions, their definitions and the prompts that apply them have to be identical everywhere, otherwise two reports are not comparable, so they are deliberately not per-project and not editable from `judge_config.json`.

`verdict_metadata` is there for the same reason. It is the header the verdict payload opens with, and every section of it describes what the *code* emits: that the aggregate tables carry no standard deviation, that token counts are output only, that an `assert` judge may use the full 1 to 10 range or only a pass/fail pair. Sections are emitted only when the payload contains what they describe, so a project with no reasoning analysis is never told how to read a reasoning profile it will not receive. What a benchmark *means* by its scores stays the author's business, in `verdict_template`.

> **Note:** unlike the sibling projects, this `config.json` is **versioned**, not git-ignored, and ships no `config.json.example`. It contains no credentials, only the measurement instrument, which must be the same for anyone who clones the repo. It is resolved from the current working directory first, then the repo root, then a read-only copy shipped inside the package (`prompttestenv/templates/default_config.json`), so an install from `requirements_prod.txt` still finds it. Every field has a default, so a missing or unreadable file degrades instead of failing.

`candidates.json`, `judge_config.json`, and `test_cases.json` are mandatory: a `run`/`render` fails with a clear error if any is missing. `global_criteria.json` is the one exception: if it's missing (or unreadable), PromptTestEnv logs a warning and silently falls back to `mode: "none"` (global criteria scoring disabled) instead of failing, since `"none"` is itself a valid, explicit way to opt out of global scoring; a missing file is treated the same as that explicit choice.

### Resume Policy

Each run writes an append-only `progress.jsonl` inside the project directory. On the next `run`, PromptTestEnv computes an MD5 hash of `candidates.json`, `judge_config.json`, `test_cases.json`, and `global_criteria.json`, and compares it against the hash stored in the first line of `progress.jsonl`:

- **Hash matches**: already-completed generation/evaluation steps (keyed by `candidate × test × repetition`) are skipped and restored from the log.
- **Hash mismatch**: the existing `progress.jsonl` is renamed to `progress.jsonl.bak` and the run refuses to resume silently: pass `--force-restart` to discard it and start clean, or restore the original config files to resume as before.

Two deliberate exclusions:

- `judge_config.json` is hashed **without** its `reasoning_analysis` scope and `reasoning_judge` block, and the rest of the file is canonicalised before hashing (so reformatting it alone is not a change). Reasoning analysis is a post-hoc pass over traces already stored in the log, so retuning it must not force you to re-buy every candidate response. Run `prompttestenv analyze` to redo just that pass.
- `config.json` is not hashed at all: it is global, so including it would invalidate every project's progress on any edit to the reasoning schema. Instead each reasoning record stores a short stamp of the schema that produced it, which is what lets a report notice it is mixing analyses from different schema versions.

`analyze` and `render` open the log **read-only**: they consume stored results and do not depend on the config still matching, so they never rename or create `progress.jsonl`.

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

# Analyze the stored reasoning traces (no generation, no judging calls)
prompttestenv analyze Projects/MyBenchmark
prompttestenv analyze Projects/MyBenchmark --force-reanalyze

# Regenerate report from existing progress without re-running
prompttestenv render Projects/MyBenchmark

# Start the MCP server (stdio)
prompttestenv mcp

# Launch the Streamlit GUI (runs benchmarks)
prompttestenv gui

# Launch the Streamlit project editor (creates and modifies projects)
prompttestenv editor
```

### MCP (Claude Desktop / AI agents)

Tools exposed:

| Tool | Description |
|---|---|
| `prompttest_init_project` | Initialize a new benchmark project directory |
| `prompttest_run_project` | Run a benchmark project |
| `prompttest_analyze_reasoning` | Analyze the reasoning traces stored in a project's progress log |
| `prompttest_get_results` | Regenerate report from existing progress |

### As a Library

```python
from prompttestenv import init_project, run_project, Candidate

init_project("Projects/MyBenchmark")
candidates = Candidate.load_all("Projects/MyBenchmark")
result = run_project("Projects/MyBenchmark", output_mode="html")
print(result)
```

#### API Reference

Public surface exposed by `from prompttestenv import ...` (see `prompttestenv/__init__.py`):

| Name | Kind | Signature | Description |
|---|---|---|---|
| `init_project` | function | `init_project(project_dir: str) -> None` | Scaffold a new benchmark project directory with default config files. |
| `run_project` | function | `run_project(project_dir: str, output_mode: str = "html", force_restart: bool = False) -> str` | Run the full benchmark and return the report path (or an error string, never raises). |
| `analyze_project` | function | `analyze_project(project_dir: str, force_reanalyze: bool = False) -> str` | Run only the reasoning-analysis phase over the traces already in `progress.jsonl`. No generation and no judging calls. |
| `render_from_progress` | function | `render_from_progress(project_dir: str) -> str` | Regenerate the report from an existing `progress.jsonl`, with no LLM calls. |
| `Candidate` | dataclass | `Candidate.load_all(project_dir) -> list[Candidate]` | Loads and resolves `candidates.json` (including `system_prompt_file` content). Raises `FileNotFoundError` if the file is missing. |
| `TestCase` | dataclass | `TestCase.load_all(project_dir) -> list[TestCase]` | Loads `test_cases.json`. Raises `FileNotFoundError` if the file is missing. |
| `JudgeConfig` | dataclass | `JudgeConfig.load(project_dir) -> JudgeConfig` | Loads `judge_config.json`. Raises `FileNotFoundError` if the file is missing. |
| `GlobalCriteria` | dataclass | `GlobalCriteria.load(project_dir) -> GlobalCriteria` | Loads `global_criteria.json`. Falls back to `mode="none"` if the file is missing or unreadable, does not raise. |

Each of the four also has a writer, used by the project editor:
`Candidate.save_all(project_dir, data)`, `TestCase.save_all(project_dir, data)`,
`JudgeConfig.save(project_dir, data)`, `GlobalCriteria.save(project_dir, data)`.
They take **raw dicts, not instances** — the loaders drop unknown keys and fill
in every omitted default, so round-tripping through the dataclasses would
rewrite far more of a project's file than its author changed, and these files
are hashed byte-for-byte. Writes are atomic, and `trailing_newline=` /
`newline=` let a caller reproduce the file's existing formatting exactly.

All names are re-exported lazily (only imported on first access), so `import prompttestenv` alone stays lightweight even when the underlying modules pull in `unified_ai_client` or `jinja2`.

### Streamlit GUI

```powershell
prompttestenv gui
```

Sidebar for the project path and run options, four actions in the main area
(Initialize / Run / Analyze Reasoning / Render), and the module log streamed
inline. It runs benchmarks; it does not edit them.

### Streamlit project editor

```powershell
prompttestenv editor
```

A second Streamlit app for **creating and modifying** projects: typed forms over
`candidates.json`, `test_cases.json`, `judge_config.json` and
`global_criteria.json`, plus editing the `system_prompts/*.txt` files and
uploading attachments into `test_files/`. **New** scaffolds a project from the
packaged templates. It never runs a benchmark — that is what `gui` is for.

Two behaviours worth knowing:

- **Saving is byte-faithful.** Three of the four config files feed the resume
  hash as raw bytes, so opening a project and saving it unchanged writes
  nothing at all: key order, numeric form (an on-disk `2` stays `2`, not `2.0`),
  trailing newline and line endings are all preserved, along with any keys the
  editor does not know about. If a save *would* invalidate an existing
  `progress.jsonl`, the editor says so and asks for confirmation first, and
  never deletes the log itself. Editing only `reasoning_analysis` or the
  `reasoning_judge` block never triggers that, because those are excluded from
  the hash (see [Resume Policy](#resume-policy)). The one deliberate exception
  to byte fidelity is an attachment path: its separators are normalised to `/`
  on save, so a project authored on Windows also runs on Linux. On a project
  holding backslashes that *is* a change, and the editor flags it as one.
- **`system_prompts/` and `test_files/` are *not* hashed.** Changing a system
  prompt or replacing an attachment does not invalidate a run, so a resumed run
  would mix responses produced under the old and the new version into one
  report. The editor warns about this where it can happen.

> **Note:** CONVENTIONS.md §2 mandates exactly four interfaces (CLI, MCP,
> library, GUI) and §2.4 names a single `gui/app.py`. A fifth `editor`
> subcommand with a second app deviates from both, deliberately: running a
> benchmark and authoring one are disjoint jobs, and folding the editor's six
> tabs of forms into the run GUI would make the common case — press Run, watch
> the log — markedly heavier for no benefit. The two apps share their helpers
> through `prompttestenv/gui/common.py`.

---

## Project Structure

```
Projects/<benchmark>/
├── candidates.json         ← LLM candidates to compare
├── judge_config.json       ← Judge LLM configuration + templates
├── test_cases.json         ← Prompts, evaluation criteria and evaluation modes
├── global_criteria.json    ← Structured global criteria JSON (with mode selector)
├── system_prompts/         ← Optional system prompt files
├── test_files/              ← Attachments (images, text files)
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
        "file": ["test_files/report.txt", "test_files/q3_chart.png"],
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

## Architecture

PromptTestEnv runs each benchmark through a two-phase pipeline, orchestrated by `runner.py`:

1. **Generation** (`generation.py`): for every candidate × test case × repetition, calls the candidate LLM and records the response, token counts, and timing to `progress.jsonl`.
2. **Evaluation** (`evaluator.py`): for every generated response, calls the judge (per the test case's `judge_type`, plus an independent global-criteria score). Results are appended to `progress.jsonl`.
3. **Reasoning analysis** (`analysis.py`, only when `reasoning_analysis` is `"best"` or `"all"`): splits each stored thinking trace into sentence-sized units and has a judge score every unit on each reasoning dimension. It reads only what generation already wrote, so it makes no generation and no judging calls and can be re-run on its own with `prompttestenv analyze`. `"best"` measures only the highest-scoring repetition of each test case, which is the one the report draws, so it costs `repetitions` times less than `"all"`; the report and the verdict payload both record which scope produced the figures, since the two are not comparable.
4. **Verdict** (`verdict.py`): a verdict LLM writes the final comparative report body (per-group verdicts plus a global verdict if `group_verdicts` is enabled, otherwise a single verdict). Its payload opens with a pooled per-candidate table, plus a second table of the reasoning profile when the analysis ran, so the judge compares candidates on figures it does not have to compute itself.
5. **Report** (`reporting.py`): renders the verdict and aggregated statistics into an HTML or Markdown report under `Report/`.

Every phase resumes safely: `progress.jsonl` is an append-only JSONL log, and each run starts by comparing an MD5 hash of the project's config files against the hash stored on the first line (see [Resume Policy](#resume-policy)).

> **A caveat about thinking traces:** some providers do not expose a model's raw chain of thought. Google returns a *summary* the model writes about its own reasoning, typically about half the length the billed thinking tokens imply, while Ollama, Anthropic and the OpenAI-compatible providers return the raw transcript. `UnifiedAiClient` reports which one you got through `AiResponse.reasoning_is_summary`, PromptTestEnv stores it with every response, and the report flags it. Trace length, composition and self-correction counts partly reflect the summariser, so never rank a summarised trace against a raw one on those figures.

### Codebase Layout

The Python package `prompttestenv` is organized as follows:

- `runner.py`: high-level orchestration for executing benchmarks (`run_project`), analysing stored traces (`analyze_project`) and rendering reports (`render_from_progress`).
- `generation.py`: Phase 1, candidate response generation.
- `evaluator.py`: Phase 2, judge evaluation orchestration and concurrency.
- `analysis.py`: Phase 3, the reasoning-analysis pass over stored traces.
- `test_judge.py`: judge dispatch logic for a single test case (`llm-judge`, `similarity`, `assert`).
- `verdict.py`: aggregate group verdicts and the global comparative conclusion.
- `reporting.py`: HTML and Markdown report compilation.
- `models.py`: domain dataclasses (`Candidate`, `TestCase`, `JudgeConfig`, `GlobalCriteria`, `ProgressState`, ...), each owning its own `.load()`/`.load_all()`.
- `config.py`: global `AppConfig` (root `config.json`), project scaffolding (`init_project`) and secrets loading.
- `progress.py`: low-level `progress.jsonl` primitives (config hashing, event appending).
- `api.py`: routing LLM calls to `UnifiedAiClient`.
- `reasoning.py`: trace segmentation, the per-dimension judge calls, and the reasoning metrics.
- `mcp_tools.py`: MCP tool registration.
- `gui/`: the two Streamlit apps — `app.py` (runs benchmarks) and `editor.py`
  (creates and modifies projects) — over `common.py` (shared helpers) and
  `projectio.py` (byte-faithful load/serialise of the config files).

---

## Dependencies

- `unified-ai-client` (editable install from `D:\Progetti\IA\UnifiedAiClient`)
- `mcp`: MCP server support
- `streamlit`: web GUI
- `jinja2`: HTML report templating
