# PromptTestEnv

LLM prompt benchmarking tool.

## Overview

Given a **project directory** containing candidate configurations and test cases,
PromptTestEnv runs each prompt candidate against one or more LLM models (with
optional multimodal attachments), evaluates responses with a judge LLM, and
generates a comparative report: a full HTML page, a short Markdown summary, a
structured JSON export, or a single winner line (see
[Output modes](#output-modes)). Resume is supported via `progress.jsonl`.

## Documentation

| Document | What it covers |
|---|---|
| [HOWTO.md](HOWTO.md) | Authoring a benchmark: every option of the four project config files, the reasoning analysis, and how to repurpose the framework for bulk document processing. |
| [docs/configuration.md](docs/configuration.md) | Key-by-key reference for the root `config.json` and `secrets.json`. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Internal design: the pipeline, the module map, and the invariants. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development install, the test suite, and the fixtures. |
| [report.schema.json](report.schema.json) | JSON Schema of the `--output-mode json` export. |

---

## Installation

From a clone of this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_prod.txt
```

That installs the `prompttestenv` console script plus its four dependencies:
`unified-ai-client` (every AI call), `mcp`, `streamlit` (the two web UIs) and
`jinja2` (the report). To work on the code, see [CONTRIBUTING.md](CONTRIBUTING.md).

Configure your API keys in `secrets.json`, in the directory you run
`prompttestenv` from (copy `secrets.json.example` to get started). Fill in only
the providers you actually use, and leave the rest empty:

```json
{
    "google_api_key": "YOUR_KEY_HERE",
    "anthropic_api_key": "",
    "openai_api_key": ""
}
```

Credentials are read by `unified_ai_client`, not by this project, so the file
also accepts `mistral_api_key`, `cohere_api_key`, `meta_api_key`, `groq_api_key`
and `xai_api_key`. Each can be given as an environment variable instead
(`GOOGLE_API_KEY` and so on), which takes priority; local providers such as
Ollama need no key. Full table in [docs/configuration.md](docs/configuration.md).

---

## Configuration

| File | Purpose |
|---|---|
| `secrets.json` | Provider API keys, read by `unified_ai_client` from the working directory. Copy `secrets.json.example` to get started. |
| `config.json` | Cross-project settings: the reasoning-analysis taxonomy and its judge prompts, the sentence-splitting parameters, the defaults for the reasoning judge, the list of locally served providers, whether candidates are warmed up before being timed, and the metadata block appended to the verdict judge's system prompt. |
| `report.schema.json` | JSON Schema (draft 2020-12) of the `--output-mode json` export. Reference documentation, not something the tool reads: nothing loads it at runtime. A copy ships inside the package as `prompttestenv/templates/report.schema.json` so a wheel install has it too, and a unit test keeps the two identical. |

Everything that describes a single benchmark lives in `Projects/<name>/` and
nothing else needs to. `config.json` holds the opposite kind of setting, the
definition of the *measurement instrument*: the reasoning dimensions and the
prompts that apply them, the warm-up switch that makes reported times comparable,
and the self-describing header of the verdict payload. All of it has to be
identical everywhere, otherwise two reports are not comparable, so it is
deliberately not per-project and not editable from `judge_config.json`.

> **Note:** this `config.json` is **versioned**, not git-ignored, and ships no
> `config.json.example`, which is the opposite of what a file of that name
> usually does. It is deliberate: the file contains no credentials, only the
> measurement instrument, which must be the same for anyone who clones the repo.
> Every key, its default and the three-step lookup that resolves the file are in
> [docs/configuration.md](docs/configuration.md).

`candidates.json`, `judge_config.json` and `test_cases.json` are mandatory: a
`run` or `render` fails with a clear error if any is missing.
`global_criteria.json` is the one exception. If it is missing or unreadable,
PromptTestEnv logs a warning and falls back to `mode: "none"` (global scoring
disabled), since `"none"` is itself a valid, explicit way to opt out and a
missing file is treated the same as that explicit choice.

### Resume Policy

Each run writes an append-only `progress.jsonl` inside the project directory. On
the next `run`, PromptTestEnv computes an MD5 hash of `candidates.json`,
`judge_config.json`, `test_cases.json` and `global_criteria.json`, and compares
it against the hash stored in the first line of `progress.jsonl`:

- **Hash matches**: already-completed generation/evaluation steps (keyed by
  `candidate x test x repetition`) are skipped and restored from the log.
- **Hash mismatch**: the existing `progress.jsonl` is renamed to
  `progress.jsonl.bak` and the run refuses to resume silently. Pass
  `--force-restart` to discard it and start clean, or restore the original
  config files to resume as before.

Two deliberate exclusions:

- `judge_config.json` is hashed **without** its `reasoning_analysis` scope and
  `reasoning_judge` block, and the rest of the file is canonicalised before
  hashing (so reformatting it alone is not a change). Reasoning analysis is a
  post-hoc pass over traces already stored in the log, so retuning it must not
  force you to re-buy every candidate response. Run `prompttestenv analyze` to
  redo just that pass.
- `config.json` is not hashed at all: it is global, so including it would
  invalidate every project's progress on any edit to the reasoning schema.
  Instead each reasoning record stores a short stamp of the schema that produced
  it, which is what lets a report notice it is mixing analyses from different
  schema versions.

`analyze` and `render` open the log **read-only**: they consume stored results
and do not depend on the config still matching, so they never rename or create
`progress.jsonl`.

---

## Usage

### CLI

```powershell
prompttestenv init Projects/MyBenchmark                        # scaffold a new project
prompttestenv run Projects/MyBenchmark --output-mode html      # run the benchmark
prompttestenv run Projects/MyBenchmark --force-restart         # re-run, ignoring progress
prompttestenv analyze Projects/MyBenchmark                     # reasoning traces only
prompttestenv analyze Projects/MyBenchmark --force-reanalyze   # redo existing analyses
prompttestenv render Projects/MyBenchmark --output-mode json   # re-render, no LLM call
prompttestenv mcp                                              # MCP server on stdio
prompttestenv gui                                              # Streamlit GUI (runs benchmarks)
prompttestenv editor                                           # Streamlit project editor
```

### Output modes

`run` and `render` take the same `--output-mode`. Every response, score, trace
and analysis is written to `progress.jsonl` as it is produced, whichever mode you
pick, so a later `render` rebuilds the other formats from the same run **without
a single LLM call**.

| Mode | Writes | What it is |
|---|---|---|
| `html` (default) | `Report/<timestamp>_<N>C_<M>T.html` | The full report, for a human. Verdict, per-test-case scores with mean and sd, best response per candidate, and the thinking trace colour-coded in place by the reasoning analysis. |
| `md` | `Report/<timestamp>_<N>C_<M>T.md` | The verdict text alone, as a readable summary. Deliberately just that: it is meant for someone who wants the conclusion without the apparatus, or for an LLM driving PromptTestEnv as a tool, which needs the outcome without spending its context on four hundred rows of statistics. The detail is not gone, it is in `progress.jsonl`. |
| `json` | `Report/<timestamp>_<N>C_<M>T.json` | The same content the HTML carries, structured for a program instead of a page, **plus** the per-repetition raw values the page reduces to mean and sd. Described field by field in [`report.schema.json`](report.schema.json). |
| `winner_only` | nothing | One line naming the highest average task score. The cheapest mode: it skips the verdict judge entirely, so it is the one to use for a smoke run. |

One exception to "render rebuilds anything": `winner_only` never calls the
verdict judge, so it writes no verdict to the log and `render` has nothing to
render, reporting the progress it found instead. To get a report out of such a
run, call `run` again with another mode: generation and evaluation are restored
from the log, so you pay only the verdict call.

Before reading a JSON export, know that **`-1` means "not measured"** and never a
low score, while `0.0` in a coverage or a rate is a real measurement. See
[HOWTO.md](HOWTO.md#reading-a-json-export) and, per field,
[`report.schema.json`](report.schema.json).

### MCP (Claude Desktop / AI agents)

Tools exposed:

| Tool | Description |
|---|---|
| `prompttest_init_project` | Initialize a new benchmark project directory |
| `prompttest_run_project` | Run a benchmark project |
| `prompttest_analyze_reasoning` | Analyze the reasoning traces stored in a project's progress log |
| `prompttest_get_results` | Regenerate report from existing progress, in any output mode, with no LLM call |

### As a Library

```python
from prompttestenv import init_project, run_project, Candidate

init_project("Projects/MyBenchmark")
candidates = Candidate.load_all("Projects/MyBenchmark")
result = run_project("Projects/MyBenchmark", output_mode="html")
print(result)
```

#### API Reference

Public surface exposed by `from prompttestenv import ...`:

| Name | Kind | Signature | Description |
|---|---|---|---|
| `init_project` | function | `init_project(project_dir: str) -> None` | Scaffold a new benchmark project directory with default config files. |
| `run_project` | function | `run_project(project_dir: str, output_mode: str = "html", force_restart: bool = False) -> str` | Run the full benchmark and return the report path (or an error string, never raises). `output_mode` is one of `html`, `md`, `json`, `winner_only`, see [Output modes](#output-modes). |
| `analyze_project` | function | `analyze_project(project_dir: str, force_reanalyze: bool = False) -> str` | Run only the reasoning-analysis phase over the traces already in `progress.jsonl`. No generation and no judging calls. |
| `render_from_progress` | function | `render_from_progress(project_dir: str, output_mode: str = "html") -> str` | Regenerate the report from an existing `progress.jsonl`, in any output mode, with no LLM calls. Returns a progress summary instead when the log holds no verdict. |
| `Candidate` | dataclass | `Candidate.load_all(project_dir) -> list[Candidate]` | Loads and resolves `candidates.json` (including `system_prompt_file` content). Raises `FileNotFoundError` if the file is missing. |
| `TestCase` | dataclass | `TestCase.load_all(project_dir) -> list[TestCase]` | Loads `test_cases.json`. Raises `FileNotFoundError` if the file is missing. |
| `JudgeConfig` | dataclass | `JudgeConfig.load(project_dir) -> JudgeConfig` | Loads `judge_config.json`. Raises `FileNotFoundError` if the file is missing. |
| `GlobalCriteria` | dataclass | `GlobalCriteria.load(project_dir) -> GlobalCriteria` | Loads `global_criteria.json`. Falls back to `mode="none"` if the file is missing or unreadable, does not raise. |

Each of the four dataclasses also has a writer, used by the project editor
(`Candidate.save_all`, `TestCase.save_all`, `JudgeConfig.save`,
`GlobalCriteria.save`). They take **raw dicts, not instances**, write atomically,
and let a caller reproduce the file's existing formatting exactly through
`trailing_newline=` and `newline=`. [ARCHITECTURE.md](ARCHITECTURE.md) explains
why byte fidelity is not optional here.

All names are re-exported lazily (imported on first access), so
`import prompttestenv` alone stays lightweight.

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
packaged templates. It never runs a benchmark, which is what `gui` is for.

Running a benchmark and authoring one are disjoint jobs, which is why they are
two apps: folding the editor's six tabs of forms into the run GUI would make the
common case, press Run and watch the log, markedly heavier for no benefit. Two
behaviours are worth knowing:

- **Saving is byte-faithful.** Opening a project and saving it unchanged writes
  nothing at all, because three of the four config files feed the resume hash as
  raw bytes. When a save *would* invalidate an existing `progress.jsonl` the
  editor says so and asks first, and it never deletes the log itself. Editing
  only `reasoning_analysis` or the `reasoning_judge` block never triggers that,
  since those are excluded from the hash (see [Resume Policy](#resume-policy)).
  The single exception is an attachment path, whose separators are normalised to
  `/` so a project authored on Windows also runs on Linux.
- **`system_prompts/` and `test_files/` are *not* hashed.** Changing a system
  prompt or replacing an attachment does not invalidate a run, so a resumed run
  would mix responses produced under the old and the new version into one report.
  The editor warns about this where it can happen.
