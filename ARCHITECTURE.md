# Architecture

Internal design of PromptTestEnv, for whoever edits the code. If you only want
to *use* the tool, `README.md` and `HOWTO.md` are the documents you want.

## The problem

Comparing LLM configurations is easy to do badly. A single response proves
nothing, so every configuration has to run every task several times; scoring the
results by hand does not scale, so the scoring itself has to be automated and
consistent; and a run that calls a dozen models a few hundred times is long and
expensive enough that losing it to a crash, a rate limit or a closed laptop is
unacceptable.

PromptTestEnv answers those three pressures with one shape: a **project
directory** describes the whole benchmark, an **append-only log** records every
call the moment it returns, and **every phase is resumable** from that log.
Nothing in the pipeline is allowed to be the only copy of a result.

A second, quieter goal shapes as much of the code: two reports must be
comparable. That is why the reasoning taxonomy and the warm-up switch live in a
root `config.json` that no project can override, why candidates are warmed up
outside the timed block, and why a summarised thinking trace is flagged as such
wherever it is shown.

## The project directory

Everything describing one benchmark lives in one folder, and everything a run
produces lands back in it:

```
Projects/<benchmark>/
├── candidates.json           ← LLM candidates to compare
├── judge_config.json         ← judge LLM configuration and templates
├── test_cases.json           ← prompts, evaluation criteria and evaluation modes
├── global_criteria.json      ← global criteria, with a mode selector
├── system_prompts/           ← optional system prompt files
├── test_files/               ← attachments (images, audio, PDFs, text files)
├── progress.jsonl            ← resume-safe run log: every response, score and analysis
├── progress.jsonl.bak        ← the previous log, kept when the config hash stops matching
├── verdict_prompt_debug*.txt ← the exact payload each verdict call received
└── Report/                   ← generated reports, one file per run (.html / .md / .json)
```

The folder is self-contained on purpose: it can be moved, archived or handed to
someone else, and it will still run and still render.

## Codemap

### The pipeline

`runner.py` is the only orchestrator. `run_project` walks three resumable phases
and then produces output; `analyze_project` runs the third phase alone;
`render_from_progress` runs none of them and only renders what the log holds.

1. **Generation** (`generation.py`). For every candidate x test case x
   repetition, calls the candidate model and records the response, its token
   counts, its thinking trace and the elapsed time as a `gen` event. Runs inside
   a single-worker thread pool purely to enforce `max_response_timeout_seconds`.
2. **Evaluation** (`evaluator.py`). For every generated response, produces two
   independent scores: the task score, dispatched by the test case's
   `judge_type`, and the global score, dispatched by `global_criteria.json`'s
   `mode`. The two are computed in parallel against a remote judge and
   sequentially against a local one, since a local backend serves one model at a
   time. Each result is an `eval` event.
3. **Reasoning analysis** (`analysis.py`, only when `reasoning_analysis` is
   `"best"` or `"all"`). Reads the traces phase 1 already stored, so it makes no
   generation and no judging calls of its own and can be re-run alone. Each
   result is a `reasoning` event. `"best"` analyses only the highest-scoring
   repetition of each candidate x test case, which is the one the report draws
   the trace for, so it costs `repetitions` times less than `"all"`. Both the
   report and the verdict payload record which scope produced the figures, since
   figures from the two scopes are not comparable.

Then `verdict.py` has a judge LLM write the report body, and `reporting.py` or
`json_report.py` renders it.

### Modules

- `runner.py`: `run_project`, `analyze_project`, `render_from_progress`. The
  three public entry points, and the only place phases are sequenced.
- `generation.py`: phase 1, including the per-candidate warm-up.
- `evaluator.py`: phase 2, judge orchestration and its concurrency policy.
- `analysis.py`: phase 3, plus `keys_to_analyze` and `_best_key`, the single
  definition of which repetition counts as "best".
- `test_judge.py`: judge dispatch for one response. Three evaluators, one score.
  The `assert` evaluator's lambda runs against an explicit namespace (its
  argument, the full builtins, `re`, `math`, `json`, `statistics`, `datetime`,
  `string`, and a `similarity(a, b)` embedding helper), not this module's
  globals.
- `reasoning.py`: trace segmentation into units, the per-dimension judge calls,
  and the metrics computed without a judge.
- `verdict.py`: the verdict payload, per-group verdicts and the global verdict.
- `reporting.py`: the Jinja2 HTML report and its presentation helpers (badges,
  trace colouring, a small Markdown to HTML converter).
- `json_report.py`: the JSON export.
- `models.py`: every domain dataclass, each owning its own file I/O, plus
  `pool_by_candidate`, the one per-candidate aggregation.
- `config.py`: `AppConfig` over the root `config.json`, and `init_project`.
- `projectio.py`: byte-faithful load and serialise of the four project config
  files, their validation, and the guards on the two asset directories. Shared
  by every interface that edits a project.
- `projectedit.py`: `read_project` and `edit_project`, the interface-agnostic
  editing API. Applies a partial patch on top of `projectio`.
- `progress.py`: the low-level `progress.jsonl` primitives, config hashing and
  event appending.
- `api.py`: the single door to `UnifiedAiClient`.
- `logger.py`: the four-backend logger (console, MCP, silent, Streamlit).
- `mcp_tools.py`: MCP tool registration.
- `__main__.py`: the CLI, and the only entry point.
- `gui/`: two Streamlit apps, `app.py` (runs benchmarks) and `editor.py`
  (creates and modifies projects), over `common.py` (shared helpers). Nothing in
  the core may import this package, which is why `projectio.py` sits above.

## Invariants

These are the rules that are invisible in the code and are the first thing an
editor breaks.

**No module talks to a provider API.** Every AI call goes through `api.py`, which
calls `UnifiedAiClient`. There is no provider dispatch, no key handling and no
attachment encoding anywhere else in this project: a test case's `file` is
resolved to a path and handed over, and the client alone decides whether it
becomes an upload, a base64 block, or text inlined ahead of the prompt.

**`-1` means "not measured", everywhere.** Every evaluator, every metric and
every aggregate uses the same sentinel for "we could not produce this figure",
and `calculate_stats` filters it out rather than averaging it in. In the
coverages and rates, `0.0` is a real measurement instead, and the two must never
be conflated. The one place the framework does not impose the range is an
`assert` criterion's return value, which is the author's own Python and is left
alone.

**Three of the four project config files are hashed as raw bytes.** That is what
makes resume safe, and it is why `projectio.py` edits raw dicts and never
dataclass instances: the loaders drop unknown keys and materialise every omitted
default, so a round trip through the dataclasses would rewrite far more of a file
than its author changed and would silently invalidate a finished run. Key order,
numeric form, trailing newline and line endings are all preserved, and identical
bytes are not written at all.

**Nothing imports `gui.app` or `gui.editor`.** Importing either executes a
Streamlit script at import time and rebinds the logging backend in the importing
process. `gui/__init__.py` exists only so the package ships in a wheel.

**`reporting.py` and `json_report.py` share no code.** They are two renderers of
one aggregation, `pool_by_candidate` in `models.py`, which is what stops the HTML
and the JSON from reporting a candidate differently. A helper both sides want
belongs in `models.py` or `reasoning.py`, never in a cross-import between them.

**The measurement instrument is not per project.** The reasoning dimensions,
their definitions and prompts, the sentence-splitting parameters, the
provider-locality table and the warm-up switch live in the root `config.json`
and are not reachable from `judge_config.json`. If each benchmark could redefine
them, no two reports would be comparable.

**Reasoning units are offsets, never text.** A trace is split procedurally in
Python and the judge only ever returns numbers against unit ids, so full coverage
and zero overlap are structural properties rather than instructions a judge may
ignore, and the report can colour the real trace in place.

**A summarised trace is never silently treated as a raw one.** Some providers do
not expose a model's raw chain of thought: Google returns a summary the model
writes about its own reasoning, typically about half the length its billed
thinking tokens imply, while Ollama, Anthropic and the OpenAI-compatible
providers return the raw transcript. `AiResponse.reasoning_is_summary` carries
which one arrived, it is stored on every `gen` event, and it gates absolute token
attribution and is flagged in the report. Trace length, composition and
self-correction counts partly reflect the summariser, so those figures must never
be compared across the two kinds. Traces recorded before providers reported the
flag carry `None` and read as `unknown`, never as `raw`.

## Boundaries

**Between the phases**: only `progress.jsonl`. Phase 3 reads what phase 1 wrote
rather than receiving it in memory, which is exactly why `analyze` can run alone
months later. The same boundary is why `render` needs no LLM.

**Between the run and the measurement**: the config hash. It covers the four
project config files and deliberately excludes two things, so that iterating on
the *measurement* never costs a re-run of the *measured*. The
`reasoning_analysis` scope and the `reasoning_judge` block are stripped out of
`judge_config.json` before hashing, and the root `config.json` is not hashed at
all. Each reasoning event instead carries a short stamp of the schema that
produced it, which is what lets a report notice that it mixes schema versions.

**Between reading and writing the log**: `ProgressState.load(..., readonly=True)`.
`analyze` and `render` only consume stored results, so they must not rename or
create the log they were asked to read.

**Between the library and its interfaces**: the interfaces are thin. The CLI, the
MCP server, the library and both GUIs all call the same three functions in
`runner.py`, which never raise and always return a descriptive string. Editing a
project is the same arrangement one layer over: every interface, the Streamlit
editor included, reaches the config files only through `projectio.py`, so no one
of them can hold a rule the others lack. `projectedit.py` adds the patch language
and the two gates a form does not need to express, validation and the refusal to
invalidate a finished run without being told to.

## Cross-cutting concerns

**Resume.** `progress.jsonl` is append-only, one JSON object per line. Line 1 is
a `meta` record holding the config hash; every later line is a `gen`, `eval`,
`reasoning` or `verdict` event. All three phases key their completed work by
`(candidate, test, repetition)` and skip anything already present, restoring its
data from the log instead of re-calling the model. A hash mismatch renames the
file to `progress.jsonl.bak` and refuses to resume silently.

**Warm-up.** Only candidates are timed, so only candidates are warmed. Before
each one's timed block, and outside it, `api.warm_up_for_run` hands
`UnifiedAiClient` that candidate's model plus the attachments of the test cases
it still has pending. Without it, every per-process one-off cost (SDK import,
client construction, DNS and TLS, attachment upload) is charged to whichever
candidate happens to run first. The rule is "warm what is still pending", never
"skip because events exist": a resumed run always starts with a cold upload
cache, but a candidate with nothing left to do is skipped entirely.

**Logging.** One logger with four backends. The business logic calls the same
functions whatever the interface, and only the entry point selects a backend:
console for the CLI, stderr for MCP (stdout carries the JSON-RPC framing, and a
log line there corrupts it), Streamlit widgets for the GUIs.

**Failure.** Core logic fails fast with typed exceptions, `FileNotFoundError` for
the three mandatory config files above all. AI calls degrade instead: a failed
judge call returns the `-1` sentinel rather than raising, so one bad response
never costs the run. The entry points catch everything and return a readable
`"Error: ..."` string.
