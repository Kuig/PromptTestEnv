# Contributing

Everything a contributor needs and a user does not: the development install, the
test suite, and the fixtures.

## Development install

From a clone of this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_dev.txt
```

`requirements_dev.txt` is two editable installs: this package, and
`UnifiedAiClient`. It expects the latter to be checked out **beside** this
repository, as `../UnifiedAiClient`. If yours lives somewhere else, install it
by its own path first and then run `pip install -e .` here:

```powershell
pip install -e <path-to-UnifiedAiClient>
pip install -e .
```

All AI calls go through that external `unified_ai_client` package. This project
never talks to a provider API directly, so a change to provider behaviour
belongs there, not here.

## Running the test suite

No API key and no network access are required: nothing under
`unified_ai_client` is called for real.

```powershell
python -m unittest discover -s tests -v
```

525 tests, on stdlib `unittest`. There is no `pytest`, `ruff` or `mypy`
configuration.

### Running a single test

`tests/` is deliberately **not** a package: there is no `tests/__init__.py`, so
the default `discover -s tests` invocation makes `tests/` itself the top-level
directory, puts it on `sys.path`, and lets every file import its siblings as
bare top-level modules (`from testutils import ...`, never
`from tests.testutils import ...`).

The consequence is that targeting one test needs `tests/` on `PYTHONPATH`
explicitly, because `python -m unittest <dotted.path>` alone does not add it:

```powershell
$env:PYTHONPATH = "tests"; python -m unittest test_models.TestJudgeConfigFromDict.test_global_criteria_as_dict -v
```

### Where to mock

Every `unified_ai_client` call is mocked **at the correct boundary**, and
`tests/testutils.py` documents the rule this codebase requires and why. The one
that bites hardest: mocks of `get_llm_response` must return an `LlmResult`, not
a tuple. A tuple-returning mock kept a real unpacking bug in `verdict.py`
invisible for a full development cycle, because the production code caught the
resulting `TypeError` while the suite stayed green.

`LoggerResetTestCase` in `testutils.py` is the base class to use for anything
that touches the logger, directly or by importing a GUI module. Importing
`prompttestenv.gui.app` rebinds the logging backend as an import-time side
effect, and without the reset that state leaks into every test that runs after.

## Fixtures

### `tests/fixtures/smoke_project`

Not something you run. It is a minimal, git-tracked project directory that
`testutils.make_temp_project()` copies into a fresh temp directory for
`test_runner.py`, `test_verdict.py`, `test_gui.py`, `test_gui_editor.py` and
`test_gui_projectio.py`, which all need a real project on disk to exercise
`progress.jsonl` and `Report/` writing while the LLM calls themselves stay
mocked.

**Never write into `tests/fixtures/smoke_project/` directly.** Treat it as
read-only: generation, evaluation and progress tracking all write into whatever
project directory they are handed.

### Manual smoke-test fixtures

`tests/fixtures/` also holds four small benchmark projects meant to be *run for
real*, each exercising a different corner of the pipeline:

| Project | Role |
|---|---|
| `RemoteLLMTest` | Remote-only candidates (Google), costly in API calls. Also the only fixture with `pass_media_to_judge: true` and with `reasoning_analysis: "all"` plus `dimension_mode: "joint"`, so it is the most expensive one to run in full. |
| `LocalLLMTest` | Local-only candidates (Ollama), slow (local inference) but free of candidate API cost. The only fixture exercising `thinking: true` on a candidate. |
| `FeaturesTest` | Kitchen sink: mixes local and remote candidates, all three `judge_type`s, two groups with `group_verdicts: true`, and `reasoning_analysis: "best"`. Covers the most features in one run. |
| `QuickTest` | Minimal feature set and the shortest timeout: fast, but not free (it still calls remote candidates and judges). Good for a quick sanity check after a change. |

They live under `Projects/` in spirit but not in location, so they are safe to
run repeatedly without any risk to your own benchmarks in the git-ignored
`Projects/` folder, and unlike a personal `Projects/_SmokeTest` they are
committed, so every contributor validates against the same fixtures:

```powershell
# Cheap: generation + evaluation only, skips verdict/report generation
prompttestenv run tests/fixtures/QuickTest --output-mode winner_only

# Full run: also produces the judge-written verdict/report
prompttestenv run tests/fixtures/QuickTest --output-mode html
```

Either form leaves the fixture's own JSON config untouched. A real run only adds
`progress.jsonl`, `Report/` and one or more `verdict_prompt_debug*.txt` files
(one per group plus one for the global verdict, when `group_verdicts` is
enabled) inside the project directory, and those are git-ignored, so running one
for a local check never shows up in your diff.

## Validating an end-to-end change

The unit suite never touches the real `unified_ai_client` path, so a change to
the pipeline still wants one real run behind it. In increasing order of cost:

- `prompttestenv render <project>` when you only touched reporting or
  aggregation code and an existing `progress.jsonl` is available. Zero API
  calls.
- `prompttestenv analyze <project>` when you touched the reasoning analysis.
  Judge calls only, no generation.
- `prompttestenv run tests/fixtures/QuickTest --output-mode winner_only` for the
  cheapest full pipeline, which skips the verdict judge.
- `prompttestenv run tests/fixtures/FeaturesTest --output-mode html` when the
  change could affect grouping, judge dispatch or the report itself.
