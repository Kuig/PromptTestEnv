# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.8] - 2026-09-12

### Changed

- `RunKey` and `ProgressState` moved out of `models.py` into `progress.py`, which
  has zero dependency on the rest of the package; every other module now imports
  both from there instead of from `models.py`.
- `run_project`, `analyze_project`, and `render_from_progress` now share one
  `_load_project_config` helper for the project-directory-exists and
  config-file-loading checks, instead of three independently drifting copies
  (one of which caught a narrower set of exceptions than the others). As a side
  effect, `render_from_progress`'s configuration check now runs before its log
  check, so a project with both a missing `progress.jsonl` and a malformed
  `judge_config.json` reports the more specific error.
- Every "not measured" figure in the report and the verdict payload (cost per
  point, score cells, the OVERALL AGGREGATE table) is now spelled the same way,
  decided in one place (`models.render_measured`) instead of by each formatter
  independently, which previously produced a mix of `"N/A"`, `"n/a"`, and
  `"not measured"` across different parts of the same report.
- The verdict payload's metadata assembly (repetition count, metric semantics,
  the judge-type legend) moved out of `verdict.py` into a new, deliberately
  lightweight `verdict_payload.py` module that imports neither `api` nor
  `reasoning`, leaving `verdict.py` with rendering, the payload/rendering glue,
  and LLM-call orchestration. `parse_grouped_verdict` moved further, into
  `models.py`, since it is pure JSON parsing with no LLM call; `reporting.py`
  and `json_report.py` no longer import anything from `verdict.py` at all.
- `models.GenerationOutcome` replaces an ad hoc dict passed from the generation
  phase to the evaluation phase; `models.apply_gen_event`/`apply_eval_event`
  centralize the per-repetition bookkeeping (tokens, times, scores) that
  `generation.py`, `evaluator.py`, and `render_from_progress` used to each
  duplicate by hand.

### Fixed

- The report could show a reasoning trace from a different, lower-scoring
  repetition than the response displayed beside it, whenever the actual winning
  repetition had no trace of its own. `analysis.attach_best` is now the single
  computation of which repetition wins (previously tracked incrementally and
  independently in `generation.py`, `evaluator.py`, and
  `render_from_progress`, sometimes disagreeing), and `attach_reasoning` now
  checks whether that same winner has a trace instead of picking its own,
  possibly different, winner over a narrower key set.
- A reasoning dimension whose judge call failed across every scoring window of
  a trace no longer leaves that dimension's per-unit intensities at their
  dataclass default of `0.0`, indistinguishable from a real "scored zero":
  every affected unit is now marked with the `-1.0` "not measured" sentinel,
  and the trace view renders it uncoloured instead of tinting it as a genuine
  zero. `density` (the sum of the three reasoning coverages) is now `-1.0`
  whenever any one of the three could not be measured for a repetition,
  instead of silently summing whichever coverages did succeed into a partial,
  non-comparable figure.
- `generate_verdict`'s failure log now includes the full traceback alongside
  the exception message, so a recurrence of the exact bug class that once let
  a stale test mock's `TypeError` go unnoticed for a full development cycle
  (a swallowed `except Exception` around an LLM-call type mismatch) surfaces
  immediately in the logs instead of only through a failing regression test.

### Added

- Windows convenience launchers (`Windows Integration/launch_gui.bat`,
  `launch_editor.bat`) that activate the virtual environment and start the two
  Streamlit GUIs from a repo checkout.

## [0.2.7] - 2026-09-09

### Changed

- Judge JSON-reply parsing (code-fence stripping, single-element list unwrapping) unified
  into one `api.parse_json_object`, used by the task judge, the reasoning judge, and the
  verdict parser instead of three independent implementations.
- The `(candidate, test_id, repetition)` key used throughout resume and progress tracking
  is now a named `RunKey`, and a judge's `(score, reasoning)` result is a named `JudgeScore`,
  in place of bare tuples. No change to `progress.jsonl`'s on-disk format.
- Deduplicated Streamlit GUI code: the folder picker button, the row action bar
  (move/duplicate/delete), and the `thinking` widget are now defined once in
  `gui/common.py` / `gui/editor.py` and shared between the two GUIs instead of being
  copied.
- `GlobalCriteria` is now built from a raw dict in one place instead of two.
- `save_all`/`save` on the four config dataclasses now accept `str | Path`, matching
  their `load*` counterparts.
- Corrected several docstrings and one stale line-number reference to match current
  behavior (a couple of missing type annotations, `generate_html_report`'s signature and
  return type, the MCP tools' description of what `run_project`/`render_from_progress`
  actually return).
- The abandoned-call buffer between PromptTestEnv's own response deadline and the
  provider's is now zero: a provider exception raised after the deadline is caught and
  stored as a placeholder result instead of racing the next call.
- The evaluation judge now retries once instead of the client's default of five, so an
  abandoned judge call does not keep spending retries in the background after the run has
  moved on.
- The local-model preload used by the reasoning judge now runs only when a call will
  actually reach it (not for `similarity`/`assert`-only evaluation), and the
  sequential/parallel gate for task and global evaluation now looks at the provider each
  side actually uses instead of only the task judge's.
- `output_mode` is now validated at every entry point (library, MCP, both GUIs), not just
  the CLI, and `render_from_progress` now runs the same pre-flight checks as `run_project`.
- Removed a `sys.path` insertion that exposed the package's internal module names
  (`config`, `api`, `models`, ...) globally.
- `init_project` now writes every scaffolded config file with consistent `\n` line
  endings, and no longer overwrites an existing `candidates.json` when a custom seed is
  provided.
- Test suite mocks of the LLM-client response boundary now use `MagicMock(spec=AiResponse)`
  instead of an unconstrained mock, so a renamed or removed field on the real response type
  fails the suite instead of passing silently.

### Fixed

- Fixed a latent bug in a config-loading helper that would have silently passed
  `dataclasses.MISSING` instead of a real default for any dataclass field using
  `default_factory`. No field does today, but the helper no longer assumes it never will.
- A provider exception during generation (e.g. a rate limit) no longer crashes the whole
  benchmark run; it is stored as a placeholder result and picked up by `--retry-errors`,
  the same as a timeout.
- A malformed or non-dict line in `progress.jsonl` no longer makes the whole log
  unreadable; the offending line is skipped instead.
- A failure on one side of an evaluation (task or global) no longer discards a score
  already computed on the other side.
- Fixed HTML-escaping gaps in the report (group verdict names) and in the Streamlit log
  view.
- `verdict._source_cell` now reports "mixed" for a pool of traces with an unknown flag
  beside a known one, instead of misreporting it as "raw".

### Added

- `api.warm_up_for_run`/model residency tracking: a local model preload now releases
  whatever the process last pinned before loading, and records what it loaded, instead of
  leaving two local models resident at once.

## [0.2.6] - 2026-09-08

### Fixed

- Removed the model-unloading behavior on timeout introduced in 0.2.5: it was unnecessary
  and harmful, evicting a model that was never actually orphaned by the timeout.

## [0.2.5] - 2026-09-08

### Fixed

- Fixed a timeout handling issue that was ending in an incorrect local model unload.

## [0.2.4] - 2026-09-06

### Fixed

- Generation events no longer receive an unwanted automatic retry from the underlying AI
  client.

## [0.2.3] - 2026-09-05

### Added

- `--retry-errors`: resume a run and also redo the steps whose stored result is a
  timeout or judge-failure placeholder, instead of only the steps missing from the log.

### Fixed

- Aligned the underlying AI client's own call timeout with PromptTestEnv's response
  timeout.

## [0.2.2] - 2026-09-03

### Added

- A duplicate button for candidate and test case rows in the project editor.

### Fixed

- Fixed a bug in the project editor GUI.
- The `assert` judge type's evaluation lambda now runs against a fixed, explicit set of
  allowed namespaces instead of an unrestricted one.

## [0.2.1] - 2026-08-31

### Fixed

- Logging now goes to stderr instead of stdout when the tool runs as an MCP server, so it
  no longer corrupts the MCP protocol stream on stdout.

## [0.2.0] - 2026-08-30

Initial tagged release.

### Added

- Core benchmarking pipeline: generation, evaluation, and optional reasoning-analysis
  phases over a project directory, resumable via `progress.jsonl`.
- CLI, MCP server, and two Streamlit GUIs (benchmark runner and project editor) as the
  four interfaces to the tool.
- Headless project editing (`show`/`edit`) through the CLI, the MCP server, and the
  library API, with the same patch-based validation across all three.
- `json` output mode alongside `html`, `md`, and `winner_only`, described by
  `report.schema.json`.
- A warm-up phase run before each candidate's timed block, so one-off costs (SDK import,
  client construction, attachment upload) are not charged to whichever candidate happens
  to run first.
- Multi-file test case attachments.
- Bundled example/fixture projects.

### Changed

- Redesigned the reasoning-analysis taxonomy and scoring, and reworked how the resulting
  statistics are computed and aggregated.
- Refined the verdict judge's payload, the default project, the trace analysis output,
  the prompt templates, and the rendered report's formatting.
- Enforced score ranges and the `-1` "not measured" sentinel consistently across
  evaluators.
- Aligned the codebase with the project's internal development conventions and removed
  several code smells identified during that pass.
- Expanded the unit test suite and the documentation.

### Fixed

- Fixed assorted GUI issues in the benchmark runner.

[0.2.8]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Kuig/PromptTestEnv/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Kuig/PromptTestEnv/releases/tag/v0.2.0
