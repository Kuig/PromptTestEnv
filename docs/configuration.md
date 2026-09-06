# Configuration Reference

Key-by-key reference for the two files that live outside a benchmark project:
the root `config.json` and `secrets.json`.

The four files that describe a single benchmark (`candidates.json`,
`test_cases.json`, `judge_config.json`, `global_criteria.json`) are covered in
`HOWTO.md`, which walks through authoring one.

## `config.json`

`config.json` holds the **measurement instrument**: the reasoning taxonomy, the
prompts that apply it, the sentence-splitting parameters, the provider-locality
table, the warm-up switch, and the metadata block appended to the verdict
judge's system prompt. These have to be identical across benchmarks, otherwise
two reports are not comparable, so they are deliberately not per-project and not
reachable from a project's `judge_config.json`.

A file called `config.json` is usually git-ignored, with a `.example` template
committed beside it. This one is the opposite: it is **versioned** and ships no
`.example`. It contains no credentials, only the definition of the instrument,
which must be the same for anyone who clones the repo.

### Resolution order

`config.json` is looked up in three places, in order, and the first readable one
wins:

1. the current working directory,
2. the repository root,
3. the read-only copy shipped inside the package, at
   `prompttestenv/templates/default_config.json`.

The third step is what makes an install from `requirements_prod.txt` work, where
there is no repo root. The two copies are kept identical by a unit test.

Every field has a default, so a missing, unreadable or partial file degrades
rather than failing: unknown keys are ignored and omitted keys fall back to the
value in the table below.

### `reasoning_schema`

The taxonomy and the judge prompts that apply it. Changing anything here changes
what the reasoning figures mean, so re-run `prompttestenv analyze` afterwards.

| Key | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | string | `""` | System persona for every reasoning-judge call. |
| `intensity_scale` | int | `3` | Top of the per-unit scoring scale. A unit scores `0` to this value on each dimension. |
| `dimensions` | list of objects | `[]` | The axes of the profile, in declaration order. See below. |
| `dimension_template` | string | `""` | Prompt for one single-dimension question, used in `split` mode. |
| `joint_template` | string | `""` | Prompt asking for all dimensions at once, used in `joint` mode. |
| `metrics_template` | string | `""` | Prompt for the evidence-anchored metrics call. |

Each entry of `dimensions` is an object:

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | `""` | Dimension name. Must be one of `framing`, `solving`, `presentation`. |
| `color` | string | `"#888888"` | Hex colour used to tint the trace in the HTML report. |
| `definition` | string | `""` | What the judge is asked to look for in a sentence. |

The shipped set is `framing` (`#4FC3F7`), `solving` (`#81C784`) and
`presentation` (`#BA68C8`). The **names are not free text**: they are mirrored as
fields on the stored analysis records, so `config.json` may reword a dimension's
definition or change its colour, but inventing a fourth dimension is a code
change, not a configuration change.

A short stamp of this section (the dimension names plus a hash of the prompts and
definitions) is written onto every reasoning record, which is what lets a report
notice that it mixes figures produced by two different schema versions.

### `unit_splitting`

Parameters of the procedural sentence splitter. No LLM is involved.

| Key | Type | Default | Description |
|---|---|---|---|
| `min_unit_chars` | int | `15` | Fragments shorter than this are merged into a neighbour, forward when the fragment opens its own line (a `2.` list marker, say), otherwise backward. |
| `abbreviations` | list of strings | `[]` | Tokens whose trailing full stop does not end a sentence. |

The shipped abbreviation list is `e.g.`, `i.e.`, `cf.`, `etc.`, `vs.`,
`approx.`, `no.`, `fig.`, `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `Prof.`, `St.`.

Fenced code blocks and heading lines are always kept atomic, whatever these
settings say.

### `reasoning_defaults`

Fallbacks for the three optional knobs of a project's `reasoning_judge` block.
A project setting any of them to `null`, or omitting it, gets the value here.

| Key | Type | Default | Description |
|---|---|---|---|
| `dimension_mode` | string | `"split"` | `"split"` sends one single-concept question per dimension, which is what a small local judge can actually answer. `"joint"` asks for all three in one call, sending the trace once. |
| `reliability_k` | int | `1` | How many times to repeat the scoring and average. Only meaningful above `temperature` 0: at low temperature the passes are near-identical and you pay k times for the same answer. |
| `max_units_per_call` | int | `150` | Window size for long traces. Above this, units are scored in successive windows, each carrying the two preceding units as unscored context. |

### `warmup`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Whether each candidate is warmed up before its timed block. |

Every provider charges some costs exactly once per process: importing an SDK,
building a client, the DNS and TLS handshake, loading a model, uploading an
attachment. Left alone, all of it lands on whichever candidate happens to run
first, and that candidate's timing is inflated for no reason of its own. In one
real run of a project attaching a single PDF, the first candidate took 128s and
the fourth took 24s for *more* generated tokens.

So before timing each candidate, and outside the measured block, PromptTestEnv
calls the client's `warm_up()` with that candidate's model and the attachments of
the test cases it still has to run. It consumes no generation tokens, never
raises, and is a free no-op on providers with nothing to warm up, so there is no
provider to exclude and nothing to configure per project. **The reported times
are warm times**, which is what makes them comparable across candidates, and why
the switch lives here rather than in a project: two reports are only comparable
on time if both were produced the same way. The one reason to set it to `false`
is to isolate a problem while debugging.

A candidate whose repetitions are all already in `progress.jsonl` is skipped
entirely, since it will make no real call. A candidate with even one repetition
left is warmed in full, attachments included: the upload cache lives in the
provider instance, so a resumed run always starts cold no matter what the log
holds.

### `logging`

| Key | Type | Default | Description |
|---|---|---|---|
| `level` | string | `"warning"` | Verbosity of the `unified_ai_client` logger. One of `"silent"`, `"error"`, `"warning"`, `"debug"` (there is no `"info"`). |

The value is handed to the client's `set_verbosity()` once per process at every
entry point (the CLI, the MCP server, and each Streamlit app). It attaches a
single stderr handler to the client's own logger, so `"warning"` and above
surface retry activity and `"debug"` adds the full request trace; `"silent"`
suppresses all of it. An unrecognised value does not abort the run: it logs a
warning and falls back to `"warning"`.

The shipped `config.json` sets `"debug"`. This section is not part of the run
hash, so changing it never invalidates a `progress.jsonl`.

Third-party SDK chatter (httpx, the provider SDKs) is handled separately and is
always quietened, independent of this setting.

### `local_providers`

| Key | Type | Default | Description |
|---|---|---|---|
| `local_providers` | list of strings | `[]` | Providers served from the local machine. |

The shipped list is `ollama`, `lmstudio`, `llamacpp`, `script`. It drives one
decision: a local backend serves one model at a time, so the task score and the
global score are computed sequentially against a local judge and in parallel
against a remote one. Reasoning-analysis calls follow the same rule.

### `verdict_metadata`

The self-describing header the verdict payload opens with, as fifteen named
sections. It is appended to the verdict judge's system prompt, so it reads as
standing instructions rather than as more data, and with `group_verdicts` on it
is not resent once per group.

This is **not** benchmark prose. Every section describes something the *code*
emits: that the aggregate tables carry no standard deviation, that token counts
are output only, that an `assert` judge may use the full 1 to 10 range or only a
pass/fail pair. What a benchmark *means* by its scores stays the author's
business, in `verdict_template`.

| Key | Emitted |
|---|---|
| `header` | Always. Opens the block. |
| `figures` | Always. What the numbers in the tables are. |
| `score_scales` | Always. Both scores run 1 to 10, and `N/A` means not computed. |
| `cost_per_point` | When any candidate has thinking tokens. |
| `reasoning_profile` | When the reasoning profile table is actually rendered. |
| `judge_types_intro` | When any judge-type legend entry is emitted. |
| `judge_llm` | When at least one test case uses `judge_type: "llm-judge"`. |
| `judge_similarity` | When at least one test case uses `judge_type: "similarity"`. |
| `judge_assert` | When at least one test case uses `judge_type: "assert"`. |
| `judge_types_mixed` | When more than one `judge_type` is in use. |
| `global_criteria_intro` | Always. Opens the global-score legend. |
| `global_criteria_llm` | When `global_criteria.mode` is `"llm-judge"`. |
| `global_criteria_similarity` | When `global_criteria.mode` is `"similarity"`. |
| `global_criteria_assert` | When `global_criteria.mode` is `"assert"`. |
| `global_criteria_none` | When `global_criteria.mode` is `"none"`. |

Exactly one of the four `global_criteria_*` mode sections always fires, since
every project has some mode.

Every field defaults to the empty string, but an empty section that the payload
*needs* is a hard error rather than a silent omission: the figures would still
reach the judge, just without the caveats that keep it from over-claiming.

The final group-synthesis call gets no metadata at all. It never sees the results
data, only the prose of the already-written group verdicts, so there is nothing
left for the metadata to explain.

## `secrets.json`

Provider API keys. This file is read by `unified_ai_client`, not by
PromptTestEnv, **from the current working directory**: the directory you run
`prompttestenv` from, not the project directory and not the repo root. It is
git-ignored, and `secrets.json.example` is the template to copy.

`prompttestenv init` creates an empty one in the working directory if none
exists.

| Key | Environment variable |
|---|---|
| `google_api_key` | `GOOGLE_API_KEY` |
| `anthropic_api_key` | `ANTHROPIC_API_KEY` |
| `openai_api_key` | `OPENAI_API_KEY` |
| `mistral_api_key` | `MISTRAL_API_KEY` |
| `cohere_api_key` | `COHERE_API_KEY` |
| `meta_api_key` | `META_API_KEY` |
| `groq_api_key` | `GROQ_API_KEY` |
| `xai_api_key` | `XAI_API_KEY` |

Fill in only the providers you actually use and leave the rest empty. The
environment variable takes priority over the file. Local providers such as
Ollama need no key at all.

Keys are scaffolded as empty strings rather than placeholder text on purpose:
the client does not validate them, it puts them straight into the request
header, so a leftover placeholder comes back as a provider authentication error
instead of an obvious missing key.
