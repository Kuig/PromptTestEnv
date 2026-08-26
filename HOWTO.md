# How to Create and Customize Tests in PromptTestEnv

This guide explains how to configure a benchmark project from scratch. It covers all the available options in the configuration files and shows how you can creatively "hack" the framework for tasks beyond simple benchmarking, such as creating a bulk **Paper Reviewer**.

## 1. Project Initialization

To create a new test project, use the CLI initialization command:
```powershell
prompttestenv init Projects/MyNewTest
```
This command scaffolds a new directory (`Projects/MyNewTest`) with the default configuration files. You will mainly interact with these four files:

---

## 2. Configuration Files & Options

### A. `candidates.json`
This file defines the LLM configurations (the "candidates") you want to compare or run. It is a JSON array of objects.

**Available properties per candidate:**
- `name` *(string, required)*: The display name of the candidate in the reports.
- `provider` *(string, required)*: The AI provider (e.g., `"google"`, `"ollama"`, `"openai"`, `"anthropic"` depending on your `UnifiedAiClient` support).
- `model` *(string, required)*: The specific model ID (e.g., `"gemini-2.5-flash"`, `"gemma4:e2b"`).
- `temperature` *(float, optional)*: Sampling temperature (default: `0.7`).
- `disable_safety` *(boolean, optional)*: Disables safety filters if supported by the provider (default: `False`).
- `thinking` *(boolean | string, optional)*: Controls the "Thinking" or reasoning mode. Can be `true`, `false`, or `"default"`.
- `system_prompt_file` *(string, optional)*: A relative path to a text file inside the `system_prompts/` folder. This allows you to assign different personas to different candidates.

### B. `test_cases.json`
This file contains the actual prompts (the tasks) that the candidates will execute. It is a JSON array of objects.

**Available properties per test case:**
- `id` *(string, required)*: A unique identifier for the test case (no spaces recommended).
- `prompt` *(string, required)*: The user prompt or instruction to send to the candidate.
- `criteria` *(string, required)*: The criteria text used for evaluation. Depending on the `judge_type`, this represents:
  - For `llm-judge`: The guidelines text given to the judge.
  - For `similarity`: The expected target text to compare the response against using cosine similarity.
  - For `assert`: A single-line vanilla Python lambda expression returning a `(score, reasoning)` tuple (e.g. `s: (10, 'OK') if s.count(',') == 2 else (1, 'Failed')`).
- `judge_type` *(string, optional)*: The mode used to judge the candidate response. Can be:
  - `"llm-judge"` *(default)*: Sends the response and criteria to the judge LLM.
  - `"similarity"`: Calculates the embedding cosine similarity between the response and the target criteria, scaled 1 to 10.
  - `"assert"`: Evaluates the criteria expression directly in Python.
- `file` *(string, optional)*: A relative path to a file inside the `test_files/` folder (e.g., `"test_files/report.pdf"`). The file will be attached to the prompt.

Every evaluator normally returns a score of 1-10, or the shared `-1` "not measured" sentinel when it failed to produce one at all (excluded from every average, never counted as a zero) — a judge LLM call that errored, or a `similarity`/`assert` evaluation that raised an exception. `assert` is the one place the framework does **not** clamp the score for you: your lambda is your own arbitrary Python, so keeping its return value in the range you intend is your job, and you may return `-1` yourself to mark a specific response "not applicable/not measured". `similarity`'s score, by contrast, is computed by the framework, so it is always clamped to 1-10 on your behalf.
- `group` *(string, optional)*: Assigns the test case to a specific category (e.g., `"Coding"`, `"Creative Writing"`). Used in combination with `group_verdicts` (default: `"Default group"`).

### C. `judge_config.json`
This is the core engine configuration file. It dictates how the framework behaves, how many times tests are repeated, and defines the LLMs used for judging and summarizing.

**Root Options:**
- `repetitions` *(int)*: How many times each candidate should execute each test case (default: `5`). Useful for checking consistency.
- `repetition_delay_seconds` *(float)*: Delay between generation requests (default: `2.0`).
- `evaluation_delay_seconds` *(float)*: Delay between judge evaluation requests (default: `2.0`).
- `max_response_timeout_seconds` *(float)*: Maximum wait time for the judge (default: `240.0`).
- `pass_media_to_judge` *(boolean)*: If `true`, the media file attached in the test case is also sent to the judge as "ground truth" context.
- `group_verdicts` *(boolean)*: If `true`, the framework will generate specific verdicts for each `group` defined in `test_cases.json`, plus a global overview.
- `reasoning_analysis` *(string)*: How much of the run the reasoning-analysis phase covers: `"none"`, `"best"` or `"all"`. See section E below.

**`test_judge` Options:**
*Configures the LLM that scores individual responses.*
- `provider`, `model`, `temperature`, `disable_safety`, `thinking`: Standard LLM parameters (typically set to a smart model with low temperature, e.g., `0.2`).
- `evaluation_system_prompt`: The system persona for the judge (e.g., "You are an impartial AI judge...").
- `evaluation_template`: The template defining how the candidate's response, the original prompt, and the criteria are presented to the judge. **Must** expect a JSON output with `score` (1-10) and `reasoning`. **Must include the variables:** `{user_prompt}`, `{candidate_response}`, and `{criteria}`.

**`similarity_judge` Options:**
*Configures the local embedding model used for similarity score calculations.*
- `provider` *(string)*: The AI provider supporting embeddings (e.g. `"ollama"`).
- `model` *(string)*: The name of the embedding model (e.g. `"bge-m3"`).

**`verdict_judge` Options:**
*Configures the LLM that writes the final markdown/HTML report.*
- `provider`, `model`, `temperature`, `disable_safety`, `thinking`: Standard LLM parameters.
- `verdict_system_prompt`: The system persona for the analyst writing the report.
- `verdict_template`: The template used to generate a single global verdict (if `group_verdicts` is false) or the individual group verdicts. **Must include the variables:** `{summary_data}` and `{global_criteria}`.
- `global_verdict_template`: The template used to generate the final overarching conclusion when `group_verdicts` is true. **Must include the variables:** `{group_verdicts_data}` and `{global_criteria}`.

### D. `global_criteria.json`
A structured JSON file that defines the global rules applied to all test cases, with support for the different evaluation modes.

**Required JSON fields:**
- `mode` *(string)*: The active global evaluation mode. Can be `"llm-judge"`, `"similarity"`, `"assert"`, or `"none"` (to disable global criteria scoring entirely).
- `llm_judge_criteria` *(string)*: The text description of the rules for the LLM judge (e.g., "1. Courtesy: The language must always be polite.").
- `similarity_criteria` *(string)*: The target reference text for similarity-based global evaluation.
- `assert_criteria` *(string)*: The python lambda expression for assertion-based global evaluation (e.g. `s: (10, 'OK') if len(s) > 10 else (1, 'Too short')`).

---

### E. Reasoning Analysis (`judge_config.json` + root `config.json`)

This optional feature analyses the internal reasoning trace (the "thinking" output) of models that support it. When enabled, it gives insight not just into *what* a model answers, but *how* it works its way there.

**Enable it** by adding to `judge_config.json`:
```json
"reasoning_analysis": "best"
```

The setting chooses how much of the run gets measured, because the cost scales with `repetitions`:

| Value | What gets analysed | Cost |
|---|---|---|
| `"none"` | nothing. The phase does not run. | zero |
| `"best"` | the highest-scoring repetition of each candidate x test case | 1 trace per test case |
| `"all"` | every repetition of every test case | `repetitions` traces per test case |

`"best"` is the recommended starting point once you want the profile at all: it measures exactly the repetition whose trace the report draws anyway, so nothing on screen is lost, at a fifth of the calls under the default `repetitions: 5`. A scaffolded project starts with `"none"`, since not every benchmark involves thinking-enabled candidates — switch it on explicitly when it does.

Be aware of what `"best"` changes, though: the profile then describes how a model reasons **when it succeeds**, not how it reasons on a typical run. That is a legitimate question to ask, but it is a different one, so the report and the verdict payload both state which scope produced the figures, and figures from the two scopes must not be compared. Use `"all"` when you want the unfiltered picture, or when repetitions disagree with each other and you want to know why.

Switching scope is free and never wastes work. Analyses already in `progress.jsonl` are kept, so narrowing from `"all"` to `"best"` throws nothing away, and widening later analyses only the repetitions that are missing.

Analysis is silently skipped for candidates that produce no reasoning output.

#### How it works

1. The trace is split into sentence-sized **units** procedurally, in Python. No LLM is involved, so the text never gets rewritten: units are character offsets into the stored trace, which makes full coverage and zero overlap structural guarantees rather than instructions a judge may ignore.
2. A judge scores **every unit on every dimension**, on a 0 to 3 scale, and is given the task the candidate was asked to perform so it can tell restating the request apart from reasoning about it.
3. One further call extracts the metrics, citing unit ids as evidence.
4. Two more metrics are computed with no LLM call at all.

#### The three dimensions

| Dimension | What it asks of each sentence |
|---|---|
| `framing` | Is it about understanding the problem: restating the request, weighing constraints, choosing an approach? |
| `solving` | Does it advance the answer: recalling a fact, deducing, computing, drafting, verifying a result? |
| `presentation` | Is it about the form of the answer: format, structure, length, tone, staying in an assigned persona? |

These are **not** a partition, and this is the whole point. A sentence such as *"So it is: Paris, Berlin, Rome, though I should say it in character"* genuinely does two things at once, and forcing a single label made the old four-category scheme assign it almost arbitrarily. Each dimension therefore gets its own independent coverage in the 0 to 100% range, and the three do not add up to 100%.

Their sum is reported separately as **density**: how many concerns the trace carries per unit of text. A density near 1.0 means the model does one thing at a time; well above 1.0 means it interleaves them.

The dimensions, their definitions, their colours and the prompts that apply them live in the **root `config.json`**, not in `judge_config.json`. They are the measurement instrument: if each benchmark could redefine them, no two reports would be comparable. Change them there and every project changes together, and re-run `analyze` to recompute.

#### `reasoning_judge` options (`judge_config.json`)

*Only the call parameters. What the judge is asked comes from `config.json`.*

- `provider`, `model`, `temperature`, `thinking`: standard LLM parameters. A fast model with `thinking: false` is the right choice.
- `context_size` *(int, optional)*: context window to allocate. **Set this for a local judge.** Ollama sizes its context at load time, and a raw thinking trace from a local model easily exceeds the default, in which case it is truncated silently and the analysis looks plausible but is wrong.
- `dimension_mode` *("split" | "joint", optional)*: `split` (the default) asks one single-concept question per dimension, which is what keeps the task tractable for a small local judge. `joint` asks for all three at once in a single call, sending the trace only once, which is worth it for very long traces or a strong judge.
- `reliability_k` *(int, optional)*: repeat the scoring k times and average. Off by default. Only meaningful with `temperature` above 0, since at low temperature the passes are near-identical and you pay k times for the same answer.
- `max_units_per_call` *(int, optional)*: window size for long traces. Above this, units are scored in successive windows, each carrying the two preceding units as unscored context.

#### The metrics

Counts are derived from the sentence ids the judge cites, so every one of them is traceable to a sentence in the report rather than being an unverifiable number.

- `alt_path` *(int)*: distinct alternative approaches introduced.
- `autocorrect` *(int)*: explicit retractions or revisions.
- `alignment_score` *(int, 1-10)*: how faithfully the final response follows the conclusions the trace reached. Below 8, the judge must cite the sentences the response failed to honour, and those are flagged in the trace view.
- `repetition_rate` *(float)*: share of repeated word trigrams, computed without an LLM. Catches the rumination that raw local traces are prone to.
- `trace_response_drift` *(float)*: cosine similarity between trace and response embeddings, via the `similarity_judge` settings. An objective companion to `alignment_score`, which tends to saturate near 10.

A value of `-1` means **not measured** (a judge call failed), and it is excluded from every average rather than counted as a zero. `0` is a real measurement.

#### Cost per point (not part of the reasoning analysis)

Every candidate carries a reasoning-token cost figure regardless of whether `reasoning_analysis` runs at all, since it needs only the thinking-token count and the task score, both recorded for every repetition no matter what. It shows up in two places, computed two different ways:

- **Per test case**, on the token line right under that test's response: the **mean of each repetition's own ratio** (that repetition's own thinking tokens divided by its own score), with a standard deviation across the repetitions of that one test.
- **Per candidate**, in the header STATS row, next to the pooled token counts: the **ratio of the pooled means** (mean thinking tokens divided by mean task score), a single figure.

The two are different statistics and will not generally agree. They coincide only when every repetition scores the same; the more the score varies, the more a repetition that failed cheaply pulls the pooled figure up without pulling the per-test-case mean up nearly as much, since that repetition's own ratio (thinking tokens over a low score) is large on its own. Both answer the question the raw token count does not, which is whether a candidate got anything back for the thinking it was billed for; **lower is cheaper** on both.

> [!WARNING]
> Read either figure as a diagnostic, never as a ranking. The task score floors at 1 rather than 0 and spans only 10x, while thinking tokens span far more, so the figure is largely a token count in disguise: **a candidate that thinks little and fails scores well on it.** A model spending 69 tokens to earn a 1 prices at 69 per point, which looks better than one spending 1051 to earn a 10. Rank on the scores, and use this to explain what the thinking cost.

#### Cost

Per trace analysed: 3 scoring calls plus 1 metrics call in `split` mode, or 2 calls in `joint` mode. The judge only ever returns a short list of numbers, never a copy of the trace, so the expensive output tokens stay near zero however long the trace is.

How many traces that is comes from the scope above: one per candidate x test case under `"best"`, or one per candidate x test case x repetition under `"all"`.

Because it reads traces that generation already stored, this phase is **re-runnable on its own**:

```powershell
prompttestenv analyze Projects/MyBenchmark --force-reanalyze
```

Retuning the schema costs only these calls, never a re-run of the candidates: the reasoning settings are deliberately excluded from the resume hash.

#### Reading the report

Each candidate gets one horizontal bar per dimension, plus the density figure, the cost in thinking tokens per point of score, and the scope the figures were measured under with the number of traces behind them. Under each test case, the **full trace is shown colour-coded in place**: every sentence is tinted by its dominant dimension, shaded by intensity, with all three scores on hover. Sentences cited as evidence carry a marker for an alternative, a self-correction, or a conclusion the response contradicted. Because the units are offsets, what you see is the trace itself, and you can check the analysis against the actual words.

> [!WARNING]
> **Some providers do not give you the raw chain of thought.** Google returns a *summary* the model writes about its own thinking, typically around half the length its billed thinking tokens imply, while Ollama, Anthropic and the OpenAI-compatible providers return the raw transcript. The report flags a summarised trace and withholds absolute token attribution for it. Trace length, composition and self-correction counts partly reflect the summariser, so do not rank a summarised trace against a raw one on those figures.

---

## 3. Hacking the Framework (Alternative Use Cases)

While PromptTestEnv is designed for benchmarking, its architecture (Parallel Generation ➔ Judge Evaluation ➔ Aggregated Report) makes it an excellent engine for **bulk processing and analysis tasks**.

### Example: The "Paper Reviewer" Hack

Imagine you have several research papers (PDFs) and you want a comprehensive, multi-faceted review for each one. Instead of having models compete, you use the framework to build a **panel of experts**.

1. **Candidates as "Experts"**: In `candidates.json`, define multiple candidates that act as different specialized reviewers. For example, one candidate focuses on "Methodology", another on "Novelty", and another on "Grammar/Clarity". You can assign different `system_prompt_file`s to give each candidate its specific persona.
2. **One Group per Paper**: In `test_cases.json`, create a single test case for each paper and assign each to a **unique `group`** (e.g., `"group": "Paper_1_AttentionIsAllYouNeed"`). This is crucial: it ensures the final judge only focuses on one paper at a time, preventing cognitive overload.
3. **The Test Judge as a Scorer**: The `test_judge` evaluates how well each expert performed their specific task and provides a score.
4. **The Verdict Judge as the "Editor in Chief"**: Enable `group_verdicts: true`. The `verdict_judge` will now receive the analyses from *all* your expert candidates for a *single* paper. Change the `verdict_template` to instruct the LLM to stitch these different perspectives together into one cohesive, comprehensive final review for that specific paper.
5. **The Global Verdict**: Since the heavy lifting is done in the group verdicts, the `global_verdict_template` simply acts as a high-level summary or an index of all the reviewed papers.

**By running the project:**
- The framework will pass each paper to all your "expert" candidates in parallel.
- The verdict judge will synthesize the diverse perspectives into a single, highly detailed review per paper.
- You obtain a massive amount of analytical work fully automated and structured!
