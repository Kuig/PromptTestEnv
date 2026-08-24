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

### E. Reasoning Analysis (`judge_config.json`)

This optional feature analyses the internal reasoning trace (the "thinking" output) of models that support it. When enabled, it gives insight not just into *what* a model answers, but *how* it thinks.

> [!WARNING]
> Reasoning analysis makes **2 additional LLM calls per candidate × test × repetition**. For a project with 3 candidates, 5 tests, and 3 repetitions, that is up to **90 extra LLM calls**. Plan your API budget accordingly.

**Enable it** by adding to `judge_config.json`:
```json
"reasoning_analysis": true
```
The feature is silently skipped for candidates that do not produce reasoning output (i.e., those with `"thinking": false`).

**`reasoning_judge` Options:**
*Configures the LLM that analyses the reasoning trace.*
- `provider`, `model`, `temperature`, `thinking`: Standard LLM parameters. Recommended: fast model with `thinking: false` (e.g. `gemini-2.5-flash`).
- `reasoning_system_prompt` *(string)*: The system persona for the reasoning judge (e.g., "You are a cognitive process analyst...").
- `segmentation_template`: Prompt template for the **first call**: the LLM must segment the reasoning trace verbatim into 4 JSON fields (see below). **Must include the variable:** `{reasoning_text}`.
- `metrics_template`: Prompt template for the **second call**: the LLM must extract 3 numeric metrics. **Must include the variables:** `{reasoning_text}` and `{candidate_response}` (the final response is passed here so that `alignment_score` can be evaluated against the actual output).

**The 4 Cognitive Categories:**
| Category | Description |
|---|---|
| `interpretation` | How the model re-reads and rephrases the user's request to ensure it understood the problem. |
| `planning` | Evaluation of approach, tools, or strategy before solving the problem. |
| `pure_reasoning` | The actual problem-solving: fact recall, deduction, alternative options, self-correction. |
| `output_formulation` | Reasoning spent purely on deciding how to format and structure the final answer. |

The `segmentation_template` must instruct the model to copy all reasoning text verbatim into exactly these 4 JSON string fields without omission.

**The 3 Metrics (extracted by `metrics_template`):**
- `alt_path` *(int)*: Number of alternative solution paths explicitly considered.
- `autocorrect` *(int)*: Number of explicit self-corrections ("Wait, that's wrong…").
- `alignment_score` *(int, 1-10)*: How well the final response reflects and builds upon the reasoning.

**Reading the Stacked Bar in the HTML Report:**

The report includes a full-width CSS stacked bar chart per candidate (both in the summary stats row and inside each per-test details section):

| Color | Category |
|---|---|
| 🔵 Blue | Interpretation |
| 🟠 Orange | Planning |
| 🟢 Green | Pure Reasoning |
| 🟣 Purple | Output Formulation |

A large **green (Pure Reasoning) segment** indicates the model dedicated most of its cognitive resources to actually solving the problem. A large **blue (Interpretation)** segment may indicate the model is over-analyzing simple prompts. A large **purple (Output Formulation)** segment may indicate excessive focus on presentation.

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
