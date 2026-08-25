from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import prompttestenv.logger as logger

DEFAULT_GROUP = "Default group"
JUDGE_TYPE_LLM = "llm-judge"
JUDGE_TYPE_SIMILARITY = "similarity"
JUDGE_TYPE_ASSERT = "assert"
GLOBAL_MODE_NONE = "none"

# Axes of the reasoning profile. Mirrored as fields on ReasoningUnit and
# ReasoningStats (CONVENTIONS 5.3 rules out a dict here), so changing this
# tuple is a code change, not a config change: config.json may reword a
# dimension's definition or colour, but not invent a new one.
REASONING_DIMENSIONS: tuple[str, ...] = ("framing", "solving", "presentation")


def calculate_stats(scores_list: list[float], default_val: float = 0.0) -> tuple[float, float]:
    valid_scores = [s for s in scores_list if s >= 0]
    if not valid_scores:
        return default_val, 0.0
    mean = statistics.mean(valid_scores)
    try:
        std = statistics.stdev(valid_scores)
    except statistics.StatisticsError:
        std = 0.0
    return mean, std


@dataclass
class LlmResult:
    """One completed LLM call, as returned by api.get_llm_response().

    Replaces the positional tuple this used to be: reasoning_is_summary was
    the fifth field, and five positional values unpacked at three call sites
    is the same problem a dict would be.
    """

    text: str = ""
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_text: str = ""
    reasoning_is_summary: bool = False


@dataclass
class CandidatePerformance:
    scores: list[float] = field(default_factory=list)
    global_scores: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    reasoning_tokens: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    reasoning_analyses: list[dict] = field(default_factory=list)

    best_output: str = ""
    best_reason: str = ""
    best_global_reason: str = ""
    # Captured alongside best_output so the report shows the trace and the
    # analysis of the SAME repetition it shows the response for.
    best_reasoning_text: str = ""
    best_reasoning_analysis: ReasoningStats | None = None

    @property
    def score_mean(self) -> float: return calculate_stats(self.scores, default_val=0.0)[0]

    @property
    def score_std(self) -> float: return calculate_stats(self.scores, default_val=0.0)[1]
    
    @property
    def global_score_mean(self) -> float: return calculate_stats(self.global_scores, default_val=-1.0)[0]

    @property
    def global_score_std(self) -> float: return calculate_stats(self.global_scores, default_val=0.0)[1]
    
    @property
    def tokens_mean(self) -> float: return calculate_stats(self.tokens, default_val=0.0)[0]

    @property
    def tokens_std(self) -> float: return calculate_stats(self.tokens, default_val=0.0)[1]
    
    @property
    def reasoning_tokens_mean(self) -> float: return calculate_stats(self.reasoning_tokens, default_val=0.0)[0]

    @property
    def reasoning_tokens_std(self) -> float: return calculate_stats(self.reasoning_tokens, default_val=0.0)[1]

    @property
    def time_mean(self) -> float: return calculate_stats(self.times, default_val=0.0)[0]

    @property
    def time_std(self) -> float: return calculate_stats(self.times, default_val=0.0)[1]


@dataclass
class TestCaseResult:
    test_id: str
    prompt: str
    criteria: str
    group: str = DEFAULT_GROUP
    file_used: str | None = None
    media_file_path: str | None = None
    candidates_perf: dict[str, CandidatePerformance] = field(default_factory=dict)
    judge_type: str = JUDGE_TYPE_LLM


@dataclass
class Candidate:
    name: str
    provider: str
    model: str
    temperature: float = 0.7
    disable_safety: bool = False
    thinking: bool | str = "default"
    system_prompt_file: str | None = None
    resolved_system_instruction: str | None = None

    @classmethod
    def load_all(cls, project_dir: str | Path) -> list[Candidate]:
        """Load and resolve all candidate configurations from a project directory.

        Reads ``<project_dir>/candidates.json`` and, for each candidate that
        references a ``system_prompt_file``, resolves its text content from
        ``<project_dir>/system_prompts/``.

        Args:
            project_dir: Path to the benchmark project directory.

        Returns:
            List of Candidate instances.

        Raises:
            FileNotFoundError: If candidates.json does not exist in project_dir.
        """
        project_dir = Path(project_dir)
        cand_file = project_dir / "candidates.json"
        if not cand_file.exists():
            raise FileNotFoundError(f"candidates.json not found in {project_dir}")

        sys_dir = project_dir / "system_prompts"
        with open(cand_file, "r", encoding="utf-8") as f:
            raw_candidates = json.load(f)

        candidates = []
        for cand in raw_candidates:
            system_instruction = None
            if cand.get("system_prompt_file"):
                file_path = sys_dir / cand["system_prompt_file"]
                if file_path.exists():
                    system_instruction = file_path.read_text(encoding="utf-8").strip()
                else:
                    logger.log_warning(
                        f"System prompt file '{file_path}' not found for candidate '{cand['name']}'."
                    )
            candidates.append(cls(
                name=cand.get("name"),
                provider=cand.get("provider", "google"),
                model=cand.get("model"),
                temperature=cand.get("temperature", 0.7),
                disable_safety=cand.get("disable_safety", False),
                thinking=cand.get("thinking", "default"),
                system_prompt_file=cand.get("system_prompt_file"),
                resolved_system_instruction=system_instruction,
            ))
        return candidates


@dataclass
class TestCase:
    id: str
    prompt: str
    criteria: str
    file: str | None = None
    group: str = DEFAULT_GROUP
    judge_type: str = JUDGE_TYPE_LLM

    @classmethod
    def load_all(cls, project_dir: str | Path) -> list[TestCase]:
        """Load all test cases from a project directory.

        Args:
            project_dir: Path to the benchmark project directory.

        Returns:
            List of TestCase instances.

        Raises:
            FileNotFoundError: If test_cases.json does not exist in project_dir.
        """
        test_file = Path(project_dir) / "test_cases.json"
        if not test_file.exists():
            raise FileNotFoundError(f"test_cases.json not found in {project_dir}")

        with open(test_file, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
        return [
            cls(
                id=tc.get("id", "N/A"),
                prompt=tc.get("prompt", ""),
                criteria=tc.get("criteria", ""),
                file=tc.get("file"),
                group=tc.get("group", DEFAULT_GROUP),
                judge_type=tc.get("judge_type", JUDGE_TYPE_LLM),
            )
            for tc in raw_cases
        ]


@dataclass
class TestJudgeSettings:
    provider: str = "google"
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.2
    disable_safety: bool = True
    thinking: bool | str = "default"
    evaluation_system_prompt: str = ""
    evaluation_template: str = ""


@dataclass
class SimilarityJudgeSettings:
    """Configuration for the embedding-based similarity judge."""

    provider: str = "ollama"
    model: str = "bge-m3"


@dataclass
class ReasoningJudgeSettings:
    """Which LLM analyses the reasoning traces, and how it is called.

    Only the call parameters live here. What the judge is asked (the dimensions,
    their definitions, the prompt templates) is the measurement instrument and
    lives in the root config.json, so that two projects produce comparable
    profiles. The three optional knobs below default to config.json's
    reasoning_defaults when left as None.
    """

    provider: str = "google"
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.2
    thinking: bool | str = False
    # Ollama allocates its context window at load time; leaving this unset lets
    # the server default silently truncate a long raw trace.
    context_size: int | None = None
    dimension_mode: str | None = None
    reliability_k: int | None = None
    max_units_per_call: int | None = None



@dataclass
class ReasoningUnit:
    """One sentence-sized span of a reasoning trace, scored on every dimension.

    ``start``/``end`` are character offsets into the trace stored in the ``gen``
    event, never a copy of the text: the trace is split procedurally and never
    round-trips through an LLM, which is what makes full coverage and zero
    overlap structural properties rather than prompt instructions.

    The per-dimension scores are independent intensities, not shares of a whole:
    a sentence that both weighs an approach and produces a step of the answer
    legitimately scores high on ``framing`` and ``solving`` at once.
    """

    start: int
    end: int
    framing: float = 0.0
    solving: float = 0.0
    presentation: float = 0.0

    @property
    def length(self) -> int:
        """Character length of the span, used as its weight when aggregating."""
        return max(0, self.end - self.start)

    def intensity(self, dimension: str) -> float:
        """Return this unit's score on one dimension.

        Args:
            dimension: One of REASONING_DIMENSIONS.

        Returns:
            The intensity, on the scale declared by the reasoning schema.
        """
        return float(getattr(self, dimension))

    def set_intensity(self, dimension: str, value: float) -> None:
        """Set this unit's score on one dimension.

        Args:
            dimension: One of REASONING_DIMENSIONS.
            value: The intensity to store.
        """
        setattr(self, dimension, float(value))


@dataclass
class ReasoningStats:
    """Reasoning analysis for a single candidate repetition.

    Every numeric field uses ``-1`` for "not measured" (a failed or skipped judge
    call), matching the sentinel ``calculate_stats`` already filters out. This
    matters most for the coverages, where ``0.0`` is a legitimate measurement:
    a trace really can spend none of itself on a dimension, and that is not the
    same as never having asked.
    """

    units: list[ReasoningUnit] = field(default_factory=list)

    # Independent per-dimension coverage in [0, 1]; they do not sum to 1.
    coverage_framing: float = -1.0
    coverage_solving: float = -1.0
    coverage_presentation: float = -1.0

    # Sum of the coverages: how many concerns the trace carries per unit of text.
    density: float = -1.0

    # Evidence-anchored judge metrics.
    alt_path: int = -1
    autocorrect: int = -1
    alignment_score: int = -1
    alt_path_units: list[int] = field(default_factory=list)
    autocorrect_units: list[int] = field(default_factory=list)
    alignment_evidence: list[int] = field(default_factory=list)

    # Procedural metrics, computed without any LLM call.
    repetition_rate: float = -1.0
    trace_response_drift: float = -1.0

    reasoning_is_summary: bool | None = None
    schema_stamp: str = ""

    def coverage(self, dimension: str) -> float:
        """Return the coverage of one dimension.

        Args:
            dimension: One of REASONING_DIMENSIONS.

        Returns:
            Coverage in [0, 1], or -1.0 when the dimension was not measured.
        """
        return float(getattr(self, f"coverage_{dimension}"))

    def set_coverage(self, dimension: str, value: float) -> None:
        """Set the coverage of one dimension.

        Args:
            dimension: One of REASONING_DIMENSIONS.
            value: Coverage in [0, 1], or -1.0 for "not measured".
        """
        setattr(self, f"coverage_{dimension}", float(value))

    def to_dict(self) -> dict:
        """Serialize to a plain dict for progress.jsonl storage.

        Returns:
            Dict representation of all fields, units included.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ReasoningStats:
        """Rebuild a ReasoningStats from a stored progress.jsonl payload.

        Args:
            data: Dict previously produced by to_dict().

        Returns:
            A populated ReasoningStats, ignoring unknown keys so that a log
            written by a different schema still renders.
        """
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["units"] = [
            ReasoningUnit(**{k: v for k, v in u.items() if k in {f.name for f in fields(ReasoningUnit)}})
            for u in data.get("units", [])
        ]
        return cls(**payload)


@dataclass
class VerdictJudgeSettings:
    provider: str = "google"
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.2
    disable_safety: bool = True
    thinking: bool | str = "default"
    verdict_system_prompt: str = ""
    verdict_template: str = ""
    global_verdict_template: str = ""


@dataclass
class GlobalCriteria:
    mode: str = JUDGE_TYPE_LLM  # "llm-judge", "similarity", "assert", "none"
    llm_judge_criteria: str = ""
    similarity_criteria: str = ""
    assert_criteria: str = ""

    def to_verdict_string(self) -> str:
        if self.mode == JUDGE_TYPE_LLM:
            return self.llm_judge_criteria
        elif self.mode == JUDGE_TYPE_SIMILARITY:
            return f"Cosine similarity (scale 1-10) between candidate response and target response: {self.similarity_criteria}"
        elif self.mode == JUDGE_TYPE_ASSERT:
            return f"Global evaluation via programmatic assertion: {self.assert_criteria}"
        else:
            return "Global evaluation disabled."

    @classmethod
    def load(cls, project_dir: str | Path) -> GlobalCriteria:
        """Load global evaluation criteria from a project directory.

        Unlike the other project config files, a missing or unreadable
        global_criteria.json gracefully degrades to mode="none" (global
        criteria scoring disabled) instead of raising — mode="none" is
        itself a legitimate, explicit way to opt out of global scoring, so
        treating "file absent" the same way is intentional, not an error.

        Args:
            project_dir: Path to the benchmark project directory.

        Returns:
            GlobalCriteria instance.
        """
        path = Path(project_dir) / "global_criteria.json"
        if not path.exists():
            logger.log_error(f"{path} not found. Using default 'none' mode.")
            return cls(mode=GLOBAL_MODE_NONE)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                mode=data.get("mode", JUDGE_TYPE_LLM),
                llm_judge_criteria=data.get("llm_judge_criteria", ""),
                similarity_criteria=data.get("similarity_criteria", ""),
                assert_criteria=data.get("assert_criteria", ""),
            )
        except Exception as exc:
            logger.log_error(f"Error reading {path}: {exc}")
            return cls(mode=GLOBAL_MODE_NONE)


def _settings_from_dict(cls: type, data: dict):
    """Populate a settings dataclass from a dict, using its own field defaults for missing keys.

    Assumes every field of `cls` has a plain literal default (no default_factory) —
    true for all 4 current call sites (TestJudgeSettings, SimilarityJudgeSettings,
    VerdictJudgeSettings, ReasoningJudgeSettings).

    Args:
        cls: The settings dataclass to construct.
        data: Raw dict (e.g. one sub-section of judge_config.json).

    Returns:
        A populated instance of cls.
    """
    return cls(**{f.name: data.get(f.name, f.default) for f in fields(cls)})


@dataclass
class JudgeConfig:
    """Top-level configuration for a benchmark project's judge settings."""

    repetitions: int = 5
    repetition_delay_seconds: float = 2.0
    evaluation_delay_seconds: float = 2.0
    max_response_timeout_seconds: float = 240.0
    pass_media_to_judge: bool = False
    group_verdicts: bool = False
    reasoning_analysis: bool = False
    global_criteria: GlobalCriteria = field(default_factory=GlobalCriteria)
    test_judge: TestJudgeSettings = field(default_factory=TestJudgeSettings)
    similarity_judge: SimilarityJudgeSettings = field(default_factory=SimilarityJudgeSettings)
    verdict_judge: VerdictJudgeSettings = field(default_factory=VerdictJudgeSettings)
    reasoning_judge: ReasoningJudgeSettings = field(default_factory=ReasoningJudgeSettings)

    @classmethod
    def from_dict(cls, data: dict) -> JudgeConfig:
        """Construct a JudgeConfig from a raw JSON dict.

        Args:
            data: Dictionary loaded from judge_config.json.

        Returns:
            Populated JudgeConfig instance.
        """
        tj_data = data.get("test_judge", {})
        sj_data = data.get("similarity_judge", {})
        vj_data = data.get("verdict_judge", {})
        rj_data = data.get("reasoning_judge", {})

        test_judge = _settings_from_dict(TestJudgeSettings, tj_data)
        similarity_judge = _settings_from_dict(SimilarityJudgeSettings, sj_data)
        verdict_judge = _settings_from_dict(VerdictJudgeSettings, vj_data)

        reasoning_judge = _settings_from_dict(ReasoningJudgeSettings, rj_data)

        gc_data = data.get("global_criteria", {})
        if isinstance(gc_data, dict):
            global_criteria = GlobalCriteria(
                mode=gc_data.get("mode", JUDGE_TYPE_LLM),
                llm_judge_criteria=gc_data.get("llm_judge_criteria", ""),
                similarity_criteria=gc_data.get("similarity_criteria", ""),
                assert_criteria=gc_data.get("assert_criteria", ""),
            )
        elif isinstance(gc_data, GlobalCriteria):
            global_criteria = gc_data
        else:
            global_criteria = GlobalCriteria(
                mode=JUDGE_TYPE_LLM if gc_data else GLOBAL_MODE_NONE,
                llm_judge_criteria=str(gc_data),
            )

        return cls(
            repetitions=data.get("repetitions", 5),
            repetition_delay_seconds=data.get("repetition_delay_seconds", 2.0),
            evaluation_delay_seconds=data.get("evaluation_delay_seconds", 2.0),
            max_response_timeout_seconds=data.get("max_response_timeout_seconds", 240.0),
            pass_media_to_judge=data.get("pass_media_to_judge", False),
            group_verdicts=data.get("group_verdicts", False),
            reasoning_analysis=data.get("reasoning_analysis", False),
            global_criteria=global_criteria,
            test_judge=test_judge,
            similarity_judge=similarity_judge,
            verdict_judge=verdict_judge,
            reasoning_judge=reasoning_judge,
        )

    @classmethod
    def load(cls, project_dir: str | Path) -> JudgeConfig:
        """Load judge configuration from a project directory.

        Args:
            project_dir: Path to the benchmark project directory.

        Returns:
            Populated JudgeConfig instance.

        Raises:
            FileNotFoundError: If judge_config.json does not exist in project_dir.
        """
        path = Path(project_dir) / "judge_config.json"
        if not path.exists():
            raise FileNotFoundError(f"judge_config.json not found in {project_dir}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass(frozen=True)
class ProgressState:
    """Read-only snapshot of a project's resume state, built once from progress.jsonl.

    ``events`` intentionally stays a list of raw dicts rather than typed event
    dataclasses: it is a direct in-memory mirror of heterogeneous JSONL log
    lines (differently shaped per ``type``), which is exactly the "dynamic/opaque
    data" case dict is meant for.

    ``gen_events``/``eval_events``/``reasoning_events`` index the "gen"/"eval"/"reasoning" event dicts
    already in ``events`` by their ``(cand_id, test_id, rep)`` key — built in
    the same pass as ``completed_gen``/``completed_eval`` — so a resumed key
    can be looked up in O(1) instead of scanning ``events`` linearly.
    """

    hash_match: bool = True
    completed_gen: set[tuple[str, str, int]] = field(default_factory=set)
    completed_eval: set[tuple[str, str, int]] = field(default_factory=set)
    events: list[dict] = field(default_factory=list)
    verdict: str | None = None
    last_hash: str | None = None
    gen_events: dict[tuple[str, str, int], dict] = field(default_factory=dict)
    eval_events: dict[tuple[str, str, int], dict] = field(default_factory=dict)
    reasoning_events: dict[tuple[str, str, int], dict] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        project_dir: str | Path,
        force_restart: bool = False,
        readonly: bool = False,
    ) -> ProgressState:
        """Initialize or read a project's progress.jsonl resume log.

        If force_restart is True, or the stored config hash no longer matches
        the current config files, the existing progress.jsonl is renamed to
        progress.jsonl.bak (or, if force_restart, simply removed) instead of
        being read.

        ``readonly`` suppresses all of that: nothing is renamed, removed or
        created, and a hash mismatch is reported through ``hash_match`` with
        the events still loaded. Commands that only consume the log (rendering
        a report, analysing stored traces) must use it, since they do not
        depend on the config still matching and must not destroy the log they
        were asked to read.

        Args:
            project_dir: Path to the benchmark project directory.
            force_restart: If True, discard any existing progress.jsonl.
            readonly: If True, never write to or rename the log.

        Returns:
            A ProgressState snapshot. Outside readonly mode, a stored config
            hash that does not match the current one leaves ``hash_match``
            False and every other field at its default (empty/None), so
            callers must check ``hash_match`` before trusting the snapshot.
        """
        import os

        from prompttestenv.progress import calculate_config_hash

        project_dir = str(project_dir)
        progress_file = os.path.join(project_dir, "progress.jsonl")
        current_hash = calculate_config_hash(project_dir)

        if force_restart and not readonly and os.path.exists(progress_file):
            os.remove(progress_file)

        hash_match = True
        completed_gen: set[tuple[str, str, int]] = set()
        completed_eval: set[tuple[str, str, int]] = set()
        events: list[dict] = []
        verdict: str | None = None
        last_hash: str | None = None
        gen_events: dict[tuple[str, str, int], dict] = {}
        eval_events: dict[tuple[str, str, int], dict] = {}
        reasoning_events: dict[tuple[str, str, int], dict] = {}

        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if lines:
                try:
                    meta = json.loads(lines[0])
                    if meta.get("type") == "meta":
                        last_hash = meta.get("config_hash")
                        if last_hash != current_hash:
                            hash_match = False
                            if not readonly:
                                logger.log_warning(
                                    "Configuration hash mismatch! "
                                    "Renaming invalid progress file to: progress.jsonl.bak"
                                )
                                os.replace(progress_file, progress_file + ".bak")
                                return cls(hash_match=False, last_hash=last_hash)
                except json.JSONDecodeError:
                    if readonly:
                        return cls(hash_match=False)
                    logger.log_warning(
                        "Corrupted progress file detected! Renaming to: progress.jsonl.bak"
                    )
                    os.replace(progress_file, progress_file + ".bak")
                    return cls(hash_match=False)

                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        events.append(event)
                        if event["type"] == "gen":
                            key = (event["cand_id"], event["test_id"], event["rep"])
                            completed_gen.add(key)
                            gen_events[key] = event
                        elif event["type"] == "eval":
                            key = (event["cand_id"], event["test_id"], event["rep"])
                            completed_eval.add(key)
                            eval_events[key] = event
                        elif event["type"] == "reasoning":
                            key = (event["cand_id"], event["test_id"], event["rep"])
                            reasoning_events[key] = event
                        elif event["type"] == "verdict":
                            verdict = event["content"]
                    except json.JSONDecodeError:
                        pass  # ignore broken lines at the end of file (crash mid-write)

        if not readonly and not os.path.exists(progress_file):
            with open(progress_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "meta", "config_hash": current_hash}) + "\n")

        return cls(
            hash_match=hash_match,
            completed_gen=completed_gen,
            completed_eval=completed_eval,
            events=events,
            verdict=verdict,
            last_hash=last_hash,
            gen_events=gen_events,
            eval_events=eval_events,
            reasoning_events=reasoning_events,
        )
