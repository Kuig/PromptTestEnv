"""Global application configuration and benchmark project scaffolding.

Two distinct kinds of configuration live in this project:

- ``config.json`` at the repo root (this module's ``AppConfig``) holds the
  *measurement instrument*: the reasoning taxonomy, its judge prompts, the
  sentence-splitting parameters and the provider locality table. These must be
  identical across every benchmark, otherwise two reports are not comparable,
  so they are deliberately NOT per-project settings.
- ``Projects/<name>/judge_config.json`` (``JudgeConfig`` in ``models.py``) holds
  what the benchmark author chooses: which judge to call and with which
  parameters.

CONVENTIONS.md 4.2 wants ``config.json`` git-ignored with a ``.example``
template alongside it. This project deviates deliberately: the file carries no
credentials, only the definition of the measurement instrument, which must be
the same for anyone who clones the repo. It is therefore versioned, and a copy
ships inside the package as ``templates/default_config.json`` so that an
install from ``requirements_prod.txt`` still resolves it (see ``AppConfig.load``).
"""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

SECRETS_FILE = "secrets.json"

# Every credential UnifiedAiClient looks for (unified_ai_client.config._ENV_VAR_MAP).
# Scaffolded empty rather than with a placeholder string: UnifiedAiClient does not
# validate keys, it puts them straight in the request header, so a leftover
# placeholder would come back as a provider auth error instead of a missing key.
# Each one can also be supplied as an environment variable (GOOGLE_API_KEY,
# ANTHROPIC_API_KEY, ...), which takes priority over this file.
SECRETS_TEMPLATE: dict[str, str] = {
    "google_api_key": "",
    "anthropic_api_key": "",
    "openai_api_key": "",
    "mistral_api_key": "",
    "cohere_api_key": "",
    "meta_api_key": "",
    "groq_api_key": "",
    "xai_api_key": "",
}

CONFIG_FILE = "config.json"
DEFAULT_CONFIG_TEMPLATE = "default_config.json"


def _load_template(filename: str) -> str:
    """Read a default-content resource file bundled under prompttestenv/templates/.

    Args:
        filename: Resource file name (e.g. "default_judge_config.json").

    Returns:
        The file's raw text content.
    """
    return importlib.resources.files("prompttestenv").joinpath(f"templates/{filename}").read_text(encoding="utf-8")


def _from_dict(cls: type, data: dict):
    """Populate a settings dataclass from a dict, falling back to field defaults.

    Args:
        cls: The dataclass to construct.
        data: Raw dict, typically one section of config.json.

    Returns:
        A populated instance of cls, ignoring unknown keys.
    """
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ReasoningDimension:
    """One axis of the reasoning profile.

    Dimensions are scored independently and are NOT a partition: a sentence may
    score high on several at once, which is exactly the signal the previous
    mutually-exclusive taxonomy destroyed.
    """

    name: str = ""
    color: str = "#888888"
    definition: str = ""


@dataclass
class ReasoningSchema:
    """The reasoning taxonomy and the judge prompts that apply it."""

    system_prompt: str = ""
    intensity_scale: int = 3
    dimensions: list[ReasoningDimension] = field(default_factory=list)
    dimension_template: str = ""
    joint_template: str = ""
    metrics_template: str = ""

    @property
    def dimension_names(self) -> list[str]:
        """Dimension names in declaration order."""
        return [d.name for d in self.dimensions]

    @property
    def stamp(self) -> str:
        """Short identifier of this schema, stored on every reasoning event.

        Lets a report detect that it mixes analyses produced by different schema
        versions, without config.json having to enter the per-project config
        hash (which would invalidate every project's progress on any edit).

        Returns:
            A string such as ``"framing+solving+presentation@1a2b3c4d"``.
        """
        payload = json.dumps(
            {
                "system_prompt": self.system_prompt,
                "intensity_scale": self.intensity_scale,
                "dimensions": [
                    {"name": d.name, "definition": d.definition} for d in self.dimensions
                ],
                "dimension_template": self.dimension_template,
                "joint_template": self.joint_template,
                "metrics_template": self.metrics_template,
            },
            sort_keys=True,
        )
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
        return "+".join(self.dimension_names) + "@" + digest


@dataclass
class UnitSplittingConfig:
    """Parameters of the procedural sentence splitter."""

    min_unit_chars: int = 15
    abbreviations: list[str] = field(default_factory=list)


@dataclass
class ReasoningDefaults:
    """Default reasoning-analysis knobs, overridable per project in judge_config.json."""

    dimension_mode: str = "split"
    reliability_k: int = 1
    max_units_per_call: int = 150


@dataclass
class VerdictMetadata:
    """The self-describing header the verdict payload opens with.

    This is not benchmark prose: every section describes something the CODE
    emits, such as that the aggregate tables carry no standard deviation or that
    an `assert` judge may use the full 1-10 range or only a pass/fail pair. Two
    projects wording it differently would make their reports incomparable, which
    is why it belongs here and not in judge_config.json. What a benchmark means
    by its scores stays the author's business, in verdict_template.

    Sections are emitted only when the payload actually contains what they
    describe, so a project with no reasoning analysis is not told how to read a
    reasoning profile it will never see.

    Every field defaults to the empty string, matching ReasoningSchema. An empty
    section that the payload needs is a refusal, never a silent omission: the
    figures would still reach the judge, just without the caveats that keep it
    from over-claiming.
    """

    header: str = ""
    figures: str = ""
    score_scales: str = ""
    cost_per_point: str = ""
    reasoning_profile: str = ""
    judge_types_intro: str = ""
    judge_llm: str = ""
    judge_similarity: str = ""
    judge_assert: str = ""
    judge_types_mixed: str = ""
    global_criteria_intro: str = ""
    global_criteria_llm: str = ""
    global_criteria_similarity: str = ""
    global_criteria_assert: str = ""
    global_criteria_none: str = ""


@dataclass
class AppConfig:
    """Global, cross-project application configuration.

    Every field has a default so the tool starts even when no config.json can be
    found anywhere (CONVENTIONS 4.3).
    """

    reasoning_schema: ReasoningSchema = field(default_factory=ReasoningSchema)
    unit_splitting: UnitSplittingConfig = field(default_factory=UnitSplittingConfig)
    reasoning_defaults: ReasoningDefaults = field(default_factory=ReasoningDefaults)
    local_providers: list[str] = field(default_factory=list)
    verdict_metadata: VerdictMetadata = field(default_factory=VerdictMetadata)

    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        """Build an AppConfig from a raw config.json dict.

        Args:
            data: Parsed contents of config.json.

        Returns:
            Populated AppConfig, falling back to field defaults for missing keys.
        """
        schema_data = dict(data.get("reasoning_schema", {}))
        raw_dimensions = schema_data.pop("dimensions", [])
        schema = _from_dict(ReasoningSchema, schema_data)
        schema.dimensions = [_from_dict(ReasoningDimension, d) for d in raw_dimensions]
        return cls(
            reasoning_schema=schema,
            unit_splitting=_from_dict(UnitSplittingConfig, data.get("unit_splitting", {})),
            reasoning_defaults=_from_dict(ReasoningDefaults, data.get("reasoning_defaults", {})),
            local_providers=data.get("local_providers", []),
            verdict_metadata=_from_dict(
                VerdictMetadata, data.get("verdict_metadata", {})
            ),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load the global configuration.

        Resolution order follows CONVENTIONS 4.4: the user's working directory
        first, then the repo root, then the read-only copy shipped inside the
        package. The last step is what an install from requirements_prod.txt
        resolves, since there the user's working directory holds no config.json.

        Args:
            path: Explicit config file path. When given, no other location is
                tried before the packaged default.

        Returns:
            Populated AppConfig. Falls back to field defaults, and never raises,
            if no readable config file exists anywhere.
        """
        import prompttestenv.logger as logger

        if path is not None:
            candidates = [Path(path)]
        else:
            candidates = [Path.cwd() / CONFIG_FILE, _PROJECT_ROOT / CONFIG_FILE]

        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        return cls.from_dict(json.load(f))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.log_warning(f"Error reading {candidate}: {exc}. Trying next location.")

        try:
            return cls.from_dict(json.loads(_load_template(DEFAULT_CONFIG_TEMPLATE)))
        except Exception as exc:
            logger.log_error(f"Could not load any {CONFIG_FILE}: {exc}. Using built-in defaults.")
            return cls()


_APP_CONFIG: AppConfig | None = None


def get_app_config(reload: bool = False) -> AppConfig:
    """Return the process-wide AppConfig, loading it on first use.

    Args:
        reload: If True, discard the cached instance and read the file again.

    Returns:
        The cached AppConfig instance.
    """
    global _APP_CONFIG
    if _APP_CONFIG is None or reload:
        _APP_CONFIG = AppConfig.load()
    return _APP_CONFIG


def init_project(project_dir: str, custom_candidates: list[dict] | None = None) -> None:
    """Initialize the project directory structure for a new benchmark.

    Creates the project folder, subdirectories, and template configuration
    files (candidates.json, judge_config.json, test_cases.json,
    global_criteria.json). Skips files that already exist.

    Args:
        project_dir: Path to the benchmark project directory to create.
        custom_candidates: Optional list of candidate dicts. If provided,
            overrides the default candidates.json content. Candidates may
            include a 'system_prompt_text' key which is auto-saved to a file.
    """
    import prompttestenv.logger as logger

    os.makedirs(project_dir, exist_ok=True)
    sys_dir = os.path.join(project_dir, "system_prompts")
    os.makedirs(sys_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "test_files"), exist_ok=True)

    # 1. secrets.json, created only if missing. It goes in the CURRENT WORKING
    # DIRECTORY because that is where UnifiedAiClient looks for it when it
    # instantiates a provider, and it is the only copy that will ever be read.
    # Writing it next to the package instead would put it inside site-packages
    # on a non-editable install, where nothing would ever find it.
    secrets_path = Path(os.getcwd()) / SECRETS_FILE
    if not secrets_path.exists():
        secrets_path.write_text(json.dumps(SECRETS_TEMPLATE, indent=4), encoding="utf-8")
        logger.log_warning(
            f"Created {secrets_path}. Fill in the key for the providers you use, "
            "or set the matching environment variable (GOOGLE_API_KEY, "
            "ANTHROPIC_API_KEY, ...), which takes priority. Local providers such "
            "as Ollama need no key at all."
        )

    # 2. Judge Configuration
    judge_file = os.path.join(project_dir, "judge_config.json")
    if not os.path.exists(judge_file):
        with open(judge_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_judge_config.json"))

    # 3. Candidates Configuration
    cand_file = os.path.join(project_dir, "candidates.json")
    if custom_candidates is not None:
        processed_candidates = []
        for i, cand in enumerate(custom_candidates):
            if "system_prompt_text" in cand:
                filename = f"custom_prompt_{i}.txt"
                filepath = os.path.join(sys_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cand["system_prompt_text"])
                new_cand = {k: v for k, v in cand.items() if k != "system_prompt_text"}
                new_cand["system_prompt_file"] = filename
                processed_candidates.append(new_cand)
            else:
                processed_candidates.append(cand)
        with open(cand_file, "w", encoding="utf-8") as f:
            json.dump(processed_candidates, f, indent=4)
    elif not os.path.exists(cand_file):
        default_prompt_filename = "pirate_prompt.txt"
        default_prompt_filepath = os.path.join(sys_dir, default_prompt_filename)
        with open(default_prompt_filepath, "w", encoding="utf-8") as f:
            f.write(_load_template("default_pirate_prompt.txt"))
        with open(cand_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_candidates.json"))

    # 4. Test Cases
    test_file = os.path.join(project_dir, "test_cases.json")
    if not os.path.exists(test_file):
        sample_txt_path = os.path.join(project_dir, "test_files", "sample.txt")
        if not os.path.exists(sample_txt_path):
            with open(sample_txt_path, "w", encoding="utf-8") as f:
                f.write(_load_template("default_sample.txt"))
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_test_cases.json"))

    # 5. Global Criteria (JSON)
    global_file = os.path.join(project_dir, "global_criteria.json")
    if not os.path.exists(global_file):
        with open(global_file, "w", encoding="utf-8") as f:
            f.write(_load_template("default_global_criteria.json"))

