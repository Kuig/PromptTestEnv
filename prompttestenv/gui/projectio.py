"""Byte-faithful load/serialise of a project's four config files, for the editor.

Why this module exists at all, rather than the editor just using the dataclasses:

``progress.calculate_config_hash`` gates run resume, and it hashes
``candidates.json``, ``test_cases.json`` and ``global_criteria.json`` as RAW
BYTES (only ``judge_config.json`` is canonicalised). So re-indenting a file,
reordering its keys, turning an on-disk ``2`` into ``2.0``, or adding a trailing
newline is enough to make a finished run refuse to resume, with nothing about
the benchmark actually changed.

Meanwhile both loaders are lossy in the write direction: they silently drop keys
they do not know, and fill in every key the author omitted. Round-tripping a
project through ``Candidate.load_all`` would therefore delete a project's extra
keys, materialise every default the templates deliberately leave out, and
persist the derived ``resolved_system_instruction``.

So the editor edits RAW DICTS, and this module holds the rules that keep
"open a project, press Save, change nothing" a genuine no-op:

1. key order preserved (iterate the original, append new keys after)
2. numeric form preserved (an on-disk int stays an int)
3. trailing newline AND line-ending style preserved (the shipped templates
   disagree on the former, and a checkout with git's ``core.autocrlf`` genuinely
   holds CRLF, so forcing LF would rewrite every line of every file)
4. a key is emitted only if it was already present, is in _ALWAYS_EMIT, or
   differs from its effective default
5. identical bytes are not written at all

The one deliberate exception is a test case's ``file``: its path separators are
normalised to ``/`` on save (see ``attachment_value``), so a project authored on
Windows runs on POSIX too. That does change bytes, and therefore the run hash,
on a project holding backslashes — which the editor already surfaces as an
invalidation warning before the user saves.

The dataclasses stay the source of truth for defaults — but only where they
actually hold one. See _LOADER_DEFAULTS.

No Streamlit import: this module is unit-tested on its own.
"""
from __future__ import annotations

import ast
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from prompttestenv.models import (
    Candidate,
    GlobalCriteria,
    JudgeConfig,
    ReasoningJudgeSettings,
    SimilarityJudgeSettings,
    TestCase,
    TestJudgeSettings,
    VerdictJudgeSettings,
    attachment_paths,
)
from prompttestenv.progress import HASHED_FILENAMES

CANDIDATES_FILE = "candidates.json"
TEST_CASES_FILE = "test_cases.json"
JUDGE_CONFIG_FILE = "judge_config.json"
GLOBAL_CRITERIA_FILE = "global_criteria.json"

SYSTEM_PROMPTS_DIR = "system_prompts"
TEST_FILES_DIR = "test_files"

# The judge_config.json blocks, in the order the shipped template writes them,
# mapped to the settings dataclass that defines each one's keys.
JUDGE_BLOCKS: dict[str, type] = {
    "test_judge": TestJudgeSettings,
    "similarity_judge": SimilarityJudgeSettings,
    "reasoning_judge": ReasoningJudgeSettings,
    "verdict_judge": VerdictJudgeSettings,
}

# Defaults that live in the LOADERS rather than in a dataclass field.
# Candidate.name/provider/model and TestCase.id/prompt/criteria are declared
# without a default (dataclasses.MISSING); the fallbacks below are what
# Candidate.load_all / TestCase.load_all actually apply. Reading fields() alone
# would make the editor write a candidate with no `provider` key and a null
# `model`. Kept honest by test_gui_projectio's classification guard.
_LOADER_DEFAULTS: dict[type, dict[str, object]] = {
    Candidate: {"name": None, "provider": "google", "model": None},
    TestCase: {"id": "N/A", "prompt": "", "criteria": ""},
}

# Keys always written, default-valued or not: omitting them yields a None or
# "N/A" at load time that breaks the run rather than falling back sensibly.
_ALWAYS_EMIT: dict[type, tuple[str, ...]] = {
    Candidate: ("name", "provider", "model"),
    TestCase: ("id", "prompt", "criteria"),
}

# Derived fields that are dataclass fields but NOT valid on-disk keys.
_NEVER_EMIT: dict[type, tuple[str, ...]] = {
    Candidate: ("resolved_system_instruction",),
}

# The only `thinking` values the editor offers. api.py normalises exactly these
# three and passes anything else through verbatim, but an out-of-vocabulary
# value is shown and preserved rather than offered as a choice.
THINKING_CHOICES = ("default", "true", "false")

# Provider names UnifiedAiClient's dispatch accepts (unified_ai_client.client
# resolves them in an if/elif chain and exports no registry to import). The
# editor treats this as a suggestion list, not a validation rule.
PROVIDERS = (
    "google", "anthropic", "openai", "mistral", "cohere", "meta", "groq", "xai",
    "ollama", "lmstudio", "llamacpp", "script",
)


def editable_fields(cls: type) -> tuple[str, ...]:
    """Field names of `cls` the editor may render and write."""
    never = _NEVER_EMIT.get(cls, ())
    return tuple(f.name for f in dataclasses.fields(cls) if f.name not in never)


def effective_default(cls: type, name: str):
    """The value `name` takes on disk when the key is absent.

    Prefers the dataclass field default; falls back to _LOADER_DEFAULTS for the
    six required fields that declare none.

    Args:
        cls: One of the config dataclasses.
        name: Field name.

    Returns:
        The effective default value.

    Raises:
        KeyError: If the field has neither a dataclass default nor a loader one,
            which means the tables above went stale.
    """
    for f in dataclasses.fields(cls):
        if f.name != name:
            continue
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return f.default_factory()  # type: ignore[misc]
        break
    loader = _LOADER_DEFAULTS.get(cls, {})
    if name in loader:
        return loader[name]
    raise KeyError(f"No known default for {cls.__name__}.{name}")


def preserve_form(original, new):
    """Return `new`, but keep `original`'s literal when they only differ in type.

    Guards the no-op-save case: the templates ship ``"repetitions": 2`` as an
    int, while a numeric input hands back ``2.0``. Same value, different bytes,
    different run hash.

    Booleans are excluded on both sides because ``True == 1`` in Python.
    """
    if (
        type(original) is not type(new)
        and isinstance(original, (int, float))
        and isinstance(new, (int, float))
        and not isinstance(original, bool)
        and not isinstance(new, bool)
        and original == new
    ):
        return original
    return new


def preserve_thinking(original, new):
    """Keep a boolean `thinking` boolean when the widget hands back its spelling.

    `thinking` is legally either a bool or one of the strings api.py normalises
    ("true"/"false"/"default"), and the templates use both spellings. A form
    renders it as text, so without this a file holding ``true`` would be
    rewritten as ``"true"`` — same behaviour, different bytes, different run
    hash, on an edit the user never made.
    """
    if isinstance(original, bool) and isinstance(new, str):
        if new.strip().lower() == str(original).lower():
            return original
    return new


def attachment_value(paths: list[str]) -> str | list[str] | None:
    """The on-disk shape for a test case's chosen attachments.

    None for zero, a plain string for one, a list for two or more. Writing a
    single-element list where the file held a plain string would rewrite every
    existing single-attachment project on a save that changed nothing.

    Args:
        paths: The attachment paths currently selected, in order.

    Returns:
        The value to store under the ``file`` key, or None to drop the key.
    """
    normalised = [path.replace("\\", "/") for path in paths if path]
    if not normalised:
        return None
    if len(normalised) == 1:
        return normalised[0]
    return normalised


def merge_preserving_shape(original: dict, new_values: dict, cls: type) -> dict:
    """Merge edited values into a raw dict without disturbing its on-disk shape.

    Key order follows `original`, so a first save does not reshuffle a
    byte-hashed file. Keys `original` did not have are appended in field order,
    and only when they carry information (see rule 4 in the module docstring).

    Args:
        original: The dict as read from disk.
        new_values: Edited values, keyed by field name. Keys absent here keep
            whatever `original` held.
        cls: The dataclass describing this dict's known fields.

    Returns:
        A new dict ready to serialise.
    """
    known = set(editable_fields(cls))
    never = set(_NEVER_EMIT.get(cls, ()))
    always = set(_ALWAYS_EMIT.get(cls, ()))

    out: dict = {}
    for key, old_value in original.items():
        if key in never:
            continue
        if key in known and key in new_values:
            new_value = new_values[key]
            if key == "thinking":
                new_value = preserve_thinking(old_value, new_value)
            out[key] = preserve_form(old_value, new_value)
        else:
            out[key] = old_value

    for name in editable_fields(cls):
        if name in out or name not in new_values:
            continue
        value = new_values[name]
        if name in always or value != effective_default(cls, name):
            out[name] = value
    return out


@dataclass
class ProjectDraft:
    """A project's four config files, as raw data plus the bytes they came from.

    `disk` is what makes "did this actually change?" answerable exactly: the
    editor serialises the current draft and compares bytes, rather than trying
    to track per-field dirtiness.
    """

    project_dir: str
    candidates: list[dict] = field(default_factory=list)
    tests: list[dict] = field(default_factory=list)
    judge: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)
    disk: dict[str, bytes | None] = field(default_factory=dict)
    trailing_newline: dict[str, bool] = field(default_factory=dict)
    newline: dict[str, str] = field(default_factory=dict)

    @property
    def system_prompts_dir(self) -> Path:
        return Path(self.project_dir) / SYSTEM_PROMPTS_DIR

    @property
    def test_files_dir(self) -> Path:
        return Path(self.project_dir) / TEST_FILES_DIR

    def system_prompt_names(self) -> list[str]:
        """Bare .txt filenames available to a candidate's system_prompt_file."""
        directory = self.system_prompts_dir
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix == ".txt")

    def test_file_names(self) -> list[str]:
        """Bare filenames available as a test case attachment."""
        directory = self.test_files_dir
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_file())


def _read_raw(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def detect_newline(raw: bytes | None) -> str:
    """The line terminator a file already uses, defaulting to LF.

    A checkout made with git's ``core.autocrlf`` holds CRLF, and these files are
    hashed byte-for-byte, so rewriting them as LF would invalidate a finished
    run on the very first save.
    """
    return "\r\n" if raw is not None and b"\r\n" in raw else "\n"


def _parse(raw: bytes | None, fallback):
    if raw is None:
        return fallback
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback


def load_project(project_dir: str) -> ProjectDraft:
    """Read a project's four config files as raw data.

    A missing or unparseable file loads as empty rather than raising: the editor
    is the tool you reach for to *fix* such a project.

    Args:
        project_dir: Path to the benchmark project directory.

    Returns:
        A ProjectDraft carrying both the parsed data and the original bytes.
    """
    base = Path(project_dir)
    raws = {name: _read_raw(base / name) for name in HASHED_FILENAMES}

    candidates = _parse(raws[CANDIDATES_FILE], [])
    tests = _parse(raws[TEST_CASES_FILE], [])
    judge = _parse(raws[JUDGE_CONFIG_FILE], {})
    criteria = _parse(raws[GLOBAL_CRITERIA_FILE], {})

    return ProjectDraft(
        project_dir=str(project_dir),
        candidates=candidates if isinstance(candidates, list) else [],
        tests=tests if isinstance(tests, list) else [],
        judge=judge if isinstance(judge, dict) else {},
        criteria=criteria if isinstance(criteria, dict) else {},
        disk=raws,
        trailing_newline={
            name: (raw.endswith(b"\n") if raw is not None else True)
            for name, raw in raws.items()
        },
        newline={name: detect_newline(raw) for name, raw in raws.items()},
    )


def serialize(data, *, trailing_newline: bool, newline: str = "\n") -> bytes:
    """Serialise one config file exactly as the writers in models.py would.

    Keeping this in step with models._write_json_atomic is what lets the editor
    compare "what I would write" against "what is on disk" byte-for-byte.
    """
    text = json.dumps(data, indent=4, ensure_ascii=False)
    if trailing_newline:
        text += "\n"
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")


def serialize_all(draft: ProjectDraft) -> dict[str, bytes]:
    """Serialise every config file of a draft.

    Returns:
        Maps each of HASHED_FILENAMES to the bytes that would be written.
    """
    payloads = {
        CANDIDATES_FILE: draft.candidates,
        TEST_CASES_FILE: draft.tests,
        JUDGE_CONFIG_FILE: draft.judge,
        GLOBAL_CRITERIA_FILE: draft.criteria,
    }
    return {
        name: serialize(
            data,
            trailing_newline=draft.trailing_newline.get(name, True),
            newline=draft.newline.get(name, "\n"),
        )
        for name, data in payloads.items()
    }


def changed_files(draft: ProjectDraft) -> dict[str, bytes]:
    """The files whose serialised bytes differ from what is on disk.

    Rule 5: anything identical is not written at all, so a no-op save touches no
    mtime and cannot invalidate a run.
    """
    return {
        name: payload
        for name, payload in serialize_all(draft).items()
        if payload != draft.disk.get(name)
    }


_SAVERS = {
    CANDIDATES_FILE: Candidate.save_all,
    TEST_CASES_FILE: TestCase.save_all,
    JUDGE_CONFIG_FILE: JudgeConfig.save,
    GLOBAL_CRITERIA_FILE: GlobalCriteria.save,
}


def save_project(draft: ProjectDraft) -> list[str]:
    """Write only the config files that actually changed.

    Args:
        draft: The draft to persist.

    Returns:
        The filenames written, in HASHED_FILENAMES order.

    Raises:
        OSError: If a file cannot be written (see models._write_json_atomic).
    """
    payloads = {
        CANDIDATES_FILE: draft.candidates,
        TEST_CASES_FILE: draft.tests,
        JUDGE_CONFIG_FILE: draft.judge,
        GLOBAL_CRITERIA_FILE: draft.criteria,
    }
    pending = changed_files(draft)
    written = []
    for name in HASHED_FILENAMES:
        if name not in pending:
            continue
        _SAVERS[name](
            draft.project_dir,
            payloads[name],
            trailing_newline=draft.trailing_newline.get(name, True),
            newline=draft.newline.get(name, "\n"),
        )
        draft.disk[name] = pending[name]
        written.append(name)
    return written


# ── Validation ────────────────────────────────────────────────────────────────
# Errors block the save; warnings do not. The split follows what the runtime
# itself does: what degrades to a logged warning at run time is a warning here.

EVALUATION_PLACEHOLDERS = ("criteria", "user_prompt", "candidate_response")


def effective_lambda(criteria: str) -> str:
    """The expression test_judge.py will actually evaluate for an assert.

    Mirrors test_judge.py's normalisation byte for byte, `startswith("lambda")`
    included — not `"lambda "`, so a body spelled `lambdas: ...` is NOT prefixed
    there and must not be prefixed here either.
    """
    expr = criteria.strip()
    if not expr.startswith("lambda"):
        expr = f"lambda {expr}"
    return expr


def check_assert_criteria(criteria: str) -> str | None:
    """Parse-check an assert lambda. Returns an error message, or None if valid.

    Parses only. The lambda is NEVER executed here: at run time test_judge.py
    eval()s it unsandboxed by documented design, but opening a form is not a
    reason to run a project's code.
    """
    try:
        ast.parse(effective_lambda(criteria), mode="eval")
    except SyntaxError as exc:
        return f"cannot be parsed as a lambda: {exc.msg}"
    return None


def check_evaluation_template(template: str) -> tuple[str | None, str | None]:
    """Check test_judge.evaluation_template, the one template still .format()ed.

    Returns:
        ``(error, warning)``, either of which may be None.

    A stray brace raises ValueError inside test_judge.py, which only catches
    KeyError — so it escapes and drives EVERY evaluation to the -1 sentinel.
    A missing placeholder formats fine but hides that input from the judge.
    """
    try:
        template.format(**{name: "" for name in EVALUATION_PLACEHOLDERS})
    except (KeyError, IndexError, ValueError) as exc:
        return (f"is not a valid format template: {exc}", None)

    missing = [f"{{{n}}}" for n in EVALUATION_PLACEHOLDERS if f"{{{n}}}" not in template]
    if missing:
        return (None, f"does not use {', '.join(missing)} — the judge will not see it")
    return (None, None)


def check_filename(name: str) -> str | None:
    """Reject anything that is not a plain filename. Returns an error, or None."""
    if not name or name in (".", ".."):
        return "Filename is empty or invalid."
    if Path(name).name != name:
        return "Filename must not contain a path — just the file name."
    return None


def _duplicates(values: list[str]) -> list[str]:
    seen, dupes = set(), []
    for value in values:
        if value in seen and value not in dupes:
            dupes.append(value)
        seen.add(value)
    return dupes


def validate(draft: ProjectDraft) -> tuple[list[str], list[str]]:
    """Validate a whole draft.

    Args:
        draft: The draft to check.

    Returns:
        ``(errors, warnings)``. A non-empty `errors` must block saving.
    """
    errors: list[str] = []
    warnings: list[str] = []

    prompts = set(draft.system_prompt_names())
    attachments = {f"{TEST_FILES_DIR}/{n}" for n in draft.test_file_names()}

    # Candidates: name keys every progress.jsonl event and pools the results,
    # so a duplicate silently merges two candidates into one.
    names = [str(c.get("name") or "") for c in draft.candidates]
    for dupe in _duplicates([n for n in names if n]):
        errors.append(f"Candidate name '{dupe}' is used more than once.")
    for index, cand in enumerate(draft.candidates, start=1):
        label = cand.get("name") or f"#{index}"
        if not cand.get("name"):
            errors.append(f"Candidate #{index} has no name.")
        if not cand.get("model"):
            errors.append(f"Candidate '{label}' has no model.")
        prompt_file = cand.get("system_prompt_file")
        if prompt_file and prompt_file not in prompts:
            warnings.append(
                f"Candidate '{label}' refers to system prompt '{prompt_file}', which is missing."
            )
        thinking = cand.get("thinking")
        if isinstance(thinking, str) and thinking.lower() not in THINKING_CHOICES:
            warnings.append(
                f"Candidate '{label}' has a non-standard thinking value '{thinking}'."
            )

    # Test cases: id keys the same events.
    ids = [str(t.get("id") or "") for t in draft.tests]
    for dupe in _duplicates([i for i in ids if i]):
        errors.append(f"Test case id '{dupe}' is used more than once.")
    for index, test in enumerate(draft.tests, start=1):
        label = test.get("id") or f"#{index}"
        if not test.get("id"):
            errors.append(f"Test case #{index} has no id.")
        if test.get("judge_type") == "assert":
            problem = check_assert_criteria(str(test.get("criteria", "")))
            if problem:
                errors.append(f"Test case '{label}' criteria {problem}")
        try:
            declared = attachment_paths(test.get("file"))
        except ValueError as exc:
            # A shape the run itself would reject, so it must block the save
            # rather than be reported as a missing file.
            errors.append(f"Test case '{label}': {exc}")
            declared = []
        for path in declared:
            if path not in attachments:
                warnings.append(
                    f"Test case '{label}' refers to attachment '{path}', which is missing."
                )

    # Judge config.
    test_judge = draft.judge.get("test_judge") or {}
    template = test_judge.get("evaluation_template")
    if isinstance(template, str) and template:
        error, warning = check_evaluation_template(template)
        if error:
            errors.append(f"test_judge.evaluation_template {error}")
        if warning:
            warnings.append(f"test_judge.evaluation_template {warning}")

    verdict_judge = draft.judge.get("verdict_judge") or {}
    if not (verdict_judge.get("verdict_template") or "").strip():
        warnings.append("verdict_judge.verdict_template is empty — no verdict will be written.")
    if draft.judge.get("group_verdicts") and not (
        verdict_judge.get("global_verdict_template") or ""
    ).strip():
        warnings.append(
            "group_verdicts is on but global_verdict_template is empty — "
            "the verdict will abort."
        )

    # Global criteria.
    if draft.criteria.get("mode") == "assert":
        problem = check_assert_criteria(str(draft.criteria.get("assert_criteria", "")))
        if problem:
            errors.append(f"global_criteria.assert_criteria {problem}")

    # Orphans, in both directions.
    used_prompts = {c.get("system_prompt_file") for c in draft.candidates}
    for name in sorted(prompts - {p for p in used_prompts if p}):
        warnings.append(f"System prompt '{name}' is not used by any candidate.")
    used_files = set()
    for test in draft.tests:
        try:
            used_files.update(attachment_paths(test.get("file")))
        except ValueError:
            continue  # already reported as an error above
    for path in sorted(attachments - used_files):
        warnings.append(f"Attachment '{path}' is not used by any test case.")

    return errors, warnings


def externally_modified(draft: ProjectDraft) -> list[str]:
    """Config files whose bytes on disk no longer match what was loaded.

    Another editor tab, a git checkout or a hand edit — saving over it would
    silently revert someone's work.
    """
    base = Path(draft.project_dir)
    return [
        name for name in HASHED_FILENAMES
        if _read_raw(base / name) != draft.disk.get(name)
    ]


# Template file -> destination, relative to the project directory. Mirrors what
# init_project() scaffolds, minus its secrets.json step (see seed_project).
_SEED_FILES = (
    ("default_candidates.json", CANDIDATES_FILE),
    ("default_judge_config.json", JUDGE_CONFIG_FILE),
    ("default_test_cases.json", TEST_CASES_FILE),
    ("default_global_criteria.json", GLOBAL_CRITERIA_FILE),
    ("default_pirate_prompt.txt", f"{SYSTEM_PROMPTS_DIR}/pirate_prompt.txt"),
    ("default_sample.txt", f"{TEST_FILES_DIR}/sample.txt"),
)


def seed_project(project_dir: str) -> list[str]:
    """Scaffold a new project from the packaged templates.

    Deliberately does NOT call config.init_project(): that also writes a
    secrets.json into os.getcwd() (config.py:314) — which, for a GUI, is
    wherever the user happened to launch the command from, not the project being
    created. The editor must not scatter files outside the project directory.
    Everything else init_project does is reproduced here.

    Existing files are never overwritten, so this is safe to run on a partially
    populated directory.

    Args:
        project_dir: Path to create the project in.

    Returns:
        The project-relative paths actually created.
    """
    from prompttestenv.config import _load_template

    base = Path(project_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / SYSTEM_PROMPTS_DIR).mkdir(exist_ok=True)
    (base / TEST_FILES_DIR).mkdir(exist_ok=True)

    created = []
    for template_name, relative in _SEED_FILES:
        destination = base / relative
        if destination.exists():
            continue
        destination.write_text(_load_template(template_name), encoding="utf-8", newline="\n")
        created.append(relative)
    return created
