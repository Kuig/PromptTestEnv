"""Byte-faithful load/serialise of a project's config files.

The four hashed configs, plus the optional ``group_clusters.json``, which
follows the same fidelity rules but sits outside the run hash and is never
created by a save that leaves it empty (see EDITABLE_FILENAMES).

Every interface that edits a project goes through here: the Streamlit editor,
`projectedit`'s headless API, and therefore the CLI and the MCP server too. It
lives at package level rather than under `gui/` for exactly that reason.

Why this module exists at all, rather than callers just using the dataclasses:

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

So every editor of a project edits RAW DICTS, and this module holds the rules
that keep "open a project, save it, change nothing" a genuine no-op:

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
on a project holding backslashes — which both editing paths surface as an
invalidation warning before anything is written.

The dataclasses stay the source of truth for defaults — but only where they
actually hold one. See _LOADER_DEFAULTS.

No Streamlit import: this module is unit-tested on its own, and the core
must never depend on the `gui` package.
"""
from __future__ import annotations

import ast
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from prompttestenv.models import (
    CANDIDATE_ROLE_CANDIDATE,
    DEFAULT_GROUP,
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
from prompttestenv.frontier import (
    GROUP_CLUSTERS_FILE,
    GroupClusters,
    check_group_clusters,
    reference_warnings,
)
from prompttestenv.progress import (
    HASHED_FILENAMES,
    config_hash_from_bytes,
    read_stored_hash,
)

CANDIDATES_FILE = "candidates.json"
TEST_CASES_FILE = "test_cases.json"
JUDGE_CONFIG_FILE = "judge_config.json"
GLOBAL_CRITERIA_FILE = "global_criteria.json"

# Every JSON file this module reads and writes: the four hashed configs plus
# group_clusters.json, which is optional and deliberately outside the run hash
# (it changes how results are read, never what was measured). The fidelity
# rules apply to it all the same; the hash only ever sees HASHED_FILENAMES.
EDITABLE_FILENAMES = HASHED_FILENAMES + (GROUP_CLUSTERS_FILE,)
# Files that may be absent. An absent one whose draft is still empty is never
# created, so opening and saving a project cannot add a file to it.
OPTIONAL_FILENAMES = (GROUP_CLUSTERS_FILE,)

SYSTEM_PROMPTS_DIR = "system_prompts"
TEST_FILES_DIR = "test_files"

# The two directories a project holds beside its four JSON configs. Neither is
# part of the run hash, which is the single most surprising thing about them and
# why every interface that touches one has to say so: see ASSETS_NOT_HASHED.
ASSET_KINDS = (SYSTEM_PROMPTS_DIR, TEST_FILES_DIR)

ASSETS_NOT_HASHED = (
    "system_prompts/ and test_files/ are not part of the run hash. Only the "
    "four JSON configs are. Changing one of these does not invalidate "
    "progress.jsonl, so a resumed run mixes responses produced under the old "
    "and the new version into one report, with nothing flagging it. Use "
    "--force-restart if that matters."
)

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
# `model`. Kept honest by test_projectio's classification guard.
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

# What a brand-new entry holds before anything is merged into it. Its keys are
# exactly _ALWAYS_EMIT's, but the VALUES are not effective_default's: the loader
# fallback for `name` and `model` is None, which is the right answer for "the key
# is absent" and the wrong one for "the author has not typed it yet". Kept in
# step with _ALWAYS_EMIT by test_projectio's classification guard.
_NEW_ROW: dict[type, dict] = {
    Candidate: {"name": "", "provider": "google", "model": ""},
    TestCase: {"id": "", "prompt": "", "criteria": ""},
}

# The only `thinking` values the editor offers. api.py normalises exactly these
# three and passes anything else through verbatim, but an out-of-vocabulary
# value is shown and preserved rather than offered as a choice.
THINKING_CHOICES = ("default", "true", "false")

# Provider names UnifiedAiClient's registry accepts. The registry itself is
# private and list_providers() reports only the providers usable right now, so
# the full set is spelled out here. The editor treats this as a suggestion
# list, not a validation rule.
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
        default = effective_default(cls, name)
        if name == "thinking":
            # Same spelling rule as for a key already present, applied against
            # the default: ReasoningJudgeSettings.thinking defaults to the bool
            # False while the widget hands back "false", and without this an
            # untouched form added a reasoning_judge block to a hashed file on
            # its second run.
            value = preserve_thinking(default, value)
        if name in always or value != default:
            out[name] = value
    return out


def blank_row(cls: type) -> dict:
    """A fresh `candidates.json` / `test_cases.json` entry, ready to merge into.

    Args:
        cls: Candidate or TestCase.

    Returns:
        A new dict holding exactly the keys _ALWAYS_EMIT requires.
    """
    return dict(_NEW_ROW[cls])


def upsert(items: list[dict], key_field: str, values: dict, cls: type) -> list[dict]:
    """Merge `values` into the entry `key_field` names, or append a new one.

    The identity is the on-disk key itself (`name` for a candidate, `id` for a
    test case), because that is what every progress.jsonl event is keyed by too.

    Args:
        items: The current list, left untouched.
        key_field: The field whose value identifies an entry.
        values: Edited values, keyed by field name.
        cls: The dataclass describing the entry.

    Returns:
        A new list. An existing entry keeps its key order and its unknown keys
        (see merge_preserving_shape); a new one starts from blank_row(), so the
        _ALWAYS_EMIT keys are present even when `values` omits them.
    """
    key = values.get(key_field)
    out = list(items)
    for index, item in enumerate(out):
        if item.get(key_field) == key:
            out[index] = merge_preserving_shape(item, values, cls)
            return out
    out.append(merge_preserving_shape(blank_row(cls), values, cls))
    return out


def merge_judge(judge: dict, values: dict) -> dict:
    """Deep-merge a partial judge_config: top-level scalars plus the four blocks.

    A block named in `values` is merged into whatever the file already had, or
    created from scratch. That differs from the editor's own rule, which never
    creates a block the file lacked — there, an empty block would appear merely
    because someone opened a tab; here, naming the block IS the edit.

    Args:
        judge: The judge_config.json dict as read from disk.
        values: A partial dict, possibly holding nested block dicts.

    Returns:
        A new dict ready to serialise.

    Raises:
        ValueError: If a value under a known block name is not a mapping.
    """
    scalars = {k: v for k, v in values.items() if k not in JUDGE_BLOCKS}
    out = merge_preserving_shape(judge, scalars, JudgeConfig)
    for name, settings_cls in JUDGE_BLOCKS.items():
        if name not in values:
            continue
        block_values = values[name]
        if not isinstance(block_values, dict):
            raise ValueError(f"judge_config.{name} must be an object")
        existing = out.get(name)
        base = existing if isinstance(existing, dict) else {}
        out[name] = merge_preserving_shape(base, block_values, settings_cls)
    return out


@dataclass
class ProjectDraft:
    """A project's config files, as raw data plus the bytes they came from.

    The four hashed files, plus the optional group_clusters.json in `clusters`
    (an empty dict both when the file is absent and when it holds `{}`; `disk`
    tells the two apart).

    `disk` is what makes "did this actually change?" answerable exactly: the
    draft is serialised and compared byte for byte, rather than tracking
    per-field dirtiness.

    `pending_assets` holds system prompts and attachments that a caller means to
    write but has not written yet (kind -> filename -> content, None meaning a
    pending delete). It exists because validate() answers "does this attachment
    exist?" by listing the directory: without it, one patch that creates a file
    AND the test case using it would report the file as missing. The Streamlit
    editor writes assets immediately, so for it this stays empty.
    """

    project_dir: str
    candidates: list[dict] = field(default_factory=list)
    tests: list[dict] = field(default_factory=list)
    judge: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)
    disk: dict[str, bytes | None] = field(default_factory=dict)
    trailing_newline: dict[str, bool] = field(default_factory=dict)
    newline: dict[str, str] = field(default_factory=dict)
    pending_assets: dict[str, dict[str, str | bytes | None]] = field(
        default_factory=lambda: {kind: {} for kind in ASSET_KINDS}
    )

    @property
    def system_prompts_dir(self) -> Path:
        return Path(self.project_dir) / SYSTEM_PROMPTS_DIR

    @property
    def test_files_dir(self) -> Path:
        return Path(self.project_dir) / TEST_FILES_DIR

    def asset_dir(self, kind: str) -> Path:
        """The directory holding assets of `kind`.

        Raises:
            ValueError: If `kind` is not one of ASSET_KINDS.
        """
        if kind not in ASSET_KINDS:
            raise ValueError(f"Unknown asset kind '{kind}'. Expected one of {ASSET_KINDS}.")
        return Path(self.project_dir) / kind

    def _names(self, kind: str, on_disk: list[str]) -> list[str]:
        """`on_disk` with this draft's pending writes added and deletes removed."""
        pending = self.pending_assets.get(kind) or {}
        names = {name for name in on_disk if pending.get(name, "") is not None}
        names.update(name for name, content in pending.items() if content is not None)
        return sorted(names)

    def system_prompt_names(self) -> list[str]:
        """Bare .txt filenames available to a candidate's system_prompt_file."""
        directory = self.system_prompts_dir
        on_disk = (
            [p.name for p in directory.iterdir() if p.is_file() and p.suffix == ".txt"]
            if directory.is_dir() else []
        )
        return self._names(SYSTEM_PROMPTS_DIR, on_disk)

    def test_file_names(self) -> list[str]:
        """Bare filenames available as a test case attachment."""
        directory = self.test_files_dir
        on_disk = (
            [p.name for p in directory.iterdir() if p.is_file()]
            if directory.is_dir() else []
        )
        return self._names(TEST_FILES_DIR, on_disk)

    def as_loaded(self) -> tuple[list[Candidate], JudgeConfig, list[TestCase]]:
        """The draft as a run would load it, for checks written against those classes.

        Goes through the same from_dict constructors the loaders use, so a
        fallback cannot differ between the two. System prompts are not resolved.

        Raises:
            Exception: Whatever the constructors raise on a malformed entry (a
                non-dict row, say). validate() reports those; a caller after
                anything else should catch and skip.
        """
        judge = JudgeConfig.from_dict(self.judge)
        judge.global_criteria = GlobalCriteria.from_dict(self.criteria)
        return (
            [Candidate.from_dict(c) for c in self.candidates],
            judge,
            [TestCase.from_dict(t) for t in self.tests],
        )


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
    """Read a project's config files as raw data.

    A missing or unparseable file loads as empty rather than raising: the editor
    is the tool you reach for to *fix* such a project.

    Args:
        project_dir: Path to the benchmark project directory.

    Returns:
        A ProjectDraft carrying both the parsed data and the original bytes.
    """
    base = Path(project_dir)
    raws = {name: _read_raw(base / name) for name in EDITABLE_FILENAMES}

    candidates = _parse(raws[CANDIDATES_FILE], [])
    tests = _parse(raws[TEST_CASES_FILE], [])
    judge = _parse(raws[JUDGE_CONFIG_FILE], {})
    criteria = _parse(raws[GLOBAL_CRITERIA_FILE], {})
    clusters = _parse(raws[GROUP_CLUSTERS_FILE], {})

    return ProjectDraft(
        project_dir=str(project_dir),
        candidates=candidates if isinstance(candidates, list) else [],
        tests=tests if isinstance(tests, list) else [],
        judge=judge if isinstance(judge, dict) else {},
        criteria=criteria if isinstance(criteria, dict) else {},
        clusters=clusters if isinstance(clusters, dict) else {},
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


def _payloads(draft: ProjectDraft) -> dict:
    """Each file's data, minus an optional file that is absent and still empty."""
    payloads = {
        CANDIDATES_FILE: draft.candidates,
        TEST_CASES_FILE: draft.tests,
        JUDGE_CONFIG_FILE: draft.judge,
        GLOBAL_CRITERIA_FILE: draft.criteria,
        GROUP_CLUSTERS_FILE: draft.clusters,
    }
    for name in OPTIONAL_FILENAMES:
        if draft.disk.get(name) is None and not payloads[name]:
            del payloads[name]
    return payloads


def serialize_all(draft: ProjectDraft) -> dict[str, bytes]:
    """Serialise every config file of a draft.

    Returns:
        Maps each of EDITABLE_FILENAMES to the bytes that would be written,
        leaving out an optional file that does not exist and would be empty.
        Only the HASHED_FILENAMES entries feed the run hash.
    """
    payloads = _payloads(draft)
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
    GROUP_CLUSTERS_FILE: GroupClusters.save,
}


def save_project(draft: ProjectDraft) -> list[str]:
    """Write only the config files that actually changed.

    Args:
        draft: The draft to persist.

    Returns:
        The filenames written, in EDITABLE_FILENAMES order.

    Raises:
        OSError: If a file cannot be written (see models._write_json_atomic).
    """
    payloads = _payloads(draft)
    pending = changed_files(draft)
    written = []
    for name in EDITABLE_FILENAMES:
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

    # An entry of the wrong shape is reported, never tripped over. load_project
    # is total by design (it reads a hand-written file to be fixed, not to be
    # trusted), so this function has to be total too, or it crashes on exactly
    # the input it exists to reject. The index is the file's, so the message
    # still points at the right entry after the malformed ones are skipped.
    bad_candidates = [i for i, c in enumerate(draft.candidates, start=1) if not isinstance(c, dict)]
    for index in bad_candidates:
        errors.append(f"Candidate #{index} is not an object.")
    candidates = [
        (i, c) for i, c in enumerate(draft.candidates, start=1) if isinstance(c, dict)
    ]

    # Candidates: name keys every progress.jsonl event and pools the results,
    # so a duplicate silently merges two candidates into one.
    names = [str(c.get("name") or "") for _, c in candidates]
    for dupe in _duplicates([n for n in names if n]):
        errors.append(f"Candidate name '{dupe}' is used more than once.")
    for index, cand in candidates:
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
        # An error, not a warning, although the loader degrades it to the
        # default: the candidate would quietly join another role's frontier.
        role = cand.get("role", CANDIDATE_ROLE_CANDIDATE)
        if not isinstance(role, str) or not role.strip():
            errors.append(f"Candidate '{label}' has an invalid role {role!r}: it must be non-empty text.")

    # Test cases: id keys the same events.
    for index, test in enumerate(draft.tests, start=1):
        if not isinstance(test, dict):
            errors.append(f"Test case #{index} is not an object.")
    tests = [(i, t) for i, t in enumerate(draft.tests, start=1) if isinstance(t, dict)]

    ids = [str(t.get("id") or "") for _, t in tests]
    for dupe in _duplicates([i for i in ids if i]):
        errors.append(f"Test case id '{dupe}' is used more than once.")
    for index, test in tests:
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

    # Judge config. `or {}` alone is not enough: it falls through for None and
    # for an empty dict, but a truthy non-mapping ("oops") reaches .get and
    # raises. That shape is a plausible hand edit, and the loaders raise on it
    # too, so it has to be reported here rather than crash the check.
    def _block(key: str) -> dict:
        value = draft.judge.get(key)
        if value is not None and not isinstance(value, dict):
            errors.append(f"judge_config.{key} must be an object.")
            return {}
        return value or {}

    test_judge = _block("test_judge")
    template = test_judge.get("evaluation_template")
    if isinstance(template, str) and template:
        error, warning = check_evaluation_template(template)
        if error:
            errors.append(f"test_judge.evaluation_template {error}")
        if warning:
            warnings.append(f"test_judge.evaluation_template {warning}")

    verdict_judge = _block("verdict_judge")
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

    # Group clusters. A file that exists but did not load as an object is an
    # error, not a quiet `{}`: saving would otherwise overwrite the author's
    # hand-written content with an empty object.
    clusters_raw = draft.disk.get(GROUP_CLUSTERS_FILE)
    if clusters_raw is not None and not isinstance(_parse(clusters_raw, None), dict) \
            and not draft.clusters:
        errors.append(
            f"{GROUP_CLUSTERS_FILE} is not a readable JSON object; fix it by hand "
            "before saving, or the save would replace it."
        )
    elif draft.clusters or clusters_raw is not None:
        cluster_errors, cluster_warnings = check_group_clusters(draft.clusters)
        errors.extend(f"{GROUP_CLUSTERS_FILE}: {e}" for e in cluster_errors)
        warnings.extend(f"{GROUP_CLUSTERS_FILE}: {w}" for w in cluster_warnings)
        if not cluster_errors:
            groups = list(dict.fromkeys(
                str(t.get("group") or DEFAULT_GROUP) for _, t in tests
            ))
            warnings.extend(reference_warnings(
                GroupClusters.from_dict(draft.clusters), groups,
                [str(c.get("name")) for _, c in candidates if c.get("name")],
            ))

    # Orphans, in both directions. Over the well-formed entries only: a
    # malformed one is already an error, and it has no references to honour.
    used_prompts = {c.get("system_prompt_file") for _, c in candidates}
    for name in sorted(prompts - {p for p in used_prompts if p}):
        warnings.append(f"System prompt '{name}' is not used by any candidate.")
    used_files = set()
    for _, test in tests:
        try:
            used_files.update(attachment_paths(test.get("file")))
        except ValueError:
            continue  # already reported as an error above
    for path in sorted(attachments - used_files):
        warnings.append(f"Attachment '{path}' is not used by any test case.")

    return errors, warnings


# ── Save status ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SaveStatus:
    """Everything a caller needs to decide whether, and how, to write a draft.

    Computed in one place because every interface needs the same four answers
    and gets them wrong differently: what would change, what blocks the write,
    whether an existing run survives it, and whether the change is real at all.
    """

    changed: list[str]
    errors: list[str]
    warnings: list[str]
    stored_hash: str | None
    would_be_hash: str
    invalidates: bool
    reformat_only: list[str]


def save_status(draft: ProjectDraft) -> SaveStatus:
    """Validate a draft and compare it against what is on disk.

    `invalidates` is the expensive one: the project has a progress.jsonl whose
    stored hash no longer matches what this draft would produce, so the next run
    renames it to progress.jsonl.bak and starts over. A draft with no log at all
    can never invalidate anything.

    `reformat_only` names changed files whose JSON content is identical, which
    is worth surfacing separately: three of the four are hashed byte for byte,
    so pure reindentation still costs the user their progress.

    Args:
        draft: The draft to inspect. Not modified.

    Returns:
        A SaveStatus. `errors` non-empty must block the write.
    """
    pending = serialize_all(draft)
    changed = [
        name for name in EDITABLE_FILENAMES
        if name in pending and pending[name] != draft.disk.get(name)
    ]
    errors, warnings = validate(draft)

    stored = read_stored_hash(draft.project_dir)
    would_be = config_hash_from_bytes(pending)

    # Hashed files only: reformatting an unhashed one costs nothing, and this
    # list exists to explain a hash change the content does not account for.
    reformat_only = []
    for name in changed:
        if name not in HASHED_FILENAMES:
            continue
        try:
            if json.loads(pending[name]) == json.loads(draft.disk.get(name) or b"null"):
                reformat_only.append(name)
        except (ValueError, TypeError):
            pass

    return SaveStatus(
        changed=changed,
        errors=errors,
        warnings=warnings,
        stored_hash=stored,
        would_be_hash=would_be,
        invalidates=stored is not None and stored != would_be,
        reformat_only=reformat_only,
    )


# ── Assets: system_prompts/*.txt and test_files/* ─────────────────────────────
# Not in the run hash (see ASSETS_NOT_HASHED), but referenced by name from two
# of the files that are, so deleting one behind a live reference silently breaks
# the run. Both editing paths share these guards.


def asset_users(draft: ProjectDraft, kind: str, name: str) -> list[str]:
    """Who still refers to this asset.

    Args:
        draft: The draft to search, in its current (post-edit) state.
        kind: One of ASSET_KINDS.
        name: The bare filename.

    Returns:
        Candidate names for a system prompt, test case ids for an attachment.

    Raises:
        ValueError: If `kind` is not one of ASSET_KINDS.
    """
    if kind == SYSTEM_PROMPTS_DIR:
        return [
            str(cand.get("name") or "")
            for cand in draft.candidates
            if cand.get("system_prompt_file") == name
        ]
    if kind == TEST_FILES_DIR:
        target = f"{TEST_FILES_DIR}/{name}"
        users = []
        for test in draft.tests:
            try:
                paths = attachment_paths(test.get("file"))
            except ValueError:
                continue  # a malformed `file` is reported by validate(), not here
            if target in paths:
                users.append(str(test.get("id") or ""))
        return users
    raise ValueError(f"Unknown asset kind '{kind}'. Expected one of {ASSET_KINDS}.")


def check_asset_name(kind: str, name: str) -> str | None:
    """Reject a filename an asset directory must not hold. Returns an error, or None."""
    problem = check_filename(name)
    if problem:
        return problem
    if kind == SYSTEM_PROMPTS_DIR and not name.endswith(".txt"):
        return "System prompt files must end in .txt"
    return None


def write_asset(draft: ProjectDraft, kind: str, name: str, content: str | bytes) -> None:
    """Write one system prompt or attachment straight to disk.

    Text is written with newline="\\n" rather than the platform default,
    matching seed_project() and the loaders, so the same project reads
    identically wherever it was authored.

    Args:
        draft: The draft the asset belongs to.
        kind: One of ASSET_KINDS.
        name: The bare filename.
        content: Text, or bytes for a binary attachment.

    Raises:
        ValueError: If `kind` or `name` is not acceptable.
        OSError: If the file cannot be written.
    """
    problem = check_asset_name(kind, name)
    if problem:
        raise ValueError(problem)
    directory = draft.asset_dir(kind)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="\n")


def delete_asset(draft: ProjectDraft, kind: str, name: str) -> None:
    """Delete one system prompt or attachment, unless something still uses it.

    Args:
        draft: The draft the asset belongs to, in its current (post-edit) state.
        kind: One of ASSET_KINDS.
        name: The bare filename.

    Raises:
        ValueError: If `kind`/`name` is invalid, the file is absent, or a
            candidate or test case still refers to it.
        OSError: If the file cannot be removed.
    """
    problem = check_filename(name)
    if problem:
        raise ValueError(problem)
    users = asset_users(draft, kind, name)
    if users:
        raise ValueError(f"'{name}' is still used by {', '.join(users)}.")
    path = draft.asset_dir(kind) / name
    if not path.is_file():
        raise ValueError(f"'{name}' does not exist in {kind}/.")
    path.unlink()


def externally_modified(draft: ProjectDraft) -> list[str]:
    """Config files whose bytes on disk no longer match what was loaded.

    Another editor tab, a git checkout or a hand edit — saving over it would
    silently revert someone's work.
    """
    base = Path(draft.project_dir)
    return [
        name for name in EDITABLE_FILENAMES
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
    secrets.json into os.getcwd() (config.py:356) — which, for a GUI, is
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
