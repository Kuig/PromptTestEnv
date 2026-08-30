"""Read and edit a benchmark project without a file editor.

This is the headless half of what `prompttestenv editor` does with forms, and
the only module the CLI, the MCP server and a calling script go through. Both
paths share `projectio`, so an edit made from here inherits every rule the
Streamlit editor obeys: byte-faithful writes, the same validation, the same
refusal to quietly cost someone a finished run.

**Why a patch and not a whole file.** A caller sends only what changes.
Candidates are identified by `name`, test cases by `id`, and everything the
patch does not mention survives untouched, extra keys and key order included.
That matters more here than it looks: three of the four config files feed the
resume hash as RAW BYTES, so a caller who reads a project, re-serialises it and
writes it all back would invalidate a run without editing anything. A patch
cannot make that mistake, because the bytes it does not speak about are never
rewritten.

**The hash gate.** When a project already holds a progress.jsonl and the edit
would change the config hash, `edit_project` REFUSES and says so, until the
caller passes `force=True`. This is the headless equivalent of the editor's
confirmation dialog, and it exists because the alternative is an agent silently
discarding an expensive run while tidying up a prompt. Nothing is ever deleted
either way: the next run renames the log to progress.jsonl.bak.

Two asymmetries worth knowing before reading further:

- Editing only `reasoning_analysis` or the `reasoning_judge` block never trips
  the gate. `progress` strips both before hashing precisely so that tuning the
  measurement never costs a re-run of the measured.
- `system_prompts/` and `test_files/` are not hashed at all, so changing one
  never trips the gate either, and a resumed run will happily mix responses
  produced under the old and the new version. That is the more dangerous of the
  two, so it comes back as a warning (`projectio.ASSETS_NOT_HASHED`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prompttestenv import projectio as pio
from prompttestenv.models import (
    Candidate,
    GlobalCriteria,
    TestCase,
    attachment_paths,
)

# Every key a patch document may carry. Anything else is an error rather than a
# no-op: a patch is typically machine-written, and a typo that silently does
# nothing is far worse than one that says so. Same posture as
# models._parse_reasoning_scope, which rejects an unknown scope instead of
# letting it fall through to a truthiness test.
PATCH_KEYS = (
    "candidates",
    "test_cases",
    "judge_config",
    "global_criteria",
    "system_prompts",
    "test_files",
    "delete",
    "order",
)

# The two list-shaped sections, and what identifies an entry in each.
_SECTIONS: dict[str, tuple[str, type]] = {
    "candidates": ("name", Candidate),
    "test_cases": ("id", TestCase),
}

# Patch section -> the ProjectDraft attribute it edits.
_DRAFT_ATTR = {"candidates": "candidates", "test_cases": "tests"}

_DELETE_KEYS = ("candidates", "test_cases", "system_prompts", "test_files")


@dataclass(frozen=True)
class EditResult:
    """The outcome of one `edit_project` call.

    `written` and `deleted` list project-relative paths, so they name both
    `candidates.json` and `system_prompts/pirate_prompt.txt`. Both empty on a
    patch that changed nothing, which is a success, not a failure.
    """

    ok: bool
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hash_changed: bool = False
    stored_hash: str | None = None
    new_hash: str = ""
    dry_run: bool = False

    def to_dict(self) -> dict:
        """This result as plain JSON-serialisable data."""
        return asdict(self)

    def summary(self) -> str:
        """One line, in the same shape the runner entry points return.

        Failures start with ``Error:`` so the GUI banner classifier and any
        caller matching on that prefix treat them the way they treat a failed
        run.
        """
        if not self.ok:
            return "Error: " + (self.errors[0] if self.errors else "edit failed.")
        if self.dry_run:
            parts = []
            if self.written:
                parts.append(f"would write {', '.join(self.written)}")
            if self.deleted:
                parts.append(f"would delete {', '.join(self.deleted)}")
            if not parts:
                return "Dry run: nothing would change."
            gate = " Passing force would be required." if self.hash_changed else ""
            return f"Dry run: {'; '.join(parts)}.{gate}"
        parts = []
        if self.written:
            parts.append(f"Wrote {', '.join(self.written)}")
        if self.deleted:
            parts.append(f"Deleted {', '.join(self.deleted)}")
        return ". ".join(parts) + "." if parts else "Nothing to change."


def _failure(errors: list[str], warnings: list[str] | None = None) -> EditResult:
    return EditResult(ok=False, errors=errors, warnings=warnings or [])


def read_project(project_dir: str) -> dict:
    """Read a whole project as plain data.

    The four config files come back exactly as they sit on disk, unknown keys
    and all, because that is what a patch has to be written against.

    System prompts come back with their content: they are text, they are small,
    and a caller with no file reader cannot edit one it has not seen. Test files
    come back as names and sizes only, since an attachment can be a large binary
    and nothing useful would come of inlining it.

    Args:
        project_dir: Path to the benchmark project directory.

    Returns:
        A JSON-serialisable dict. `progress_valid` answers whether an existing
        progress.jsonl still matches the config on disk; it is True when there
        is no log at all.
    """
    base = Path(project_dir)
    draft = pio.load_project(project_dir)
    status = pio.save_status(draft)

    prompts = {}
    for name in draft.system_prompt_names():
        try:
            prompts[name] = (draft.system_prompts_dir / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            prompts[name] = f"[unreadable: {exc}]"

    files = []
    for name in draft.test_file_names():
        path = draft.test_files_dir / name
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        files.append({"name": name, "path": f"{pio.TEST_FILES_DIR}/{name}", "bytes": size})

    return {
        "project_dir": str(project_dir),
        "exists": base.is_dir(),
        "candidates": draft.candidates,
        "test_cases": draft.tests,
        "judge_config": draft.judge,
        "global_criteria": draft.criteria,
        "system_prompts": prompts,
        "test_files": files,
        "errors": status.errors,
        "warnings": status.warnings,
        "config_hash": status.would_be_hash,
        "stored_hash": status.stored_hash,
        "progress_valid": not status.invalidates,
    }


def _apply_section(draft: pio.ProjectDraft, section: str, entries, errors: list[str]) -> None:
    """Upsert a list of partial entries into `candidates` or `test_cases`."""
    key_field, cls = _SECTIONS[section]
    attribute = _DRAFT_ATTR[section]
    if not isinstance(entries, list):
        errors.append(f"'{section}' must be a list of objects.")
        return
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"{section}[{index}] must be an object.")
            continue
        if not entry.get(key_field):
            errors.append(f"{section}[{index}] has no '{key_field}', which identifies it.")
            continue
        values = dict(entry)
        if section == "test_cases" and "file" in values:
            # The one value a patch may spell three ways. Routing it through the
            # same pair the editor uses normalises separators and collapses a
            # one-element list back to a plain string, so a project authored on
            # Windows runs on POSIX and a no-op patch stays a no-op.
            try:
                values["file"] = pio.attachment_value(attachment_paths(values["file"]))
            except ValueError as exc:
                errors.append(f"{section}[{index}]: {exc}")
                continue
            if values["file"] is None:
                values.pop("file")
                current = getattr(draft, attribute)
                for item in current:
                    if item.get(key_field) == entry[key_field]:
                        item.pop("file", None)
        setattr(
            draft, attribute,
            pio.upsert(getattr(draft, attribute), key_field, values, cls),
        )


def _apply_delete(draft: pio.ProjectDraft, spec, errors: list[str]) -> None:
    if not isinstance(spec, dict):
        errors.append("'delete' must be an object.")
        return
    for key in spec:
        if key not in _DELETE_KEYS:
            errors.append(f"delete.{key} is not one of {', '.join(_DELETE_KEYS)}.")

    for section in ("candidates", "test_cases"):
        names = spec.get(section)
        if names is None:
            continue
        if not isinstance(names, list):
            errors.append(f"delete.{section} must be a list of names.")
            continue
        key_field = _SECTIONS[section][0]
        attribute = _DRAFT_ATTR[section]
        items = getattr(draft, attribute)
        present = {item.get(key_field) for item in items}
        for name in names:
            if name not in present:
                errors.append(f"delete.{section}: no entry with {key_field} '{name}'.")
        setattr(draft, attribute,
                [item for item in items if item.get(key_field) not in names])

    for kind in pio.ASSET_KINDS:
        names = spec.get(kind)
        if names is None:
            continue
        if not isinstance(names, list):
            errors.append(f"delete.{kind} must be a list of filenames.")
            continue
        for name in names:
            draft.pending_assets[kind][name] = None


def _apply_order(draft: pio.ProjectDraft, spec, errors: list[str]) -> None:
    if not isinstance(spec, dict):
        errors.append("'order' must be an object.")
        return
    for section, wanted in spec.items():
        if section not in _SECTIONS:
            errors.append(f"order.{section} is not one of candidates, test_cases.")
            continue
        if not isinstance(wanted, list):
            errors.append(f"order.{section} must be a list of names.")
            continue
        key_field = _SECTIONS[section][0]
        attribute = _DRAFT_ATTR[section]
        items = getattr(draft, attribute)
        current = [item.get(key_field) for item in items]
        if sorted(map(str, wanted)) != sorted(map(str, current)):
            # A partial order would silently drop whatever it omitted, which is
            # a deletion the caller did not ask for.
            errors.append(
                f"order.{section} must list every entry exactly once; "
                f"got {wanted}, expected a permutation of {current}."
            )
            continue
        by_key = {item.get(key_field): item for item in items}
        setattr(draft, attribute, [by_key[name] for name in wanted])


def _apply_assets(draft: pio.ProjectDraft, kind: str, spec, errors: list[str]) -> None:
    if not isinstance(spec, dict):
        errors.append(f"'{kind}' must be an object mapping filename to content.")
        return
    for name, content in spec.items():
        problem = pio.check_asset_name(kind, name)
        if problem:
            errors.append(f"{kind}/{name}: {problem}")
            continue
        if content is None:
            errors.append(
                f"{kind}/{name}: use delete.{kind} to remove a file, not a null value."
            )
            continue
        if not isinstance(content, str):
            errors.append(
                f"{kind}/{name}: content must be text. Binary attachments cannot "
                "travel in a patch and have to be placed in the directory directly."
            )
            continue
        draft.pending_assets[kind][name] = content


def _apply_patch(draft: pio.ProjectDraft, patch: dict, errors: list[str]) -> None:
    """Fold a patch document into `draft`, appending anything wrong to `errors`."""
    for key in patch:
        if key not in PATCH_KEYS:
            errors.append(f"Unknown patch key '{key}'. Expected one of {', '.join(PATCH_KEYS)}.")

    for section in _SECTIONS:
        if section in patch:
            _apply_section(draft, section, patch[section], errors)

    if "judge_config" in patch:
        values = patch["judge_config"]
        if not isinstance(values, dict):
            errors.append("'judge_config' must be an object.")
        else:
            try:
                draft.judge = pio.merge_judge(draft.judge, values)
            except ValueError as exc:
                errors.append(str(exc))

    if "global_criteria" in patch:
        values = patch["global_criteria"]
        if not isinstance(values, dict):
            errors.append("'global_criteria' must be an object.")
        else:
            draft.criteria = pio.merge_preserving_shape(
                draft.criteria, values, GlobalCriteria
            )

    for kind in pio.ASSET_KINDS:
        if kind in patch:
            _apply_assets(draft, kind, patch[kind], errors)

    # Deletes run last, so a patch may add a replacement and drop the old entry
    # in one call without the two fighting over ordering.
    if "delete" in patch:
        _apply_delete(draft, patch["delete"], errors)
    if "order" in patch:
        _apply_order(draft, patch["order"], errors)


def _asset_plan(draft: pio.ProjectDraft, errors: list[str]) -> list[tuple[str, str, object]]:
    """Check the staged asset operations and return them as (kind, name, content).

    Deletion is checked against the POST-patch draft, so dropping a system
    prompt and the candidate that used it in the same call is allowed, while
    dropping one still referenced is not.
    """
    plan: list[tuple[str, str, object]] = []
    for kind in pio.ASSET_KINDS:
        for name, content in (draft.pending_assets.get(kind) or {}).items():
            if content is None:
                users = pio.asset_users(draft, kind, name)
                if users:
                    errors.append(
                        f"Cannot delete {kind}/{name}: still used by {', '.join(users)}."
                    )
                    continue
                if not (draft.asset_dir(kind) / name).is_file():
                    errors.append(f"Cannot delete {kind}/{name}: it does not exist.")
                    continue
            plan.append((kind, name, content))
    return plan


def _hash_gate_error(status: pio.SaveStatus) -> str:
    reformat = ""
    if status.reformat_only:
        reformat = (
            f" Reformatting only in {', '.join(status.reformat_only)}: the content "
            "is unchanged, but those files are hashed byte for byte, so the hash "
            "changes anyway."
        )
    return (
        f"This edit changes the config hash ({status.stored_hash[:8]} -> "
        f"{status.would_be_hash[:8]}), so the existing progress.jsonl no longer "
        f"matches and the next run would rename it to progress.jsonl.bak and start "
        f"over. Changed: {', '.join(status.changed)}.{reformat} "
        "Nothing has been written. Pass force to proceed anyway."
    )


def edit_project(
    project_dir: str,
    patch: dict,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> EditResult:
    """Apply a patch document to a project's configuration.

    See the module docstring for the patch language and the hash gate. Never
    raises: every failure comes back as ``ok=False`` with a populated `errors`,
    matching the contract the runner entry points keep.

    Args:
        project_dir: Path to an existing benchmark project directory.
        patch: The patch document. An empty patch is legal and writes nothing.
        dry_run: Validate and report, but write nothing at all.
        force: Write even when the edit invalidates an existing progress.jsonl.

    Returns:
        An EditResult. `warnings` is populated on success too, and is worth
        surfacing: it carries the missing-attachment and unused-prompt findings
        that do not block a run but usually mean a typo.
    """
    try:
        if not Path(project_dir).is_dir():
            return _failure(
                [f"Project folder '{project_dir}' does not exist. Use 'init' first."]
            )
        if not isinstance(patch, dict):
            return _failure(["The patch must be a JSON object."])

        draft = pio.load_project(project_dir)
        errors: list[str] = []
        _apply_patch(draft, patch, errors)
        if errors:
            return _failure(errors)

        plan = _asset_plan(draft, errors)
        if errors:
            return _failure(errors)

        status = pio.save_status(draft)
        warnings = list(status.warnings)
        if plan:
            warnings.append(pio.ASSETS_NOT_HASHED)

        if status.errors:
            return EditResult(
                ok=False,
                errors=status.errors,
                warnings=warnings,
                hash_changed=status.invalidates,
                stored_hash=status.stored_hash,
                new_hash=status.would_be_hash,
            )

        would_write = [
            *status.changed,
            *(f"{kind}/{name}" for kind, name, content in plan if content is not None),
        ]
        would_delete = [f"{kind}/{name}" for kind, name, content in plan if content is None]

        # A dry run answers "what would this do?", so it reports the hash change
        # rather than refusing over it. Nothing is written either way, and an
        # agent that has to pass force blindly just to see the answer would be
        # learning the wrong habit.
        if dry_run:
            return EditResult(
                ok=True,
                written=would_write,
                deleted=would_delete,
                warnings=warnings,
                hash_changed=status.invalidates,
                stored_hash=status.stored_hash,
                new_hash=status.would_be_hash,
                dry_run=True,
            )

        if status.invalidates and not force:
            return EditResult(
                ok=False,
                errors=[_hash_gate_error(status)],
                warnings=warnings,
                hash_changed=True,
                stored_hash=status.stored_hash,
                new_hash=status.would_be_hash,
            )

        written = list(pio.save_project(draft))
        deleted = []
        for kind, name, content in plan:
            if content is None:
                (draft.asset_dir(kind) / name).unlink()
                deleted.append(f"{kind}/{name}")
            else:
                pio.write_asset(draft, kind, name, content)
                written.append(f"{kind}/{name}")
        draft.pending_assets = {kind: {} for kind in pio.ASSET_KINDS}

        return EditResult(
            ok=True,
            written=written,
            deleted=deleted,
            warnings=warnings,
            hash_changed=status.invalidates,
            stored_hash=status.stored_hash,
            new_hash=status.would_be_hash,
        )
    except (OSError, ValueError) as exc:
        return _failure([str(exc)])


def parse_patch(text: str) -> dict:
    """Parse a patch document from JSON text.

    Args:
        text: The JSON source.

    Returns:
        The parsed object.

    Raises:
        ValueError: If the text is not JSON, or is not a JSON object.
    """
    try:
        patch = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Patch is not valid JSON: {exc}") from exc
    if not isinstance(patch, dict):
        raise ValueError("Patch must be a JSON object.")
    return patch
