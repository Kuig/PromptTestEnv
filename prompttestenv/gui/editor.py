"""Streamlit project editor — `prompttestenv editor`.

Creates and modifies benchmark projects. It never runs one: `prompttestenv gui`
does that.

Two things govern the implementation and are easy to break by accident:

**Byte fidelity.** Three of the four config files feed the run hash as raw
bytes, so opening a project and saving it unchanged must not alter a single
byte. All of that lives in `projectio`; this module only feeds it edited values.

**Harvest-then-render.** Streamlit garbage-collects the state of widgets that a
run did not render, so reading widget keys at save time breaks the moment a
filter hides a row. Instead the model in `st.session_state.ed` is the single
source of truth: `_harvest()` copies last run's widget values into it *before*
any widget is created, and every widget then renders from the model.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

import prompttestenv.logger as logger
from prompttestenv.config import get_app_config
from prompttestenv.gui import projectio as pio
from prompttestenv.gui.common import pick_directory
from prompttestenv.models import (
    REASONING_SCOPES,
    ReasoningJudgeSettings,
    SimilarityJudgeSettings,
    TestJudgeSettings,
    VerdictJudgeSettings,
)
from prompttestenv.progress import (
    HASHED_FILENAMES,
    config_hash_from_bytes,
    read_stored_hash,
)

logger.set_backend("streamlit")

st.set_page_config(page_title="PromptTestEnv Editor", page_icon="📝", layout="wide")
st.title("📝 PromptTestEnv — Project Editor")

JUDGE_TYPES = ("llm-judge", "similarity", "assert")
GLOBAL_MODES = ("llm-judge", "similarity", "assert", "none")
DIMENSION_MODES = ("split", "joint")

_NULLABLE_SENTINEL = "(config.json default)"


# ── Model ─────────────────────────────────────────────────────────────────────

def _rows(items: list[dict]) -> list[dict]:
    """Wrap raw dicts as uid-tagged rows.

    The uid lives OUTSIDE the data dict on purpose. Carrying it inside as a
    private key and stripping it on save is one forgotten strip away from
    writing it into candidates.json and changing that file's hash forever.
    """
    return [{"uid": uuid4().hex, "data": item} for item in items]


def _load(project_dir: str) -> None:
    """Load a project into session state, invalidating every widget key."""
    draft = pio.load_project(project_dir)
    previous = st.session_state.get("ed", {}).get("gen", 0)
    st.session_state.ed = {
        "project_dir": project_dir,
        # Bumping this changes every widget key, so every widget is created
        # fresh and seeds from the model. Without it, session state would win
        # over `value=` and the form would keep showing the previous project.
        "gen": previous + 1,
        "draft": draft,
        "candidates": _rows(draft.candidates),
        "tests": _rows(draft.tests),
        "confirm_save": False,
        "flash": None,
    }


def _ed() -> dict | None:
    return st.session_state.get("ed")


def _wkey(kind: str, uid: str, field: str) -> str:
    return f"ed{st.session_state.ed['gen']}:{kind}:{uid}:{field}"


def _sync_draft() -> None:
    """Point the draft's lists at the current row data, in current order."""
    ed = st.session_state.ed
    ed["draft"].candidates = [row["data"] for row in ed["candidates"]]
    ed["draft"].tests = [row["data"] for row in ed["tests"]]


# Field name -> how to read the widget value back. `None` means "as is".
_CANDIDATE_FIELDS = ("name", "provider", "model", "temperature",
                     "disable_safety", "thinking", "system_prompt_file")
_TEST_FIELDS = ("id", "group", "judge_type", "prompt", "criteria", "file")

# judge_config.json widgets: stable widget key -> field name. Top-level scalars
# first, then one map per nested block. Harvesting from a table rather than
# assigning during render is what lets validation see an edit on the SAME run
# the user made it, instead of one run later.
_JUDGE_TOP_KEYS = {
    "jc:repetitions": "repetitions",
    "jc:rep_delay": "repetition_delay_seconds",
    "jc:timeout": "max_response_timeout_seconds",
    "jc:eval_delay": "evaluation_delay_seconds",
    "jc:media": "pass_media_to_judge",
    "jc:groups": "group_verdicts",
    "jc:scope": "reasoning_analysis",
}

_JUDGE_BLOCK_KEYS = {
    "test_judge": {
        "tj:provider": "provider", "tj:model": "model",
        "tj:temperature": "temperature", "tj:thinking": "thinking",
        "tj:safety": "disable_safety",
        "tj:sys": "evaluation_system_prompt", "tj:template": "evaluation_template",
    },
    "similarity_judge": {"sj:provider": "provider", "sj:model": "model"},
    "verdict_judge": {
        "vj:provider": "provider", "vj:model": "model",
        "vj:temperature": "temperature", "vj:thinking": "thinking",
        "vj:safety": "disable_safety",
        "vj:sys": "verdict_system_prompt", "vj:template": "verdict_template",
        "vj:global": "global_verdict_template",
    },
    "reasoning_judge": {
        "rj:provider": "provider", "rj:model": "model",
        "rj:temperature": "temperature", "rj:thinking": "thinking",
    },
}

# Nullable reasoning_judge settings: None genuinely means "fall back to
# config.json", which no number input can express — hence the checkbox pair.
_JUDGE_NULLABLES = (
    ("rj:ctx", "context_size"),
    ("rj:k", "reliability_k"),
    ("rj:units", "max_units_per_call"),
)

_GLOBAL_KEYS = {
    "gc:mode": "mode",
    "gc:llm_judge_criteria": "llm_judge_criteria",
    "gc:similarity_criteria": "similarity_criteria",
    "gc:assert_criteria": "assert_criteria",
    "gc:other:llm_judge_criteria": "llm_judge_criteria",
    "gc:other:similarity_criteria": "similarity_criteria",
    "gc:other:assert_criteria": "assert_criteria",
}


def _harvest_row(kind: str, row: dict, names: tuple[str, ...], cls: type) -> None:
    values = {}
    for name in names:
        key = _wkey(kind, row["uid"], name)
        if key in st.session_state:
            values[name] = st.session_state[key]
    if values:
        row["data"] = pio.merge_preserving_shape(row["data"], values, cls)


def _harvest_mapping(target: dict, key_map: dict[str, str], cls: type) -> dict:
    values = {
        field: st.session_state[key]
        for key, field in key_map.items()
        if key in st.session_state
    }
    return pio.merge_preserving_shape(target, values, cls) if values else target


def _harvest_nullables(block: dict) -> dict:
    """Read the checkbox+input pairs of reasoning_judge's nullable settings.

    Unchecking means None ("fall back to config.json"), which is only written
    when the key was already in the file — adding an explicit null where the
    author simply omitted the key would be a change they did not make.
    """
    out = dict(block)
    for key, field_name in _JUDGE_NULLABLES:
        if f"{key}:on" not in st.session_state:
            continue
        if not st.session_state[f"{key}:on"]:
            if field_name in out:
                out[field_name] = None
        elif key in st.session_state:
            out[field_name] = int(st.session_state[key])
    return out


def _harvest() -> None:
    """Copy last run's widget values into the model, before rendering anything.

    A key that is absent — because its widget was filtered out, or Streamlit
    collected it — simply leaves the model's own value alone. That is what makes
    filtering and collapsing safe.
    """
    from prompttestenv.models import Candidate, GlobalCriteria, JudgeConfig, TestCase

    ed = st.session_state.ed
    draft = ed["draft"]

    for row in ed["candidates"]:
        _harvest_row("cand", row, _CANDIDATE_FIELDS, Candidate)
    for row in ed["tests"]:
        _harvest_row("test", row, _TEST_FIELDS, TestCase)

    draft.judge = _harvest_mapping(draft.judge, _JUDGE_TOP_KEYS, JudgeConfig)
    for block_name, key_map in _JUDGE_BLOCK_KEYS.items():
        existing = draft.judge.get(block_name)
        if not isinstance(existing, dict) and existing is not None:
            continue
        merged = _harvest_mapping(existing or {}, key_map, pio.JUDGE_BLOCKS[block_name])
        if block_name == "reasoning_judge":
            merged = _harvest_nullables(merged)
        # A block the file never had is only created once it holds something
        # other than defaults. Otherwise merely opening the tab would add an
        # empty block — and for a hashed file that is a real change.
        if existing is not None or merged:
            draft.judge[block_name] = merged

    draft.criteria = _harvest_mapping(draft.criteria, _GLOBAL_KEYS, GlobalCriteria)
    _sync_draft()


# ── Widget helpers ────────────────────────────────────────────────────────────

def _open_choice(label: str, options, current, key: str, help_text: str | None = None):
    """A selectbox that offers `options` but keeps an out-of-vocabulary value."""
    choices = list(dict.fromkeys([*options, current] if current else list(options)))
    index = choices.index(current) if current in choices else 0
    return st.selectbox(label, choices, index=index, key=key, help=help_text,
                        accept_new_options=True)


def _nullable_number(label: str, current, key: str, *, minimum: int, step: int,
                     fallback, help_text: str) -> None:
    """A checkbox+input pair for a field where None means "use config.json".

    st.number_input cannot express None, and None here is a real, distinct
    setting rather than a missing one. Renders only — _harvest reads it back.
    """
    enabled = st.checkbox(
        f"Set {label}", value=current is not None, key=f"{key}:on", help=help_text,
    )
    if not enabled:
        st.caption(f"Using the config.json default: {fallback}")
        return
    st.number_input(
        label, min_value=minimum, step=step,
        value=int(current) if current is not None else int(minimum), key=key,
    )


def _assert_panel(criteria: str) -> None:
    """Show what test_judge.py will actually evaluate, and whether it parses."""
    st.caption("This is the expression test_judge.py will build:")
    st.code(pio.effective_lambda(criteria), language="python")
    problem = pio.check_assert_criteria(criteria)
    if problem:
        st.error(f"Cannot be parsed: {problem}")
    else:
        st.success("Parses as a lambda.")
    st.caption(
        "Must return a `(score, reasoning)` tuple. The score is NOT clamped to "
        "1-10 for assert — returning -1 deliberately means 'not measured'."
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.session_state.setdefault("path_input", "")

with st.sidebar:
    st.header("⚙️ Project")
    path_input = st.text_input(
        "Project directory path",
        value=st.session_state.path_input,
        placeholder="Projects/MyBenchmark",
    )
    st.session_state.path_input = path_input

    if st.button("📁 Browse...", width="stretch"):
        init_dir = st.session_state.path_input or str(Path.cwd())
        if not Path(init_dir).is_absolute():
            init_dir = str(Path.cwd() / init_dir)
        picked = None
        try:
            picked = pick_directory(init_dir)
        except Exception as exc:
            st.warning(f"Folder picker unavailable: {exc}. Type the path manually.")
        if picked:
            try:
                st.session_state.path_input = Path(picked).relative_to(Path.cwd()).as_posix()
            except ValueError:
                st.session_state.path_input = Path(picked).as_posix()
            st.rerun()

    col_open, col_new = st.columns(2)
    if col_open.button("📂 Open", width="stretch", type="primary"):
        if not path_input:
            st.error("Project directory is required.")
        elif not Path(path_input).is_dir():
            st.error(f"'{path_input}' is not a directory.")
        else:
            _load(path_input)
            st.rerun()

    if col_new.button("✨ New", width="stretch"):
        if not path_input:
            st.error("Project directory is required.")
        else:
            try:
                created = pio.seed_project(path_input)
                _load(path_input)
                st.session_state.ed["flash"] = (
                    "success",
                    f"Created {len(created)} file(s) from the templates."
                    if created else "Directory already had every config file.",
                )
                st.rerun()
            except OSError as exc:
                st.error(f"Error: {exc}")

if _ed() is None:
    st.info("Open an existing project directory, or create one with **New**.")
    st.stop()

_harvest()

ed = st.session_state.ed
draft = ed["draft"]
pending = pio.serialize_all(draft)
changed = [n for n in HASHED_FILENAMES if pending[n] != draft.disk.get(n)]
errors, warnings = pio.validate(draft)

stored_hash = read_stored_hash(draft.project_dir)
would_be_hash = config_hash_from_bytes(pending)
invalidates = stored_hash is not None and stored_hash != would_be_hash


def _semantically_identical() -> list[str]:
    """Changed files whose JSON content is unchanged — pure reformatting.

    Worth naming explicitly: for the three byte-hashed files this still costs
    the user their progress, which is deeply unobvious.
    """
    import json
    same = []
    for name in changed:
        try:
            if json.loads(pending[name]) == json.loads((draft.disk.get(name) or b"null")):
                same.append(name)
        except (ValueError, TypeError):
            pass
    return same


def _do_save() -> None:
    try:
        written = pio.save_project(draft)
    except OSError as exc:
        ed["flash"] = ("error", f"Could not write: {exc}")
        return
    ed["confirm_save"] = False
    ed["flash"] = ("success", f"Saved: {', '.join(written)}" if written else "Nothing to save.")


with st.sidebar:
    st.divider()
    st.caption(f"📁 `{draft.project_dir}`")

    if changed:
        st.warning("Unsaved changes: " + ", ".join(changed))
    else:
        st.success("No unsaved changes.")

    if errors:
        st.error(f"{len(errors)} error(s) block saving.")
    if warnings:
        st.info(f"{len(warnings)} warning(s).")

    if stored_hash is not None:
        st.caption(
            "⚠️ Saving invalidates progress.jsonl" if invalidates
            else "✅ progress.jsonl stays valid"
        )

    save_col, discard_col = st.columns(2)
    if save_col.button("💾 Save", width="stretch", type="primary",
                       disabled=bool(errors) or not changed):
        stale = pio.externally_modified(draft)
        if stale:
            ed["flash"] = (
                "error",
                f"Changed on disk by something else: {', '.join(stale)}. "
                "Re-open the project to pick those changes up before saving.",
            )
        elif invalidates:
            ed["confirm_save"] = True
        else:
            _do_save()
        st.rerun()

    if discard_col.button("↩️ Discard", width="stretch", disabled=not changed):
        _load(draft.project_dir)
        st.rerun()

    st.caption("This editor never runs a benchmark — use `prompttestenv gui` for that.")

# ── Fixed slots above the tabs ────────────────────────────────────────────────
# Each of these three blocks holds a VARIABLE number of elements: the flash
# message comes and goes, the confirmation appears only mid-save, and the
# validation output grows and shrinks as you type. Rendered directly, that
# changes how many elements sit before st.tabs — which changes the tabs' own
# identity in the element tree, so Streamlit rebuilds them and the selection
# snaps back to the first tab. Editing the evaluation template did exactly that,
# because its validation flips between error, warning and neither.
#
# An unconditional st.container() is ONE element however much (or little) goes
# inside it, so the tabs' position is constant. Do not unwrap these.

flash_slot = st.container()
confirm_slot = st.container()
issues_slot = st.container()

with flash_slot:
    flash = ed["flash"]
    if flash:
        level, text = flash
        getattr(st, level)(text)
        ed["flash"] = None

# An inline bordered block rather than st.dialog: the requirement is an explicit
# confirmation, not a modal, and AppTest can drive (and assert on) this one.
if ed["confirm_save"]:
    with confirm_slot.container(border=True):
        st.warning(
            f"This save changes the config hash "
            f"(`{stored_hash[:8]}…` → `{would_be_hash[:8]}…`).\n\n"
            "The existing **progress.jsonl** will no longer match. The next run "
            "renames it to `progress.jsonl.bak` and starts over unless you pass "
            "`--force-restart`. **Nothing is deleted right now.**"
        )
        reformat_only = _semantically_identical()
        if reformat_only:
            st.info(
                "Reformatting only in: " + ", ".join(reformat_only) +
                " — the content is unchanged, but these files are hashed "
                "byte-for-byte, so the hash changes anyway."
            )
        yes, no = st.columns(2)
        if yes.button("Save anyway", type="primary", key="confirm_yes"):
            _do_save()
            st.rerun()
        if no.button("Cancel", key="confirm_no"):
            ed["confirm_save"] = False
            st.rerun()

with issues_slot:
    for message in errors:
        st.error(message)
    for message in warnings:
        st.warning(message)

tab_cand, tab_tests, tab_judge, tab_global, tab_prompts, tab_files = st.tabs(
    ["Candidates", "Test cases", "Judge config", "Global criteria",
     "System prompts", "Test files"]
)


# ── Candidates ────────────────────────────────────────────────────────────────

with tab_cand:
    if st.button("➕ Add candidate", key="add_cand"):
        ed["candidates"].append(
            {"uid": uuid4().hex,
             "data": {"name": "", "provider": "google", "model": ""}}
        )
        _sync_draft()
        st.rerun()

    if ed["candidates"]:
        st.dataframe(
            [{"name": c["data"].get("name"), "provider": c["data"].get("provider"),
              "model": c["data"].get("model"),
              "system prompt": c["data"].get("system_prompt_file") or "—"}
             for c in ed["candidates"]],
            width="stretch", hide_index=True,
        )

    prompt_names = draft.system_prompt_names()

    for index, row in enumerate(ed["candidates"]):
        data = row["data"]
        title = f"{index + 1}. {data.get('name') or '(unnamed)'} — " \
                f"{data.get('provider', '')}/{data.get('model') or '?'}"
        with st.expander(title, expanded=len(ed["candidates"]) <= 3):
            st.text_input("Name", value=data.get("name") or "",
                          key=_wkey("cand", row["uid"], "name"))
            left, right = st.columns(2)
            with left:
                _open_choice("Provider", pio.PROVIDERS, data.get("provider", "google"),
                             _wkey("cand", row["uid"], "provider"))
            with right:
                st.text_input("Model", value=data.get("model") or "",
                              key=_wkey("cand", row["uid"], "model"))

            left, mid, right = st.columns(3)
            with left:
                st.number_input(
                    "Temperature", min_value=0.0, max_value=2.0, step=0.05,
                    value=float(data.get("temperature", 0.7)),
                    key=_wkey("cand", row["uid"], "temperature"),
                )
            with mid:
                thinking = data.get("thinking", "default")
                thinking_str = str(thinking).lower() if isinstance(thinking, bool) else thinking
                if thinking_str in pio.THINKING_CHOICES:
                    st.selectbox(
                        "Thinking", pio.THINKING_CHOICES,
                        index=pio.THINKING_CHOICES.index(thinking_str),
                        key=_wkey("cand", row["uid"], "thinking"),
                    )
                else:
                    # Kept, not silently rewritten: it reaches the provider verbatim.
                    st.text_input("Thinking (non-standard)", value=str(thinking),
                                  key=_wkey("cand", row["uid"], "thinking"))
            with right:
                st.checkbox("Disable safety filters",
                            value=bool(data.get("disable_safety", False)),
                            key=_wkey("cand", row["uid"], "disable_safety"),
                            help="Google Gemini only.")

            current_prompt = data.get("system_prompt_file")
            options = ["(none)"] + prompt_names
            if current_prompt and current_prompt not in prompt_names:
                options.append(current_prompt)
            picked_prompt = st.selectbox(
                "System prompt file", options,
                index=options.index(current_prompt) if current_prompt in options else 0,
                key=f"{_wkey('cand', row['uid'], 'system_prompt_file')}:pick",
                help="A bare filename, resolved under system_prompts/.",
            )
            st.session_state[_wkey("cand", row["uid"], "system_prompt_file")] = (
                None if picked_prompt == "(none)" else picked_prompt
            )

            extras = [k for k in data if k not in _CANDIDATE_FIELDS]
            if extras:
                st.caption("Preserved extra keys: " + ", ".join(extras))

            up, down, delete, _ = st.columns([1, 1, 1, 7])
            if up.button("▲", key=f"cup:{row['uid']}", disabled=index == 0):
                ed["candidates"][index - 1], ed["candidates"][index] = \
                    ed["candidates"][index], ed["candidates"][index - 1]
                _sync_draft()
                st.rerun()
            if down.button("▼", key=f"cdn:{row['uid']}",
                           disabled=index == len(ed["candidates"]) - 1):
                ed["candidates"][index + 1], ed["candidates"][index] = \
                    ed["candidates"][index], ed["candidates"][index + 1]
                _sync_draft()
                st.rerun()
            if delete.button("🗑", key=f"crm:{row['uid']}"):
                ed["candidates"].pop(index)
                _sync_draft()
                st.rerun()


# ── Test cases ────────────────────────────────────────────────────────────────

def _attachments_of(data: dict) -> list[str]:
    """A test case's attachments, tolerating a `file` the run would reject.

    Rendering must not crash on a hand-written file: pio.validate() is what
    reports the malformed value, as an error that blocks the save.
    """
    try:
        return pio.attachment_paths(data.get("file"))
    except ValueError:
        return []


with tab_tests:
    if st.button("➕ Add test case", key="add_test"):
        ed["tests"].append(
            {"uid": uuid4().hex,
             "data": {"id": "", "prompt": "", "criteria": ""}}
        )
        _sync_draft()
        st.rerun()

    all_groups = sorted({str(t["data"].get("group", "Default group")) for t in ed["tests"]})
    filter_col, group_col = st.columns([1, 2])
    needle = filter_col.text_input("Filter by id or prompt", key="test_filter")
    chosen_groups = group_col.multiselect("Groups", all_groups, key="test_groups")

    if ed["tests"]:
        st.dataframe(
            [{"id": t["data"].get("id"), "group": t["data"].get("group", "Default group"),
              "judge": t["data"].get("judge_type", "llm-judge"),
              "attachments": ", ".join(_attachments_of(t["data"])) or "—"}
             for t in ed["tests"]],
            width="stretch", hide_index=True,
        )

    attachments = draft.test_file_names()
    embedding_model = (draft.judge.get("similarity_judge") or {}).get("model", "bge-m3")

    for index, row in enumerate(ed["tests"]):
        data = row["data"]
        group = str(data.get("group", "Default group"))
        if chosen_groups and group not in chosen_groups:
            continue
        if needle and needle.lower() not in (
            f"{data.get('id', '')} {data.get('prompt', '')}".lower()
        ):
            continue

        judge_type = data.get("judge_type", "llm-judge")
        with st.expander(f"{data.get('id') or '(no id)'} · [{group}] · {judge_type}"):
            left, right = st.columns(2)
            with left:
                st.text_input("ID", value=data.get("id") or "",
                              key=_wkey("test", row["uid"], "id"))
            with right:
                _open_choice("Group", all_groups or ["Default group"], group,
                             _wkey("test", row["uid"], "group"))

            st.radio("Judge type", JUDGE_TYPES,
                     index=JUDGE_TYPES.index(judge_type) if judge_type in JUDGE_TYPES else 0,
                     horizontal=True, key=_wkey("test", row["uid"], "judge_type"))

            st.text_area("Prompt", value=data.get("prompt") or "", height=180,
                         key=_wkey("test", row["uid"], "prompt"))

            criteria = data.get("criteria") or ""
            if judge_type == "similarity":
                label = "Reference text (embedded and compared against the response)"
                help_text = f"Embedded with {embedding_model}, scaled to 1-10."
            elif judge_type == "assert":
                label = "Assert lambda body"
                help_text = "Evaluated unsandboxed at run time. Never executed here."
            else:
                label = "Criteria (given to the judge, not to the candidate)"
                help_text = None
            st.text_area(label, value=criteria, height=160, help=help_text,
                         key=_wkey("test", row["uid"], "criteria"))
            if judge_type == "assert":
                _assert_panel(criteria)

            current_files = _attachments_of(data)
            options = [f"{pio.TEST_FILES_DIR}/{n}" for n in attachments]
            # A referenced-but-absent path stays selectable, so it remains
            # visible in the UI instead of silently dropping out of the file.
            options += [p for p in current_files if p not in options]
            picked_files = st.multiselect(
                "Attachments", options, default=current_files,
                help="Sent to the candidate, and to the judge when "
                     "pass_media_to_judge is on. Order is the order they reach "
                     "the model.",
                key=f"{_wkey('test', row['uid'], 'file')}:pick",
            )
            st.session_state[_wkey("test", row["uid"], "file")] = (
                pio.attachment_value(picked_files)
            )

            extras = [k for k in data if k not in _TEST_FIELDS]
            if extras:
                st.caption("Preserved extra keys: " + ", ".join(extras))

            up, down, delete, _ = st.columns([1, 1, 1, 7])
            if up.button("▲", key=f"tup:{row['uid']}", disabled=index == 0):
                ed["tests"][index - 1], ed["tests"][index] = \
                    ed["tests"][index], ed["tests"][index - 1]
                _sync_draft()
                st.rerun()
            if down.button("▼", key=f"tdn:{row['uid']}",
                           disabled=index == len(ed["tests"]) - 1):
                ed["tests"][index + 1], ed["tests"][index] = \
                    ed["tests"][index], ed["tests"][index + 1]
                _sync_draft()
                st.rerun()
            if delete.button("🗑", key=f"trm:{row['uid']}"):
                ed["tests"].pop(index)
                _sync_draft()
                st.rerun()


# ── Judge config ──────────────────────────────────────────────────────────────

def _block(name: str) -> dict:
    """Read one judge block. Never creates it — see _harvest's block rule."""
    existing = draft.judge.get(name)
    return existing if isinstance(existing, dict) else {}


def _field(block: dict, name: str, cls: type):
    """A block field's current value, falling back to its REAL default.

    Rendering a hardcoded "" for an absent key would make the harvest see a
    change the user never made, and write the block out.
    """
    return block.get(name, pio.effective_default(cls, name))


def _thinking_widget(label_prefix: str, current, key: str) -> None:
    """Render `thinking` as a closed choice, or as text if the file disagrees.

    api.py normalises exactly "true"/"false"/"default" (booleans included) and
    passes anything else through verbatim, so an out-of-vocabulary value is
    shown as-is rather than silently rewritten into one of the three.
    """
    current_str = str(current).lower() if isinstance(current, bool) else current
    if current_str in pio.THINKING_CHOICES:
        st.selectbox(f"{label_prefix}Thinking", pio.THINKING_CHOICES,
                     index=pio.THINKING_CHOICES.index(current_str), key=key)
    else:
        st.text_input(f"{label_prefix}Thinking (non-standard)", value=str(current), key=key)


def _judge_common(block: dict, prefix: str, cls: type, *, safety: bool) -> None:
    """Render one judge block's shared fields. Renders only — see _harvest."""
    left, right = st.columns(2)
    with left:
        _open_choice("Provider", pio.PROVIDERS, _field(block, "provider", cls),
                     f"{prefix}:provider")
    with right:
        st.text_input("Model", value=_field(block, "model", cls), key=f"{prefix}:model")
    left, mid, right = st.columns(3)
    with left:
        st.number_input("Temperature", min_value=0.0, max_value=2.0, step=0.05,
                        value=float(_field(block, "temperature", cls)),
                        key=f"{prefix}:temperature")
    with mid:
        _thinking_widget("", _field(block, "thinking", cls), f"{prefix}:thinking")
    if safety:
        with right:
            st.checkbox("Disable safety filters",
                        value=bool(_field(block, "disable_safety", cls)),
                        key=f"{prefix}:safety")


with tab_judge:
    with st.expander("General", expanded=True):
        left, right = st.columns(2)
        with left:
            st.number_input("Repetitions", min_value=1, step=1,
                            value=int(draft.judge.get("repetitions", 5)),
                            key="jc:repetitions")
            st.number_input("Delay between repetitions (s)", min_value=0.0, step=0.5,
                            value=float(draft.judge.get("repetition_delay_seconds", 2)),
                            key="jc:rep_delay")
        with right:
            st.number_input("Response timeout (s)", min_value=1.0, step=10.0,
                            value=float(draft.judge.get("max_response_timeout_seconds", 300)),
                            key="jc:timeout")
            st.number_input("Delay between evaluations (s)", min_value=0.0, step=0.5,
                            value=float(draft.judge.get("evaluation_delay_seconds", 2)),
                            key="jc:eval_delay")

        st.checkbox("Pass attachments to the judge too",
                    value=bool(draft.judge.get("pass_media_to_judge", False)), key="jc:media")
        st.checkbox("One verdict per group (plus a global one)",
                    value=bool(draft.judge.get("group_verdicts", False)), key="jc:groups")

        scope = draft.judge.get("reasoning_analysis", "none")
        scope = scope if scope in REASONING_SCOPES else "none"
        st.radio(
            "Reasoning analysis scope", REASONING_SCOPES,
            index=REASONING_SCOPES.index(scope), horizontal=True, key="jc:scope",
            help="'best' analyses only the highest-scoring repetition of each "
                 "candidate x test, costing `repetitions` times less than 'all'.",
        )
        st.caption(
            "🔓 Excluded from the run hash — changing this never invalidates progress.jsonl."
        )

    with st.expander("Test judge"):
        block = _block("test_judge")
        _judge_common(block, "tj", TestJudgeSettings, safety=True)
        st.text_area("System prompt", value=_field(block, "evaluation_system_prompt", TestJudgeSettings),
                     height=200, key="tj:sys")
        template = _field(block, "evaluation_template", TestJudgeSettings)
        st.text_area("Evaluation template", value=template, height=280, key="tj:template")

        st.caption("This is the only template still passed through `str.format()`:")
        for placeholder in pio.EVALUATION_PLACEHOLDERS:
            token = "{" + placeholder + "}"
            st.caption(("✅ " if token in template else "⚠️ ") + token)
        error, warning = pio.check_evaluation_template(template)
        if error:
            st.error(f"Template {error}")
        elif warning:
            st.warning(f"Template {warning}")
        else:
            st.success("Template formats correctly.")

    with st.expander("Similarity judge"):
        block = _block("similarity_judge")
        left, right = st.columns(2)
        with left:
            _open_choice("Provider", pio.PROVIDERS, _field(block, "provider", SimilarityJudgeSettings),
                         "sj:provider")
        with right:
            st.text_input("Embedding model", value=_field(block, "model", SimilarityJudgeSettings),
                          key="sj:model", help="An embedding model, not a chat model.")
        st.caption(
            "This block has exactly two keys. Anything else is dropped at load "
            "(`models._settings_from_dict`), though the editor still preserves it on disk."
        )

    with st.expander("Verdict judge"):
        block = _block("verdict_judge")
        _judge_common(block, "vj", VerdictJudgeSettings, safety=True)
        st.text_area("System prompt", value=_field(block, "verdict_system_prompt", VerdictJudgeSettings),
                     height=200, key="vj:sys")
        st.text_area("Verdict template", value=_field(block, "verdict_template", VerdictJudgeSettings),
                     height=300, key="vj:template")
        if not draft.judge.get("group_verdicts"):
            st.info("Group verdicts are off, so the template below is unused.")
        st.text_area("Global verdict template",
                     value=_field(block, "global_verdict_template", VerdictJudgeSettings),
                     height=300, key="vj:global")
        st.caption(
            "Both templates are appended verbatim after the data — braces are "
            "literal and there are no `{placeholders}` to fill in."
        )

    with st.expander("Reasoning judge"):
        st.info(
            "🔓 This whole block, plus the scope above, is stripped before hashing. "
            "Editing it never invalidates progress.jsonl."
        )
        block = _block("reasoning_judge")
        _judge_common(block, "rj", ReasoningJudgeSettings, safety=False)  # this block has no disable_safety

        defaults = get_app_config().reasoning_defaults
        left, right = st.columns(2)
        with left:
            current_mode = block.get("dimension_mode")
            options = [_NULLABLE_SENTINEL, *DIMENSION_MODES]
            picked = st.selectbox(
                "Dimension mode", options,
                index=options.index(current_mode) if current_mode in options else 0,
                key="rj:dimmode",
                help="'split' asks one single-concept question per dimension, "
                     "which a small local judge can actually answer.",
            )
            if "dimension_mode" in block or picked != _NULLABLE_SENTINEL:
                block["dimension_mode"] = None if picked == _NULLABLE_SENTINEL else picked

            _nullable_number(
                "context size", block.get("context_size"), "rj:ctx",
                minimum=256, step=1024, fallback="the server's own",
                help_text="Ollama only. Unset (or 0) means the server default.",
            )
        with right:
            _nullable_number(
                "reliability k", block.get("reliability_k"), "rj:k",
                minimum=1, step=1, fallback=defaults.reliability_k,
                help_text="How many times each scoring call is repeated.",
            )
            _nullable_number(
                "max units per call", block.get("max_units_per_call"), "rj:units",
                minimum=1, step=10, fallback=defaults.max_units_per_call,
                help_text="Trace units sent to the judge in one request.",
            )


# ── Global criteria ───────────────────────────────────────────────────────────

_CRITERIA_FIELDS = {
    "llm-judge": ("llm_judge_criteria", "Rubric handed to the judge"),
    "similarity": ("similarity_criteria", "Reference text to embed against"),
    "assert": ("assert_criteria", "Assert lambda body"),
}

with tab_global:
    mode = draft.criteria.get("mode", "llm-judge")
    active = mode if mode in GLOBAL_MODES else "llm-judge"
    st.radio(
        "Mode", GLOBAL_MODES, index=GLOBAL_MODES.index(active), horizontal=True, key="gc:mode",
        help="Scored on every response, independently of the per-test criteria.",
    )

    if active == "none":
        st.info("Global scoring is disabled. The criteria below are kept in the file.")
    else:
        field_name, label = _CRITERIA_FIELDS[active]
        criteria_text = draft.criteria.get(field_name, "")
        st.text_area(label, value=criteria_text, height=200, key=f"gc:{field_name}")
        if active == "assert":
            _assert_panel(criteria_text)

    # All three are rendered whatever the mode: the schema is flat and every
    # field is always present on disk, so hiding the inactive ones would only
    # invite deleting them by accident.
    with st.expander("Criteria for the other modes (kept in the file)"):
        for other, (field_name, label) in _CRITERIA_FIELDS.items():
            if other == active:
                continue
            st.text_area(f"{label} — {other}", value=draft.criteria.get(field_name, ""),
                         height=140, key=f"gc:other:{field_name}")


# ── System prompts and test files ─────────────────────────────────────────────

_NOT_HASHED_WARNING = (
    "⚠️ **These files are not part of the run hash.** Only the four JSON configs "
    "are. Editing anything here does *not* invalidate `progress.jsonl`, so a "
    "resumed run will mix responses produced under the old and the new version "
    "into one report, with nothing flagging it. Use `--force-restart` if that matters."
)

with tab_prompts:
    st.warning(_NOT_HASHED_WARNING)
    names = draft.system_prompt_names()
    choice = st.selectbox("File", ["➕ New file…", *names], key="sp:pick")

    if choice == "➕ New file…":
        new_name = st.text_input("Filename", value="", placeholder="my_prompt.txt",
                                 key="sp:newname")
        content = st.text_area("Content", value="", height=400, key="sp:newbody")
        if st.button("Create", key="sp:create"):
            problem = pio.check_filename(new_name)
            if problem:
                st.error(problem)
            elif not new_name.endswith(".txt"):
                st.error("System prompt files must end in .txt")
            else:
                draft.system_prompts_dir.mkdir(parents=True, exist_ok=True)
                (draft.system_prompts_dir / new_name).write_text(
                    content, encoding="utf-8", newline="\n")
                st.rerun()
    else:
        path = draft.system_prompts_dir / choice
        body = st.text_area("Content", value=path.read_text(encoding="utf-8"),
                            height=400, key=f"sp:body:{choice}")
        users = [c["data"].get("name") for c in ed["candidates"]
                 if c["data"].get("system_prompt_file") == choice]
        st.caption("Used by: " + (", ".join(str(u) for u in users) if users else "nothing"))

        save_col, delete_col = st.columns(2)
        if save_col.button("💾 Save prompt", key="sp:save"):
            path.write_text(body, encoding="utf-8", newline="\n")
            st.success(f"Saved {choice}")
        if delete_col.button("🗑 Delete", key="sp:delete"):
            if users:
                st.error(f"Still used by {', '.join(str(u) for u in users)}.")
            else:
                path.unlink()
                st.rerun()

with tab_files:
    st.warning(_NOT_HASHED_WARNING)
    uploaded = st.file_uploader("Add attachments", accept_multiple_files=True, key="tf:upload")
    if uploaded:
        draft.test_files_dir.mkdir(parents=True, exist_ok=True)
        for item in uploaded:
            safe = Path(item.name).name
            problem = pio.check_filename(safe)
            if problem:
                st.error(f"{item.name}: {problem}")
                continue
            (draft.test_files_dir / safe).write_bytes(item.getvalue())
            st.success(f"Stored {safe}")

    for name in draft.test_file_names():
        path = draft.test_files_dir / name
        users = [t["data"].get("id") for t in ed["tests"]
                 if f"{pio.TEST_FILES_DIR}/{name}" in _attachments_of(t["data"])]
        left, right = st.columns([5, 1])
        left.write(
            f"**{name}** — {path.stat().st_size:,} bytes · used by: "
            + (", ".join(str(u) for u in users) if users else "nothing")
        )
        if right.button("🗑", key=f"tf:rm:{name}"):
            if users:
                st.error(f"'{name}' is still used by {', '.join(str(u) for u in users)}.")
            else:
                path.unlink()
                st.rerun()
