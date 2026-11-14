"""What the AI client can reach, and what a project asks of it.

Two halves, both answered by UnifiedAiClient rather than by tables kept here:

- ``list_catalog`` reports the providers usable right now and the models one of
  them offers, as JSON-safe dicts for the CLI, the MCP server and the editor.
  It touches the network (local servers are probed, model listings are live),
  so it is never called by ``run``.
- ``offline_findings`` checks a loaded project against the capabilities each
  provider declares in the client's own registry, with no network at all, so
  ``run`` can afford it before the first call. Errors are about the judges,
  whose failure would degrade every score; warnings are about candidates,
  whose unsupported tests are skipped and reported rather than blocking.
  ``online_findings`` is its counterpart against live listings the caller
  already holds (the editor's), and only ever warns.

No Streamlit import, so the editor can call this and the suite can test it alone.
"""
from __future__ import annotations

import os

from prompttestenv.api import (
    ModelInfo,
    ProviderInfo,
    UnsupportedFileError,
    call_with_timeout,
    get_provider,
    list_models,
    list_providers,
    validate_files,
)
from prompttestenv.models import (
    JUDGE_TYPE_ASSERT,
    JUDGE_TYPE_LLM,
    JUDGE_TYPE_SIMILARITY,
    Candidate,
    JudgeConfig,
    TestCase,
)

# Bound on one listing, since a local server that accepts the connection and
# never answers would otherwise hang the editor or the CLI.
CATALOG_TIMEOUT_SECONDS = 15.0

# The one provider the client never checks attachments for: its empty
# SUPPORTED_FILE_TYPES would refuse files call_ai() hands to the script as is.
_UNCHECKED_PROVIDER = "script"

# TEMPORARY, by name rather than by capability: the Messages API has no JSON
# mode without a schema, and both the test judge and the reasoning judge ask
# for JSON without one, so UnifiedAiClient refuses every such call. The same
# holds for a candidate on a test with json_output, which is only a warning
# (see offline_findings). Remove this together with that limitation (passing
# each judge's schema through extra_options), not before.
_NO_SCHEMALESS_JSON = frozenset({"anthropic"})


def _provider_dict(info: ProviderInfo) -> dict:
    return {
        "name": info.name,
        "local": info.local,
        "requires_api_key": info.requires_api_key,
        "file_types": sorted(info.file_types),
        "supports_unload": info.supports_unload,
        "supports_embeddings": info.supports_embeddings,
    }


def _model_dict(info: ModelInfo) -> dict:
    # None stays None: it means "the provider does not say", never "no".
    return {
        "id": info.id,
        "display_name": info.display_name,
        "reasoning": info.reasoning,
        "tool_use": info.tool_use,
        "file_types": None if info.file_types is None else sorted(info.file_types),
        "embeddings": info.embeddings,
        "context_length": info.context_length,
    }


def list_catalog(
    provider: str | None = None,
    *,
    probe_local: bool = True,
    timeout: float = CATALOG_TIMEOUT_SECONDS,
) -> dict:
    """List the usable providers, or the models one provider offers.

    Never raises: a failure is reported in the ``error`` field, next to an empty
    list, so a caller can render what it has.

    Args:
        provider: A provider name to list the models of. None lists providers.
        probe_local: Whether the provider listing asks the local servers
            (ollama, lmstudio, llamacpp) if they are running. Without asking
            they are left out. Ignored when `provider` is given.
        timeout: Seconds to wait for the whole listing.

    Returns:
        ``{"providers": [...], "error": str | None}`` without `provider`, in the
        client's registry order; otherwise ``{"provider": name, "models": [...],
        "error": str | None}``, sorted by id. A provider that cannot enumerate
        its models (``script``) gets an empty list and no error.
    """
    if provider is None:
        try:
            infos, timed_out = call_with_timeout(
                list_providers, fn_kwargs={"probe_local": probe_local}, timeout=timeout,
            )
        except Exception as exc:
            return {"providers": [], "error": f"{type(exc).__name__}: {exc}"}
        if timed_out:
            return {"providers": [], "error": f"Provider listing timed out after {timeout:g}s."}
        return {"providers": [_provider_dict(i) for i in infos], "error": None}

    name = provider.strip().lower()
    out: dict = {"provider": name, "models": [], "error": None}
    try:
        infos, timed_out = call_with_timeout(
            list_models, name, fn_kwargs={"timeout": max(1, int(timeout))}, timeout=timeout,
        )
    except NotImplementedError:
        return out
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if timed_out:
        out["error"] = f"Model listing for '{name}' timed out after {timeout:g}s."
        return out
    out["models"] = sorted((_model_dict(i) for i in infos), key=lambda m: m["id"])
    return out


def _registered(provider: str):
    """The client's provider instance, or None for a name it does not know."""
    try:
        return get_provider(provider.strip().lower())
    except ValueError:
        return None


def _rejection(path: str, provider_name: str, instance) -> str | None:
    """Why `instance` cannot take the file at `path`, or None if it can."""
    try:
        validate_files(
            [path], provider_name, instance.SUPPORTED_FILE_TYPES,
            extensions=instance.SUPPORTED_EXTENSIONS,
        )
    except UnsupportedFileError as exc:
        return str(exc)
    except FileNotFoundError:
        # Not this check's business: runner._missing_attachments reports it.
        return None
    return None


def _uses_similarity(expression: str) -> bool:
    """Whether an assert lambda calls the similarity() helper."""
    return "similarity(" in (expression or "")


def _judges_in_use(judge_config: JudgeConfig, test_cases: list[TestCase],
                   verdict: bool) -> list[tuple[str, object]]:
    """``(role, settings)`` for each judge this project will actually call.

    The test judge when some score is an llm-judge one, the similarity judge
    when something embeds, the reasoning judge when the analysis is enabled,
    the verdict judge when `verdict` says a verdict will be written.
    """
    gc = judge_config.global_criteria
    global_mode = getattr(gc, "mode", None)
    judge_types = {t.judge_type for t in test_cases}

    uses_test_judge = JUDGE_TYPE_LLM in judge_types or global_mode == JUDGE_TYPE_LLM
    uses_embeddings = (
        JUDGE_TYPE_SIMILARITY in judge_types
        or global_mode == JUDGE_TYPE_SIMILARITY
        or judge_config.reasoning_enabled
        or any(t.judge_type == JUDGE_TYPE_ASSERT and _uses_similarity(t.criteria)
               for t in test_cases)
        or (global_mode == JUDGE_TYPE_ASSERT
            and _uses_similarity(getattr(gc, "assert_criteria", "")))
    )

    judges: list[tuple[str, object]] = []
    if uses_test_judge:
        judges.append(("test_judge", judge_config.test_judge))
    if uses_embeddings:
        judges.append(("similarity_judge", judge_config.similarity_judge))
    if judge_config.reasoning_enabled:
        judges.append(("reasoning_judge", judge_config.reasoning_judge))
    if verdict:
        judges.append(("verdict_judge", judge_config.verdict_judge))
    return judges


def offline_findings(
    candidates: list[Candidate],
    judge_config: JudgeConfig,
    test_cases: list[TestCase],
    project_dir: str,
    *,
    verdict: bool = True,
) -> tuple[list[str], list[str]]:
    """Check a project against what each provider declares, without the network.

    Only judges that this project will actually call are checked (see
    ``_judges_in_use``).

    Args:
        candidates: The loaded candidates.
        judge_config: The loaded JudgeConfig, with ``global_criteria`` set.
        test_cases: The loaded test cases.
        project_dir: The project directory attachment paths are relative to.
        verdict: Whether this run will call the verdict judge.

    Returns:
        ``(errors, warnings)``. An error means a judge would fail on every call
        it makes, so the run should not start. A warning names a candidate x
        test pair that will be skipped, or a candidate provider nobody knows.
    """
    errors: list[str] = []
    warnings: list[str] = []
    global_mode = getattr(judge_config.global_criteria, "mode", None)

    instances = {}
    for role, settings in _judges_in_use(judge_config, test_cases, verdict):
        provider = settings.provider
        instance = _registered(provider)
        if instance is None:
            errors.append(f"{role}: unknown provider '{provider}'.")
            continue
        instances[role] = instance
        name = provider.strip().lower()
        if role in ("test_judge", "reasoning_judge") and name in _NO_SCHEMALESS_JSON:
            errors.append(
                f"{role}: provider '{name}' cannot return JSON without a schema yet, "
                f"so every call it makes would fail and score -1."
            )
        if role == "similarity_judge" and not instance.SUPPORTS_EMBEDDINGS:
            errors.append(f"similarity_judge: provider '{name}' has no embeddings.")

    test_judge = instances.get("test_judge")
    tj_name = judge_config.test_judge.provider.strip().lower()
    if (judge_config.pass_media_to_judge and test_judge is not None
            and tj_name != _UNCHECKED_PROVIDER):
        for test in test_cases:
            if not (test.judge_type == JUDGE_TYPE_LLM or global_mode == JUDGE_TYPE_LLM):
                continue
            for path in test.attachments():
                reason = _rejection(os.path.join(project_dir, path), tj_name, test_judge)
                if reason:
                    errors.append(
                        f"test_judge cannot take test '{test.id}' attachment "
                        f"'{path}' and pass_media_to_judge is on: {reason}"
                    )

    for cand in candidates:
        name = (cand.provider or "").strip().lower()
        instance = _registered(name)
        if instance is None:
            warnings.append(
                f"Candidate '{cand.name}': unknown provider '{cand.provider}', "
                f"so every generation will fail."
            )
            continue
        if name in _NO_SCHEMALESS_JSON:
            json_tests = [t.id for t in test_cases if t.json_output]
            if json_tests:
                warnings.append(
                    f"Candidate '{cand.name}': provider '{name}' cannot return JSON "
                    f"without a schema yet, so test(s) {', '.join(json_tests)} "
                    f"(json_output) will fail with a generation error."
                )
        if name == _UNCHECKED_PROVIDER:
            continue
        for test in test_cases:
            for path in test.attachments():
                reason = _rejection(os.path.join(project_dir, path), name, instance)
                if reason:
                    warnings.append(
                        f"Candidate '{cand.name}' will skip test '{test.id}': {reason}"
                    )
                    break
    return errors, warnings


def _listed(models: list[dict], model_id: str | None) -> dict | None:
    """The listing's entry for `model_id`. An untagged Ollama name means ':latest'."""
    wanted = (model_id or "").strip()
    for info in models:
        if info["id"] in (wanted, f"{wanted}:latest"):
            return info
    return None


def _wants_thinking(value) -> bool:
    """Whether a `thinking` setting asks for reasoning (api.py's reading of it)."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() not in ("", "false", "default")


def online_findings(
    candidates: list[Candidate],
    judge_config: JudgeConfig,
    test_cases: list[TestCase],
    listings: dict[str, dict],
    *,
    verdict: bool = True,
) -> list[str]:
    """Check a project against live model listings. Warnings only, never errors.

    Pure: the listings are fetched by the caller (the editor caches them), so
    this makes no request. Unknown stays unknown: a capability the listing
    reports as None is never treated as missing.

    Args:
        candidates: The candidates, as Candidate.from_dict builds them.
        judge_config: The JudgeConfig, with ``global_criteria`` set.
        test_cases: The test cases, used to decide which judges are called.
        listings: Provider name (lower case) -> a ``list_catalog(provider)``
            result. A provider that is absent, failed, or listed no models is
            not checked, since there is nothing to compare against.
        verdict: Whether the verdict judge will be called.

    Returns:
        One message per finding: a model the provider does not list, a chat
        model used for embeddings or the reverse, thinking asked of a model
        that does not reason, a context size beyond the model's.
    """
    uses = [(f"Candidate '{c.name}'", c.provider, c.model, False, c.thinking, None)
            for c in candidates]
    for role, settings in _judges_in_use(judge_config, test_cases, verdict):
        uses.append((role, settings.provider, settings.model, role == "similarity_judge",
                     getattr(settings, "thinking", False),
                     getattr(settings, "context_size", None)))

    findings: list[str] = []
    for who, provider, model, embeds, thinking, context_size in uses:
        name = (provider or "").strip().lower()
        listing = listings.get(name) or {}
        if listing.get("error") or not listing.get("models"):
            continue
        info = _listed(listing["models"], model)
        if info is None:
            findings.append(f"{who}: '{name}' lists no model '{model}'.")
            continue
        if embeds and info["embeddings"] is False:
            findings.append(f"{who}: '{model}' is not an embedding model.")
        if not embeds and info["embeddings"] is True:
            findings.append(f"{who}: '{model}' is an embedding model, not a chat model.")
        if not embeds and _wants_thinking(thinking) and info["reasoning"] is False:
            findings.append(f"{who}: thinking is on, but '{model}' does not reason.")
        limit = info["context_length"]
        if context_size and limit and context_size > limit:
            findings.append(
                f"{who}: context_size {context_size} exceeds the {limit} '{model}' supports."
            )
    return findings
