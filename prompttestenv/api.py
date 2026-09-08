from __future__ import annotations

import concurrent.futures
import math
from typing import Any, Callable

from unified_ai_client import (
    call_ai,
    preload_model,
    unload_model,
    warm_up,
    cleanup,
    get_embedding,
    set_verbosity,
)

import prompttestenv.logger as logger
from prompttestenv.config import get_app_config
from prompttestenv.models import LlmResult

# Added to max_response_timeout_seconds to derive the provider-level network
# timeout: it must expire slightly AFTER the outer call_with_timeout deadline,
# never before it (which would trigger call_ai's own retries prematurely) and
# never much later (which would leave an abandoned thread hanging around long
# after PromptTestEnv itself has given up).
PROVIDER_TIMEOUT_BUFFER_SECONDS = 5

# Ollama's keep_alive is the idle timer that starts when a request FINISHES,
# not a budget for the request itself: a model serving a call is never evicted
# mid-wait. Deriving it from max_response_timeout_seconds conflated two
# unrelated things. A benchmark wants residency decided by the run, so a local
# model is pinned for as long as it is being measured and released explicitly
# at the candidate boundary (and by UnifiedAiClient's cleanup() at exit).
LOCAL_KEEP_ALIVE = -1

# Level handed to unified_ai_client.set_verbosity when config.json's
# logging.level is missing or unrecognised.
DEFAULT_AI_LOG_LEVEL = "warning"

# The local model a candidate is known to have loaded, as (provider, model).
# Whoever loads one records it here; release_previous_candidate drops it at the
# candidate boundary. That split matters: a candidate whose repetitions are all
# already in the log, or a run with warm-up disabled, must still leave the slot
# telling the truth about what is actually resident.
_resident_candidate: tuple[str, str] | None = None


def configure_ai_logging() -> None:
    """Apply config.json's ``logging.level`` to UnifiedAiClient's logger.

    Called once per process from every entry point (the CLI ``main()``, and each
    Streamlit app, which run as their own processes). An unrecognised level is
    reported through the project logger and falls back to DEFAULT_AI_LOG_LEVEL
    rather than aborting startup.
    """
    level = get_app_config().logging.level
    try:
        set_verbosity(level)
    except ValueError:
        logger.log_warning(
            f"Unknown logging.level {level!r} in config.json; using "
            f"{DEFAULT_AI_LOG_LEVEL!r}. Valid: silent, error, warning, debug."
        )
        set_verbosity(DEFAULT_AI_LOG_LEVEL)


def _buffered_provider_timeout(max_response_timeout_seconds: float) -> int:
    """Derive the provider-level timeout from the outer enforced one.

    Args:
        max_response_timeout_seconds: The judge_config value that
            call_with_timeout already enforces from outside.

    Returns:
        That value plus PROVIDER_TIMEOUT_BUFFER_SECONDS, as a whole number of
        seconds (call_ai's own ``timeout`` parameter is typed ``int``).
    """
    return int(max_response_timeout_seconds) + PROVIDER_TIMEOUT_BUFFER_SECONDS


def is_local_provider(provider: str) -> bool:
    """Report whether a provider runs models on this machine.

    Local backends serve one model at a time: firing concurrent requests at the
    same model queues them, or forces a second load into VRAM, so callers must
    run their calls sequentially instead of in a thread pool. The list is read
    from config.json rather than hardcoded, so a new local backend does not
    require a code change.

    Args:
        provider: Provider name, case-insensitive.

    Returns:
        True if the provider is served locally.
    """
    return provider.lower() in {p.lower() for p in get_app_config().local_providers}


def get_text_embedding(
    provider: str,
    model_name: str,
    text: str,
) -> list[float]:
    """Get high-dimensional embedding vector for text using UnifiedAiClient.

    Args:
        provider: AI provider name (e.g. 'google', 'ollama').
        model_name: Embedding model identifier.
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    normalized_provider = provider.lower()
    return get_embedding(
        provider=normalized_provider,
        model=model_name,
        text=text,
    )


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors.

    Args:
        vec_a: The first vector of floats.
        vec_b: The second vector of floats.

    Returns:
        The cosine similarity score as a float in [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)




def get_llm_response(
    provider: str,
    model_name: str,
    system_instruction: str | None,
    user_prompt: str,
    local_media_paths: list[str] | None = None,
    temp: float = 0.7,
    response_mime_type: str | None = None,
    thinking: bool | str = "default",
    disable_safety: bool = False,
    max_response_timeout_seconds: float | None = None,
    max_retries: int = 5,
) -> LlmResult:
    """Route an LLM generation request to the appropriate provider via UnifiedAiClient.

    File type classification, encoding, upload, caching, and cleanup are handled
    entirely by UnifiedAiClient. Supported types include images, audio, text files,
    and PDFs, across all providers.

    Args:
        provider: LLM provider name ('google' or 'ollama').
        model_name: Model identifier string.
        system_instruction: Optional system prompt.
        user_prompt: The user-facing prompt text.
        local_media_paths: Optional local file paths for multimodal input, in
            the order they should reach the model. They are passed directly to
            UnifiedAiClient, which classifies and encodes each one according to
            the provider (base64, upload, or text inline). Text attachments are
            inlined AHEAD of user_prompt, in this order.
        temp: Sampling temperature.
        response_mime_type: If 'application/json', enables JSON output mode.
        thinking: Whether to enable thinking/reasoning mode.
        disable_safety: Whether to disable safety settings (Google Gemini only).
        max_response_timeout_seconds: The judge_config value that
            call_with_timeout enforces around this call from outside. When
            given, the underlying network timeout is set to this value plus
            PROVIDER_TIMEOUT_BUFFER_SECONDS, so the provider layer never times
            out before the outer deadline (which would trigger call_ai's own
            retries prematurely) nor leaves the call hanging around long after
            it. When None, call_ai's own default applies. It has no bearing on
            how long a local model stays resident: see LOCAL_KEEP_ALIVE.
        max_retries: How many times call_ai retries a retryable failure before
            giving up. Defaults to 5 for judge/verdict/reasoning calls;
            candidate generation passes 0 so a failure surfaces immediately
            instead of being retried on the measured clock.

    Returns:
        An LlmResult. Its reasoning_text is the thinking transcript when the
        model produced one, and reasoning_is_summary says whether that text is
        the raw chain of thought or a summary the provider wrote about it.
    """
    normalized_provider = provider.lower()

    # Normalize thinking argument
    if isinstance(thinking, str):
        thinking_lower = thinking.lower().strip()
        if thinking_lower == "true":
            thinking = True
        elif thinking_lower == "false":
            thinking = False
        elif thinking_lower == "default":
            thinking = "default"

    extra_options: dict[str, Any] = {}
    if disable_safety:
        extra_options["disable_safety"] = True
    if is_local_provider(normalized_provider):
        # Pinned for the whole run: residency is managed explicitly, at the
        # candidate boundary and by teardown, never by a timer.
        extra_options["keep_alive"] = LOCAL_KEEP_ALIVE
        _note_local_model_resident(normalized_provider, model_name)

    call_kwargs: dict[str, Any] = dict(
        provider=normalized_provider,
        model=model_name,
        prompt=user_prompt,
        system_prompt=system_instruction,
        file_path=local_media_paths or None,
        temperature=temp,
        thinking=thinking,
        format_json=(response_mime_type == "application/json"),
        max_retries=max_retries,
        extra_options=extra_options,
    )
    if max_response_timeout_seconds is not None:
        call_kwargs["timeout"] = _buffered_provider_timeout(max_response_timeout_seconds)

    response = call_ai(**call_kwargs)
    return LlmResult(
        text=response.text,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
        reasoning_text=response.reasoning_text or "",
        reasoning_is_summary=bool(getattr(response, "reasoning_is_summary", False)),
    )


def _note_local_model_resident(provider: str, model_name: str) -> None:
    """Record that a local model has just been pinned into memory.

    Called from wherever a load actually happens, so the slot never claims a
    model that was never loaded: releasing one of those would make Ollama load
    it just to drop it again.

    Args:
        provider: LLM provider name, already known to be local.
        model_name: Model identifier now resident.
    """
    global _resident_candidate

    _resident_candidate = (provider.lower(), model_name)


def release_previous_candidate(provider: str, model_name: str) -> None:
    """Free the VRAM of the candidate that ran before this one.

    Local models are pinned for the whole run (see LOCAL_KEEP_ALIVE), so
    something has to let them go, or every model a benchmark touches would be
    resident at once and the later candidates would be timed while competing
    for VRAM with the earlier ones.

    A candidate using the same model as its predecessor releases nothing:
    candidates differing only by system prompt or temperature are a common
    shape, and they must not pay a reload between them.

    Args:
        provider: Provider of the candidate about to run.
        model_name: Model of the candidate about to run.
    """
    global _resident_candidate

    key = (provider.lower(), model_name) if is_local_provider(provider) else None
    if _resident_candidate is None or _resident_candidate == key:
        return
    unload_model(*_resident_candidate)
    _resident_candidate = None


def warm_up_for_run(
    provider: str,
    model_name: str,
    media_paths: list[str] | None = None,
    *,
    max_response_timeout_seconds: float | None = None,
) -> bool:
    """Pay a provider's one-off costs before a candidate starts being timed.

    Without this, the setup a provider charges once per process — SDK import,
    client construction, DNS + TCP + TLS handshake, model load, and above all
    the upload of every attachment — all lands on whichever call happens to run
    first, making the first candidate of the list look slow purely because it
    went first.

    Use this for anything whose elapsed time is measured. The sibling
    ``preload_model_for_run`` survives for the one thing warm-up cannot do:
    allocate an Ollama model with a specific ``context_size``, which Ollama
    fixes at load time.

    UnifiedAiClient consumes no generation tokens here and never raises, so
    there is nothing to guard against and no provider to exclude: where there
    is nothing to warm up it is a free no-op.

    Releasing the previous candidate is a separate step,
    ``release_previous_candidate``: this one is skipped when warm-up is disabled
    or when the candidate has nothing left to generate, and the boundary has to
    hold in both of those cases too.

    Args:
        provider: LLM provider name.
        model_name: Model identifier to warm the connection with.
        media_paths: Attachments to upload ahead of time. Only providers with a
            remote file store (currently Google) act on them.
        max_response_timeout_seconds: The judge_config value this candidate's
            calls are bounded by. Used as the load's own socket timeout, so a
            project configured for a slow local model gets a proportionally
            patient warm-up instead of the client's default.

    Returns:
        True if the provider actually warmed something up.
    """
    local = is_local_provider(provider)
    if local:
        _note_local_model_resident(provider, model_name)

    timeout = (
        _buffered_provider_timeout(max_response_timeout_seconds)
        if max_response_timeout_seconds is not None
        else None
    )
    return warm_up(
        provider=provider,
        model=model_name,
        file_paths=media_paths,
        keep_alive=LOCAL_KEEP_ALIVE if local else None,
        timeout=timeout,
    )


def preload_model_for_run(
    provider: str,
    model_name: str,
    context_size: int | None = None,
) -> None:
    """Preload an Ollama model into memory with a specific context window.

    Narrower than ``warm_up_for_run``, and kept only for what warm-up cannot
    express: ``unified_ai_client.warm_up`` loads an Ollama model with whatever
    context the provider config carries, so a caller that needs a particular
    ``context_size`` — the reasoning judge, which feeds whole traces to a local
    model — has to come through here instead.

    ``context_size`` must be set here rather than per call: Ollama allocates the
    context window at load time, so a later call asking for a different num_ctx
    forces a reload. Leaving it unset means the server default applies, which
    silently truncates long inputs such as a raw reasoning trace.

    Args:
        provider: LLM provider name.
        model_name: Model identifier to preload.
        context_size: Context window to allocate, in tokens. Ignored when None.
    """
    if provider.lower() == "ollama":
        extra = {"context_size": context_size} if context_size else {}
        preload_model(
            provider=provider,
            model=model_name,
            keep_alive=LOCAL_KEEP_ALIVE,
            **extra,
        )


def teardown() -> None:
    """Release everything this run allocated on a provider's side.

    UnifiedAiClient's cleanup() drops Google's uploaded files and unloads every
    local model it has loaded, its own registry included, so nothing has to be
    tracked here. Resetting the candidate slot is the part that does belong
    here: without it a second run in the same process — which is exactly the
    MCP server's shape — would try to release a model cleanup() already let go.

    Should be called at the end of each run session or in a finally block.
    UnifiedAiClient also registers cleanup automatically via atexit.
    """
    global _resident_candidate

    _resident_candidate = None
    cleanup()


def _reset_residency_state() -> None:
    """Forget which candidate model is resident, without touching the provider.

    For tests only: the slot is module state, so a key left behind by one test
    would make the next one attempt a real unload.
    """
    global _resident_candidate

    _resident_candidate = None


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    fn_kwargs: dict | None = None,
    timeout: float,
    provider: str,
    model: str,
) -> tuple[Any, bool]:
    """Run fn in a single-worker executor with a timeout.

    On timeout the model is released, so a call this process has given up on
    does not leave a pinned model holding VRAM for the rest of the run. That
    goes through the client rather than the `ollama stop` CLI, so it needs no
    binary on PATH and reaches a remote or non-default-port server, which the
    CLI would not. It is a no-op for providers with no residency concept, so no
    provider check is needed here.

    Args:
        fn: Callable to execute.
        *args: Positional arguments passed to fn.
        fn_kwargs: Keyword arguments passed to fn.
        timeout: Timeout in seconds.
        provider: LLM provider name, for the unload on timeout.
        model: Model identifier to release on timeout.

    Returns:
        Tuple of (result, timed_out). When timed_out is True, result is None
        and the caller builds its own fallback value.
    """
    fn_kwargs = fn_kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **fn_kwargs)
        try:
            return future.result(timeout=timeout), False
        except concurrent.futures.TimeoutError:
            unload_model(provider, model)
            return None, True
