from __future__ import annotations

import json
import math
import threading
from typing import Any, Callable

from unified_ai_client import (
    call_ai,
    configure_provider,
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
# timeout. Zero, so the abandoned call's socket closes at the same moment we
# stop waiting for it.
#
# It used to be 5, then 2, to guarantee that OUR deadline expired first: if the
# provider's fired first, call_ai raised, and generation.py had no except around
# get_llm_response, so a plain timeout surfaced as a traceback instead of a
# placeholder. generation.py now catches that and writes GEN_ERROR_TEXT, so
# losing the race costs nothing and the buffer has no job left.
#
# What the buffer cost is why it is gone: until the provider timeout closes the
# socket, the abandoned call is still running, and on Ollama the next request is
# issued the moment we log the timeout. For the length of the buffer two
# requests were live against the same model, and with OLLAMA_NUM_PARALLEL=1 the
# second one queued, putting that wait inside the next measured elapsed.
#
# What remains is a boundary case, and it is diagnostic only: right at the
# deadline the two timers are simultaneous, so a genuine timeout can be recorded
# as GEN_ERROR_TEXT if the provider's exception lands first. Both are
# placeholders and both are retryable. In practice ours still expires first,
# because the provider's timer starts later, at the socket operation, downstream
# of building the payload.
PROVIDER_TIMEOUT_BUFFER_SECONDS = 0

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

# Whether preload_model_for_run registered an Ollama context_size. UnifiedAiClient
# persists it through configure_provider, so every later Ollama call in the
# process carries that num_ctx: in the long-lived MCP server, the next run's
# candidates included. teardown() takes it back.
_context_size_registered = False


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
    *,
    max_response_timeout_seconds: float | None = None,
    max_retries: int = 5,
) -> list[float]:
    """Get high-dimensional embedding vector for text using UnifiedAiClient.

    Bounded the same way as get_llm_response: get_embedding retries with
    exponential backoff, so without an explicit budget an embedding against a
    server that is down blocks for tens of seconds per call, outliving the
    judge deadline it runs under.

    Args:
        provider: AI provider name (e.g. 'google', 'ollama').
        model_name: Embedding model identifier.
        text: The input text to embed.
        max_response_timeout_seconds: The judge_config deadline. When given,
            the network timeout of each attempt is derived from it exactly as
            in get_llm_response; when None, the client's default applies.
        max_retries: How many times a retryable failure is retried.

    Returns:
        A list of floats representing the embedding vector.
    """
    call_kwargs: dict[str, Any] = dict(
        provider=provider.lower(),
        model=model_name,
        text=text,
        max_retries=max_retries,
    )
    if max_response_timeout_seconds is not None:
        call_kwargs["timeout"] = _buffered_provider_timeout(max_response_timeout_seconds)
    return get_embedding(**call_kwargs)


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
            PROVIDER_TIMEOUT_BUFFER_SECONDS, which is 0, so the socket of a
            call we have given up on closes when we stop waiting rather than
            outliving the deadline into the next measured call. When None,
            call_ai's own default applies. It has no bearing on
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
        # Direct, not getattr with a default: AiResponse declares this field,
        # so a defensive read would quietly turn a rename in the client into
        # "raw" on every trace instead of failing where we could see it.
        reasoning_is_summary=bool(response.reasoning_is_summary),
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
    """Free the VRAM of whatever this process last pinned.

    Usually the candidate that ran before this one, which is where the name
    comes from, but a judge preload goes through it too.

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
    global _context_size_registered

    if provider.lower() == "ollama":
        # A preload is a load, so it follows the same discipline as
        # get_llm_response and warm_up_for_run: release whatever this process
        # has pinned, then record what is resident now. This was the one loader
        # exempt from it, and the cost was real. Without the release, phase 3
        # ran its reasoning judge while the last candidate's model still held
        # VRAM; without the record, release_previous_candidate could never let
        # the judge's own model go, so it stayed pinned until teardown.
        #
        # release_previous_candidate returns early when the slot already holds
        # this pair, so a judge reusing the candidate's model pays no reload.
        #
        # The guard is == "ollama" rather than is_local_provider because
        # context_size is an Ollama notion and that is all this function is
        # still for. Another local backend does not come through here at all.
        release_previous_candidate(provider, model_name)
        extra = {"context_size": context_size} if context_size else {}
        _context_size_registered = _context_size_registered or bool(extra)
        preload_model(
            provider=provider,
            model=model_name,
            keep_alive=LOCAL_KEEP_ALIVE,
            **extra,
        )
        _note_local_model_resident(provider, model_name)


def teardown() -> None:
    """Release everything this run allocated on a provider's side.

    UnifiedAiClient's cleanup() drops Google's uploaded files and unloads every
    local model it has loaded, its own registry included, so nothing has to be
    tracked here. Resetting the candidate slot is the part that does belong
    here: without it a second run in the same process — which is exactly the
    MCP server's shape — would try to release a model cleanup() already let go.
    So is taking back a context_size preload_model_for_run registered, which
    would otherwise size every Ollama load of the next run.

    Should be called at the end of each run session or in a finally block.
    UnifiedAiClient also registers cleanup automatically via atexit.
    """
    global _resident_candidate, _context_size_registered

    _resident_candidate = None
    cleanup()
    if _context_size_registered:
        configure_provider("ollama", context_size=None)
        _context_size_registered = False


def _reset_residency_state() -> None:
    """Forget which candidate model is resident, without touching the provider.

    For tests only: the slot is module state, so a key left behind by one test
    would make the next one attempt a real unload.
    """
    global _resident_candidate, _context_size_registered

    _resident_candidate = None
    _context_size_registered = False


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    fn_kwargs: dict | None = None,
    timeout: float,
) -> tuple[Any, bool]:
    """Run fn in an abandonable daemon thread with a real deadline.

    A single daemon thread rather than a ThreadPoolExecutor, because the pool
    could not deliver the deadline it existed for. `with ThreadPoolExecutor(...)`
    calls shutdown(wait=True) on exit, so returning after a timeout still waited
    for the very call it had given up on: measured, a 0.5s deadline over a 3s
    call returned at 3.0s. shutdown(wait=False) only moves that block, since
    concurrent.futures registers an atexit hook that joins its workers - the CLI
    would print the report path and then hang until the socket died. A daemon
    thread is the only shape that is abandonable in both places, and the pool
    was never a pool anyway: one submit, one worker, no reuse.

    Abandoning the worker is safe HERE specifically because it writes nothing
    durable. append_event is called by our caller, after this returns, never
    inside fn; the worker only touches process-lifetime caches (the residency
    slot, UnifiedAiClient's loaded-model registry, GoogleProvider's upload
    cache), all of which die with the process.

    On the generation path the worker does not outlive the deadline at all:
    max_retries is 0 and PROVIDER_TIMEOUT_BUFFER_SECONDS is 0, so the provider
    timeout closes the socket at the same moment we stop waiting, and an
    abandoned generation cannot overlap the next measured call. The judge path
    is the one that can, since one deadline covers two calls with a retry each.
    Note the limit on Google: UnifiedAiClient wraps its SDK call in a pool of
    its own whose threads are not daemons, so there the process exit is bounded
    by the SDK, not by us.

    Relying on UnifiedAiClient's own timeout instead was considered and does not
    work: it is per attempt, not total ((max_retries + 1) x timeout plus
    backoff), it is a socket timeout covering only the HTTP leg, and it cannot
    bound evaluate_with_judge, which is two calls plus a nested pool.

    This deliberately does **not** touch model residency. It used to release
    the model on timeout, on the theory that a call this process had given up
    on must not leave a pinned model holding VRAM; but the model in question is
    never orphaned, and unloading it here cost more than it saved:

    - It is the running candidate's own model, recorded in the residency slot,
      so `release_previous_candidate` frees it at the next candidate boundary
      and `cleanup()` frees it at exit either way. There is no leak to prevent.
    - Warm-up runs once per candidate, outside the timed block, so the reload
      the unload forces lands inside the `elapsed` of the next repetition or
      test case of that same candidate. That is exactly the measurement error
      warm-up exists to remove.
    - It did not even shorten the wait. The provider-level timeout is only
      PROVIDER_TIMEOUT_BUFFER_SECONDS behind this one, and closing that socket
      is what stops the abandoned generation.

    Residency stays what LOCAL_KEEP_ALIVE says it is: a decision the run makes
    at the candidate boundary and at teardown, never one a timer makes.

    Args:
        fn: Callable to execute.
        *args: Positional arguments passed to fn.
        fn_kwargs: Keyword arguments passed to fn.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (result, timed_out). When timed_out is True, result is None
        and the caller builds its own fallback value.

    Raises:
        BaseException: Whatever fn raised, re-raised in the calling thread.
            Future.result() gave this for free and callers depend on it: a
            phase that fails must still propagate out of run_project.
    """
    fn_kwargs = fn_kwargs or {}
    outcome: list[tuple[Any, BaseException | None]] = []

    def _run() -> None:
        try:
            outcome.append((fn(*args, **fn_kwargs), None))
        except BaseException as exc:  # noqa: BLE001 - mirrors Future.result()
            outcome.append((None, exc))

    worker = threading.Thread(target=_run, name="ptenv-timed-call", daemon=True)
    worker.start()
    worker.join(timeout)

    # A non-empty `outcome` is the exact discriminant, not worker.is_alive():
    # the thread is still alive for a moment after appending its result, so
    # is_alive() would report a timeout for a call that had just succeeded.
    if not outcome:
        return None, True
    result, exc = outcome[0]
    if exc is not None:
        raise exc
    return result, False


def _strip_code_fence(text: str) -> str:
    """Strip a wrapping Markdown code fence from LLM output, if present.

    Args:
        text: Response text, already .strip()-ed.

    Returns:
        text with the first/last fence lines removed if text starts with
        ``` and ends with a lone ``` line; otherwise text unchanged.
    """
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def parse_json_object(text: str) -> dict | None:
    """Parse a judge's JSON reply, tolerating a fence or a one-element list.

    Every judge call asks for response_mime_type="application/json", which
    makes a wrapped or list-wrapped reply the rare case rather than the
    common one, but it is not guaranteed for every backend the way it is for
    Google: Ollama's `format: json` and the OpenAI-compatible `json_object`
    constrain the syntax, and a model can still fence or wrap what it emits.
    Without this, a wrapped reply loses an already-paid-for call silently.

    Args:
        text: Raw response text, not yet stripped or parsed.

    Returns:
        The parsed dict, or None if the text is not valid JSON, or parses to
        something that is neither a dict nor a list whose first element is one.
    """
    stripped = _strip_code_fence(text.strip())
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else None
    return data if isinstance(data, dict) else None
