from __future__ import annotations

import concurrent.futures
import math
import subprocess
from typing import Any, Callable

from unified_ai_client import call_ai, preload_model, cleanup, get_embedding

from prompttestenv.config import get_app_config
from prompttestenv.models import LlmResult


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
    local_media_path: str | None = None,
    temp: float = 0.7,
    response_mime_type: str | None = None,
    thinking: bool | str = "default",
    disable_safety: bool = False,
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
        local_media_path: Optional local file path for multimodal input. The file
            is passed directly to UnifiedAiClient, which classifies and encodes it
            according to the provider (base64, upload, or text inline).
        temp: Sampling temperature.
        response_mime_type: If 'application/json', enables JSON output mode.
        thinking: Whether to enable thinking/reasoning mode.
        disable_safety: Whether to disable safety settings (Google Gemini only).

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

    extra_options = {}
    if disable_safety:
        extra_options["disable_safety"] = True

    response = call_ai(
        provider=normalized_provider,
        model=model_name,
        prompt=user_prompt,
        system_prompt=system_instruction,
        file_path=local_media_path,
        temperature=temp,
        thinking=thinking,
        format_json=(response_mime_type == "application/json"),
        max_retries=5,
        extra_options=extra_options,
    )
    return LlmResult(
        text=response.text,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
        reasoning_text=response.reasoning_text or "",
        reasoning_is_summary=bool(getattr(response, "reasoning_is_summary", False)),
    )


def preload_model_for_run(
    provider: str,
    model_name: str,
    context_size: int | None = None,
) -> None:
    """Preload a model into memory before starting a benchmark run.

    Only meaningful for Ollama (Google does not support preloading).

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
        preload_model(provider=provider, model=model_name, keep_alive="15m", **extra)


def teardown() -> None:
    """Clean up all remote resources (Google uploaded files).

    Should be called at the end of each run session or in a finally block.
    UnifiedAiClient also registers this automatically via atexit.
    """
    cleanup()


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    fn_kwargs: dict | None = None,
    timeout: float,
    provider: str,
    model: str,
) -> tuple[Any, bool]:
    """Run fn in a single-worker executor with a timeout.

    On timeout, best-effort stops the Ollama model via `ollama stop` (only
    when provider is Ollama) so a hung process doesn't keep hogging resources.

    Args:
        fn: Callable to execute.
        *args: Positional arguments passed to fn.
        fn_kwargs: Keyword arguments passed to fn.
        timeout: Timeout in seconds.
        provider: LLM provider name (decides whether to run `ollama stop`).
        model: Model identifier passed to `ollama stop` on timeout.

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
            if provider.lower() == "ollama":
                subprocess.run(["ollama", "stop", model], capture_output=True)
            return None, True
