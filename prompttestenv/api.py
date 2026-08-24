from __future__ import annotations

import math
from unified_ai_client import call_ai, preload_model, cleanup, get_embedding


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
) -> tuple[str, int, int, str]:
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
        Tuple of (response_text, output_tokens, reasoning_tokens, reasoning_text).
        reasoning_text is the raw thinking transcript if available, otherwise an empty string.
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
    return response.text, response.output_tokens, response.reasoning_tokens, (response.reasoning_text or "")


def preload_model_for_run(provider: str, model_name: str) -> None:
    """Preload a model into memory before starting a benchmark run.

    Only meaningful for Ollama (Google does not support preloading).

    Args:
        provider: LLM provider name.
        model_name: Model identifier to preload.
    """
    if provider.lower() == "ollama":
        preload_model(provider=provider, model=model_name, keep_alive="15m")


def teardown() -> None:
    """Clean up all remote resources (Google uploaded files).

    Should be called at the end of each run session or in a finally block.
    UnifiedAiClient also registers this automatically via atexit.
    """
    cleanup()
