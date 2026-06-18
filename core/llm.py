"""
Shared Groq helper for the AI core (parsing, scoring, diagnosis).

Groq is the project's free LLM provider. Set GROQ_API_KEY in the environment.
The client is built lazily so importing modules never crashes without a key.

Resilience: each call tries the primary model, and on a rate-limit / too-large
error it automatically retries on a FALLBACK model. The primary (8B) and the
fallback (70B) have separate free-tier token buckets, so this roughly doubles
effective free capacity. If every model is rate-limited, a GroqRateLimit is
raised so the web layer can show a friendly "high demand" message.
"""
import json
import os
import time
from typing import Any

from groq import Groq

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")

# Ordered list of models to try; skip the fallback if it equals the primary.
_MODELS = [GROQ_MODEL] + ([GROQ_FALLBACK_MODEL] if GROQ_FALLBACK_MODEL != GROQ_MODEL else [])

_client: Groq | None = None


class GroqRateLimit(RuntimeError):
    """Raised when every model is rate-limited / over budget."""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        _client = Groq(api_key=api_key)
    return _client


def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return any(
        token in s
        for token in ("rate_limit", "429", "413", "too large", "tokens per", "rate limit")
    )


def complete_json(messages: list[dict], max_tokens: int = 1536, retries: int = 2) -> dict[str, Any]:
    """
    Call Groq in JSON mode, trying each model in turn. Within a model, retry on
    transient errors; on a rate-limit, move straight to the next model. Returns
    the parsed JSON object. Raises GroqRateLimit if all models are throttled.
    """
    client = get_client()
    last_err: Exception | None = None
    hit_rate_limit = False

    for model in _MODELS:
        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max_tokens,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as err:  # noqa: BLE001
                last_err = err
                if _is_rate_limit(err):
                    hit_rate_limit = True
                    break  # don't waste retries on this model — try the next one
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))

    if hit_rate_limit:
        raise GroqRateLimit(
            "All Groq models are currently rate-limited / over budget."
        ) from last_err
    raise RuntimeError(f"Groq call failed after trying {_MODELS}: {last_err}")


def chat_json(system: str, user: str, retries: int = 2, max_tokens: int = 1536) -> dict[str, Any]:
    """Convenience wrapper: system+user prompt -> parsed JSON, with model fallback."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return complete_json(messages, max_tokens=max_tokens, retries=retries)
