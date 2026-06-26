"""
LLM helper for the AI core (parsing, scoring, diagnosis).

PRIMARY provider: Anthropic Claude Haiku 4.5 (best quality-per-dollar for
structured JSON extraction; minimal hallucination on resume/JD tasks).
FALLBACK provider: Groq llama-3.3-70b-versatile (free tier; independent token
bucket means we still have capacity when Anthropic is unavailable).

Required env vars:
  ANTHROPIC_API_KEY   -- your Anthropic key (get one at console.anthropic.com)
  GROQ_API_KEY        -- your Groq key (fallback; keep the existing one)

Optional overrides:
  ANTHROPIC_MODEL     -- default: claude-haiku-4-5
  GROQ_FALLBACK_MODEL -- default: llama-3.3-70b-versatile
"""
import json
import os
import time
from typing import Any

import anthropic
from groq import Groq

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
GROQ_FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")

_anthropic_client: anthropic.Anthropic | None = None
_groq_client: Groq | None = None


class GroqRateLimit(RuntimeError):
    """Raised when every provider/model is rate-limited or over budget."""


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return any(tok in s for tok in (
        "rate_limit", "429", "413", "too large", "tokens per", "rate limit",
        "overloaded", "capacity",
    ))


def _call_anthropic(messages: list[dict], max_tokens: int, retries: int) -> "dict[str, Any] | None":
    """
    Call Claude Haiku 4.5. Returns parsed JSON dict on success, None on
    rate-limit (caller falls through to Groq). Raises on hard errors.
    """
    client = _get_anthropic()

    system_content = ""
    user_messages: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)

    system_with_json = (
        system_content
        + "\n\nCRITICAL: Your entire response must be a single valid JSON object. "
        "Do not include any text, markdown, code fences, or explanation outside the JSON."
    )

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                system=system_with_json,
                messages=user_messages,
            )
            raw = response.content[0].text.strip()
            # Strip accidental markdown fences if the model adds them
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed)}")
            return parsed
        except Exception as err:  # noqa: BLE001
            last_err = err
            if _is_rate_limit(err):
                return None  # signal: fall through to Groq
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Anthropic call failed after {retries + 1} attempts: {last_err}") from last_err


def _call_groq(messages: list[dict], max_tokens: int, retries: int) -> "dict[str, Any]":
    """Call Groq 70B in JSON mode. Raises GroqRateLimit if throttled."""
    client = _get_groq()
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as err:  # noqa: BLE001
            last_err = err
            if _is_rate_limit(err):
                raise GroqRateLimit(
                    f"Groq fallback ({GROQ_FALLBACK_MODEL}) is also rate-limited."
                ) from err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Groq fallback call failed: {last_err}") from last_err


def complete_json(messages: list[dict], max_tokens: int = 1536, retries: int = 2) -> "dict[str, Any]":
    """
    Try Claude Haiku 4.5 first; fall back to Groq 70B on rate-limits.
    Returns a parsed JSON dict. Raises GroqRateLimit if all providers throttled.
    """
    try:
        result = _call_anthropic(messages, max_tokens=max_tokens, retries=retries)
        if result is not None:
            return result
        # None => rate-limited, fall through to Groq
    except GroqRateLimit:
        raise
    except RuntimeError:
        raise

    return _call_groq(messages, max_tokens=max_tokens, retries=retries)


def chat_json(system: str, user: str, retries: int = 2, max_tokens: int = 1536) -> "dict[str, Any]":
    """Convenience wrapper: system + user prompt -> parsed JSON dict."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return complete_json(messages, max_tokens=max_tokens, retries=retries)


def get_client() -> Groq:
    """Returns the Groq client (kept for backward compatibility)."""
    return _get_groq()
