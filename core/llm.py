"""
LLM helper for the AI core (parsing, scoring, diagnosis).

PRIMARY provider: Anthropic Claude Haiku 4.5 (best quality-per-dollar for
structured JSON extraction; minimal hallucination on resume/JD tasks).
FALLBACK provider: Groq llama-3.3-70b-versatile (free tier; independent token
bucket means we still have capacity when Anthropic is unavailable).

Env vars:
  GROQ_API_KEY        -- REQUIRED. Powers the fallback path (and the whole app if
                         no Anthropic key is present).
  ANTHROPIC_API_KEY   -- OPTIONAL but recommended. When set, Claude is the primary;
                         if it's missing/invalid/rate-limited we transparently fall
                         back to Groq, so a bad or absent key never breaks requests.

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


def _is_auth_or_config(err: Exception) -> bool:
    """Primary provider is unusable because of a missing/invalid key, not load."""
    s = str(err).lower()
    return any(tok in s for tok in (
        "anthropic_api_key", "authentication", "x-api-key", "401", "403",
        "unauthorized", "permission", "invalid api key", "invalid x-api-key",
    ))


def _repair_json(raw: str) -> dict:
    """
    Best-effort repair of a truncated JSON string from a token-limited response.
    Tries json.loads first; if that fails, attempts to close unclosed structures.
    Raises json.JSONDecodeError if the string is too broken to recover.
    """
    raw = raw.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Fast path — already valid
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Attempt repair: strip to the last complete top-level key-value pair,
    # close any open string, then close open braces/brackets.
    # Strategy: find the last comma at depth 1 and truncate there, then close.
    depth = 0
    in_str = False
    escape = False
    last_safe = 0  # index of last ',' at depth 1 (safe truncation point)

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 1:
            last_safe = i

    # If we never found a safe point, the JSON is too broken
    if last_safe == 0:
        raise json.JSONDecodeError("Unrepairable JSON", raw, 0)

    truncated = raw[:last_safe]

    # Count unclosed brackets
    opens = truncated.count("{") - truncated.count("}")
    arr_opens = truncated.count("[") - truncated.count("]")

    # Close any unclosed arrays then objects
    closing = "]" * max(arr_opens, 0) + "}" * max(opens, 0)
    repaired = truncated + closing

    result = json.loads(repaired)  # raises if still broken
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object after repair, got {type(result)}")
    return result


def _parse_llm_json(raw: str) -> dict:
    """Parse LLM output as JSON, with repair fallback for truncated responses."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        raise ValueError(f"Expected JSON object, got {type(result)}")
    except json.JSONDecodeError:
        return _repair_json(raw)


def _call_anthropic(messages: list[dict], max_tokens: int, retries: int) -> "dict[str, Any] | None":
    """
    Call Claude Haiku 4.5. Returns parsed JSON dict on success, or None when the
    primary is unusable (rate-limited, or no/invalid ANTHROPIC_API_KEY) so the
    caller transparently falls through to the Groq fallback. Raises only on
    genuinely unexpected hard errors.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # Primary not configured on this host -> use the Groq fallback silently.
        return None
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
        "Do not include any text, markdown, code fences, or explanation outside the JSON. "
        "Keep string values concise — rationale fields ≤ 25 words, explanation ≤ 60 words."
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
            parsed = _parse_llm_json(raw)
            return parsed
        except Exception as err:  # noqa: BLE001
            last_err = err
            if _is_rate_limit(err) or _is_auth_or_config(err):
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
            return _parse_llm_json(response.choices[0].message.content)
        except Exception as err:  # noqa: BLE001
            last_err = err
            if _is_rate_limit(err):
                raise GroqRateLimit(
                    f"Groq fallback ({GROQ_FALLBACK_MODEL}) is also rate-limited."
                ) from err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Groq fallback call failed: {last_err}") from last_err


def complete_json(messages: list[dict], max_tokens: int = 2800, retries: int = 2) -> "dict[str, Any]":
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


def chat_json(system: str, user: str, retries: int = 2, max_tokens: int = 2800) -> "dict[str, Any]":
    """Convenience wrapper: system + user prompt -> parsed JSON dict."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return complete_json(messages, max_tokens=max_tokens, retries=retries)


def chat_text(system: str, user: str, max_tokens: int = 3000, retries: int = 2) -> str:
    """Like chat_json but returns a plain text string (e.g. for resume rewriting)."""
    # Try Anthropic primary
    if os.environ.get("ANTHROPIC_API_KEY"):
        client = _get_anthropic()
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    temperature=0.35,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text.strip()
            except Exception as err:  # noqa: BLE001
                last_err = err
                if _is_rate_limit(err) or _is_auth_or_config(err):
                    break  # fall through to Groq
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))

    # Groq fallback — plain text (no JSON mode)
    client = _get_groq()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.35,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as err:  # noqa: BLE001
            last_err = err
            if _is_rate_limit(err):
                raise GroqRateLimit("Groq rate-limited during text generation.") from err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Text generation failed after retries: {last_err}") from last_err


def get_client() -> Groq:
    """Returns the Groq client (kept for backward compatibility)."""
    return _get_groq()
