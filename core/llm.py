"""
Shared Groq helper for the AI core (scoring, diagnosis).

Groq is the project's free LLM provider. Set GROQ_API_KEY in the environment.
The client is built lazily so importing modules never crashes without a key.
"""
import json
import os
import time
from typing import Any

from groq import Groq

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        _client = Groq(api_key=api_key)
    return _client


def chat_json(system: str, user: str, retries: int = 2) -> dict[str, Any]:
    """Call Groq in JSON mode and return the parsed JSON object, with retries."""
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as err:  # noqa: BLE001 — retry on any transient/parse error
            last_err = err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Groq call failed after {retries + 1} attempts: {last_err}")
