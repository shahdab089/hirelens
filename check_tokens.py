"""
Quick Groq budget check — prints how many tokens/requests you have left.

Usage:
    python check_tokens.py

Reads GROQ_API_KEY from the environment (same one the app uses). Makes one
tiny call and reads the rate-limit headers Groq returns on every response.
"""
import os

from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def main() -> None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not set in this terminal.")
        return

    client = Groq(api_key=key)
    # .with_raw_response gives us the HTTP headers, not just the parsed body.
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    h = raw.headers

    print(f"Model: {MODEL}\n")
    print("Per-minute window:")
    print(f"  tokens remaining : {h.get('x-ratelimit-remaining-tokens', '?')} / {h.get('x-ratelimit-limit-tokens', '?')}")
    print(f"  requests remaining: {h.get('x-ratelimit-remaining-requests', '?')} / {h.get('x-ratelimit-limit-requests', '?')}")
    print(f"  tokens reset in   : {h.get('x-ratelimit-reset-tokens', '?')}")
    print(f"  requests reset in : {h.get('x-ratelimit-reset-requests', '?')}")


if __name__ == "__main__":
    main()
