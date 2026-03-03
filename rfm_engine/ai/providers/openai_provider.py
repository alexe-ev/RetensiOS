"""OpenAI provider adapter."""

from __future__ import annotations

from openai import OpenAI


def call(api_key: str, system_prompt: str, user_prompt: str) -> str:
    """Execute a completion request via OpenAI Responses API."""
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = getattr(response, "output_text", "")
    if not text:
        raise RuntimeError("Empty AI response.")
    return text.strip()
