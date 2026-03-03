"""Anthropic provider adapter."""

from __future__ import annotations

from anthropic import Anthropic


def call(api_key: str, system_prompt: str, user_prompt: str) -> str:
    """Execute a completion request via Anthropic Messages API."""
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    parts = getattr(response, "content", None)
    if not parts:
        raise RuntimeError("Empty AI response.")

    text_chunks: list[str] = []
    for part in parts:
        chunk = getattr(part, "text", "")
        if isinstance(chunk, str) and chunk.strip():
            text_chunks.append(chunk.strip())

    if not text_chunks:
        raise RuntimeError("Empty AI response.")

    return "\n\n".join(text_chunks)
