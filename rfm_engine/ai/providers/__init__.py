"""AI provider registry and dispatch helpers."""

from __future__ import annotations

from types import ModuleType

from rfm_engine.ai.providers import anthropic_provider, openai_provider

PROVIDERS: dict[str, ModuleType] = {
    "openai": openai_provider,
    "anthropic": anthropic_provider,
}


def get_supported_providers() -> list[str]:
    """Return sorted supported provider names."""
    return sorted(PROVIDERS.keys())


def get_provider(name: str) -> ModuleType:
    """Return provider module by name."""
    normalized = (name or "").strip().lower()
    provider = PROVIDERS.get(normalized)
    if provider is None:
        supported = ", ".join(get_supported_providers())
        raise ValueError(f"Unknown AI provider '{name}'. Supported providers: {supported}.")
    return provider
