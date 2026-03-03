from __future__ import annotations

import types

from rfm_engine.ai.providers import anthropic_provider, get_provider, openai_provider


def test_provider_registry_resolves_known_providers() -> None:
    assert get_provider("openai") is openai_provider
    assert get_provider("anthropic") is anthropic_provider


def test_provider_registry_raises_clear_error_for_unknown_provider() -> None:
    try:
        get_provider("foo")
    except ValueError as exc:
        text = str(exc)
        assert "Unknown AI provider 'foo'" in text
        assert "anthropic" in text
        assert "openai" in text
    else:
        raise AssertionError("Expected ValueError for unknown provider.")


def test_openai_provider_calls_sdk_with_expected_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(output_text="ok")

    class FakeOpenAI:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(openai_provider, "OpenAI", FakeOpenAI)

    result = openai_provider.call("sk-openai", "system", "user")
    assert result == "ok"
    assert captured["api_key"] == "sk-openai"
    assert captured["kwargs"]["model"] == "gpt-4o-mini"


def test_anthropic_provider_calls_sdk_with_expected_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="ok")])

    class FakeAnthropic:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic_provider, "Anthropic", FakeAnthropic)

    result = anthropic_provider.call("sk-ant", "system", "user")
    assert result == "ok"
    assert captured["api_key"] == "sk-ant"
    assert captured["kwargs"]["model"] == "claude-3-5-haiku-latest"
