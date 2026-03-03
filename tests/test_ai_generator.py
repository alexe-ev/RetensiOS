from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfm_engine.ai import generator


def _segment_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": "Champions",
                "user_count": 120,
                "revenue_share_pct": 48.0,
                "avg_churn_risk": 18.0,
                "avg_recency_days": 6.2,
                "avg_frequency": 7.0,
            },
            {
                "segment": "At Risk",
                "user_count": 80,
                "revenue_share_pct": 36.0,
                "avg_churn_risk": 72.0,
                "avg_recency_days": 29.4,
                "avg_frequency": 2.1,
            },
        ]
    )


def test_detect_api_key_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(generator, "get_api_key_from_config", lambda: None)
    assert generator.detect_api_key() is None


def test_generate_playbooks_writes_markdown_when_api_calls_succeed(
    monkeypatch, tmp_path: Path
) -> None:
    class DummyProvider:
        @staticmethod
        def call(**_: str) -> str:
            return "### Diagnosis\nHealthy segment."

    monkeypatch.setattr(generator, "get_provider", lambda _name: DummyProvider)

    output = generator.generate_playbooks(_segment_summary(), str(tmp_path), api_key="dummy")

    assert output is not None
    content = Path(output).read_text(encoding="utf-8")
    assert content.startswith("# Segment Playbooks")
    assert "## Champions" in content
    assert "## At Risk" in content
    assert "### Diagnosis" in content


def test_generate_playbooks_continues_on_partial_failure(monkeypatch, tmp_path: Path) -> None:
    class DummyProvider:
        @staticmethod
        def call(*, user_prompt: str, **_: str) -> str:
            if "Segment name: At Risk" in user_prompt:
                raise TimeoutError("timed out")
            return "### Tactical Actions\nLaunch retention campaign."

    monkeypatch.setattr(generator, "get_provider", lambda _name: DummyProvider)

    output = generator.generate_playbooks(_segment_summary(), str(tmp_path), api_key="dummy")
    assert output is not None

    content = Path(output).read_text(encoding="utf-8")
    assert "## Champions" in content
    assert "## At Risk" in content
    assert "_Playbook generation failed for this segment._" in content


def test_generate_playbooks_returns_none_when_all_segments_fail(
    monkeypatch, tmp_path: Path
) -> None:
    class DummyProvider:
        @staticmethod
        def call(**_: str) -> str:
            raise TimeoutError("timed out")

    monkeypatch.setattr(generator, "get_provider", lambda _name: DummyProvider)

    output = generator.generate_playbooks(_segment_summary(), str(tmp_path), api_key="dummy")
    assert output is None
    assert not (tmp_path / "segment_playbooks.md").exists()


def test_generate_playbooks_handles_unrecoverable_error(monkeypatch, tmp_path: Path) -> None:
    class DummyProvider:
        @staticmethod
        def call(**_: str) -> str:
            return "ok"

    monkeypatch.setattr(generator, "get_provider", lambda _name: DummyProvider)

    def raise_os_error(*_args, **_kwargs) -> None:
        raise OSError

    monkeypatch.setattr(generator.Path, "mkdir", raise_os_error)

    output = generator.generate_playbooks(_segment_summary(), str(tmp_path), api_key="dummy")
    assert output is None
