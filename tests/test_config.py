from __future__ import annotations

import stat
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from rfm_engine import config as app_config
from rfm_engine.ai import generator
from rfm_engine.cli.main import cli


def _patch_config_paths(monkeypatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / ".config" / "retensios"
    config_file = config_dir / "config.toml"
    legacy_config_dir = tmp_path / ".config" / "rfm-engine"
    legacy_config_file = legacy_config_dir / "config.toml"
    monkeypatch.setattr(app_config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(app_config, "CONFIG_FILE", config_file)
    monkeypatch.setattr(app_config, "LEGACY_CONFIG_DIR", legacy_config_dir)
    monkeypatch.setattr(app_config, "LEGACY_CONFIG_FILE", legacy_config_file)
    monkeypatch.setattr("rfm_engine.cli.main.CONFIG_FILE", config_file)
    return config_file


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_detect_api_key_priority_chain(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app_config.write_config({"ai": {"api_key": "sk-config-123456"}})

    assert (
        generator.detect_api_key(cli_api_key="sk-cli-777777", provider="openai")
        == "sk-cli-777777"
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-888888")
    assert generator.detect_api_key(provider="openai") == "sk-env-888888"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert generator.detect_api_key(provider="openai") == "sk-config-123456"

    app_config.CONFIG_FILE.unlink()
    assert generator.detect_api_key(provider="openai") is None


def test_detect_api_key_returns_none_on_missing_config(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert generator.detect_api_key(provider="openai") is None


def test_read_config_migrates_legacy_config_and_prints_notice(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    legacy_file = tmp_path / ".config" / "rfm-engine" / "config.toml"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text('[ai]\napi_key = "sk-legacy-123"\n', encoding="utf-8")

    config = app_config.read_config()
    captured = capsys.readouterr()

    assert config_file.exists()
    assert config["ai"]["api_key"] == "sk-legacy-123"
    assert "migrated config from" in captured.out


def test_read_config_skips_migration_without_legacy_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)

    config = app_config.read_config()
    captured = capsys.readouterr()

    assert config == {}
    assert not config_file.exists()
    assert captured.out == ""


def test_detect_api_key_handles_malformed_config(monkeypatch, tmp_path: Path, caplog) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("[ai\napi_key = broken", encoding="utf-8")

    assert generator.detect_api_key(provider="openai") is None
    assert "Malformed config file" in caplog.text


def test_config_set_creates_secure_file_and_permissions(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "set", "api-key", "sk-test-123456"])

    assert result.exit_code == 0
    assert config_file.exists()
    assert '[ai]\napi_key = "sk-test-123456"\n' == config_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(config_file.parent.stat().st_mode) == 0o700


def test_config_get_prints_source_and_masked_value(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()
    app_config.write_config({"ai": {"api_key": "sk-config-123456"}})

    result = runner.invoke(cli, ["config", "get", "api-key"])

    assert result.exit_code == 0
    assert "API key source: config" in result.output
    assert "API key value: sk-...456" in result.output


def test_config_path_prints_config_file_path(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "path"])

    assert result.exit_code == 0
    assert result.output.strip() == str(config_file)


def test_config_set_and_get_provider(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    set_result = runner.invoke(cli, ["config", "set", "provider", "anthropic"])
    get_result = runner.invoke(cli, ["config", "get", "provider"])

    assert set_result.exit_code == 0
    assert get_result.exit_code == 0
    assert 'provider = "anthropic"' in config_file.read_text(encoding="utf-8")
    assert get_result.output.strip() == "anthropic"


def test_detect_api_key_uses_provider_specific_env(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-111111")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-222222")

    assert generator.detect_api_key(provider="anthropic") == "sk-ant-222222"
    assert generator.detect_api_key(provider="openai") == "sk-openai-111111"


def test_run_api_key_flag_passes_override_to_ai_generator(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, str | None] = {"cli_api_key": None, "api_key": None, "provider": None}

    def fake_detect_api_key(
        cli_api_key: str | None = None,
        provider: str | None = None,
    ) -> str | None:
        captured["cli_api_key"] = cli_api_key
        captured["provider"] = provider
        return cli_api_key

    def fake_detect_provider(cli_provider: str | None = None) -> str:
        return cli_provider or "openai"

    def fake_generate_playbooks(
        segment_summary,
        output_dir: str,
        api_key: str,
        provider_name: str = "openai",
        suffix: str = "",
        progress_callback=None,
    ) -> str:
        captured["api_key"] = api_key
        captured["provider"] = provider_name
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        playbook_name = f"segment_playbooks_{suffix}.md" if suffix else "segment_playbooks.md"
        playbook_path = output_path / playbook_name
        playbook_path.write_text("# Segment Playbooks\n", encoding="utf-8")
        return str(playbook_path)

    monkeypatch.setattr("rfm_engine.ai.generator.detect_api_key", fake_detect_api_key)
    monkeypatch.setattr("rfm_engine.ai.generator.detect_provider", fake_detect_provider)
    monkeypatch.setattr("rfm_engine.ai.generator.generate_playbooks", fake_generate_playbooks)

    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [{"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0}],
        )
        result = runner.invoke(
            cli, ["run", str(input_path), "--api-key", "sk-cli-654321", "--provider", "openai"]
        )

    assert result.exit_code == 0
    assert captured["cli_api_key"] == "sk-cli-654321"
    assert captured["api_key"] == "sk-cli-654321"
    assert captured["provider"] == "openai"
