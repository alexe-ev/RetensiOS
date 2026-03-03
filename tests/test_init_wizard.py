from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
from click.testing import CliRunner

from rfm_engine import config as app_config
from rfm_engine.cli import init as init_wizard
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
    monkeypatch.setattr(init_wizard, "CONFIG_FILE", config_file)
    return config_file


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_init_first_run_defaults_create_config(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["init"],
        input="\n\n\nsk-ant-test-123456\n\n",
    )

    assert result.exit_code == 0
    assert config_file.exists()
    config = app_config.read_config()
    assert config["output"]["path"] == "./output"
    assert config["ai"]["enabled"] is True
    assert config["ai"]["provider"] == "anthropic"
    assert config["ai"]["api_key"] == "sk-ant-test-123456"


def test_init_ai_disabled_skips_provider_and_key_prompts(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["init"],
        input="\n2\n\n",
    )

    assert result.exit_code == 0
    assert "[3/5] AI provider?" not in result.output
    assert "[4/5] Paste your API key" not in result.output
    config = app_config.read_config()
    assert config["ai"]["enabled"] is False
    assert "provider" not in config["ai"]
    assert "api_key" not in config["ai"]


def test_init_custom_output_path_is_saved(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["init"],
        input="2\nexports\n2\n\n",
    )

    assert result.exit_code == 0
    config = app_config.read_config()
    assert config["output"]["path"] == "exports"


def test_init_provider_selection_openai(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["init"],
        input="\n\n2\nsk-openai-123\n\n",
    )

    assert result.exit_code == 0
    config = app_config.read_config()
    assert config["ai"]["provider"] == "openai"


def test_init_repeat_run_file_processes_csv(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "exports"
    app_config.write_config({"output": {"path": str(output_root)}, "ai": {"enabled": False}})
    runner = CliRunner()

    input_csv = _write_csv(
        tmp_path / "input.csv",
        [{"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0}],
    )
    result = runner.invoke(cli, ["init"], input=f"1\n{input_csv}\n")

    assert result.exit_code == 0
    assert config_file.exists()
    assert (output_root / "input" / "customers_rfm_input.csv").exists()


def test_init_repeat_change_settings_overwrites_config(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    app_config.write_config({"output": {"path": "old"}, "ai": {"enabled": False}})
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["init"],
        input="2\n2\nnew-output\n2\n\n",
    )

    assert result.exit_code == 0
    config = app_config.read_config()
    assert config["output"]["path"] == "new-output"
    assert config["ai"]["enabled"] is False


def test_init_repeat_view_current_settings_masks_api_key(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    app_config.write_config(
        {
            "output": {"path": "exports"},
            "ai": {"enabled": True, "provider": "anthropic", "api_key": "sk-secret-123456"},
        }
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["init"], input="3\n")

    assert result.exit_code == 0
    assert "Current settings:" in result.output
    assert "- Output path: exports" in result.output
    assert "- AI enabled: yes" in result.output
    assert "- Provider: anthropic" in result.output
    assert "- API key: sk-...456" in result.output


def test_init_repeat_run_file_invalid_path_is_friendly(monkeypatch, tmp_path: Path) -> None:
    _patch_config_paths(monkeypatch, tmp_path)
    app_config.write_config({"output": {"path": "exports"}, "ai": {"enabled": False}})
    runner = CliRunner()

    result = runner.invoke(cli, ["init"], input="1\nmissing.csv\n")

    assert result.exit_code == 0
    assert "Input error: file does not exist: missing.csv" in result.output


def test_init_ctrl_c_exits_without_partial_write(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()
    monkeypatch.setattr(
        init_wizard,
        "_prompt_output_path",
        lambda: (_ for _ in ()).throw(click.Abort()),
    )

    result = runner.invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "Setup cancelled." in result.output
    assert not config_file.exists()


def test_init_invalid_csv_path_allows_retry_or_skip(monkeypatch, tmp_path: Path) -> None:
    config_file = _patch_config_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["init"],
        input="\n2\n1\nmissing.csv\n2\n",
    )

    assert result.exit_code == 0
    assert config_file.exists()
    assert "Input error: file does not exist: missing.csv" in result.output
    assert "Path not found. What would you like to do?" in result.output


def test_api_key_preview_is_masked_and_keeps_provider_prefix() -> None:
    preview = init_wizard._render_api_key_preview("sk-proj-abcdef1234567a3f", "openai")
    assert preview == "Saved: sk-proj-••••••7a3f"


def test_api_key_prefix_mismatch_prompts_confirmation(monkeypatch, capsys) -> None:
    monkeypatch.setattr(init_wizard, "_prompt_confirm", lambda *_args, **_kwargs: True)
    value = init_wizard._validate_api_key_prefix("anthropic", "sk-test-123456")
    output = capsys.readouterr().out

    assert value == "sk-test-123456"
    assert "doesn't look like an Anthropic key" in output
    assert "expected prefix: sk-ant-" in output


def test_prompt_menu_uses_inquirer_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(init_wizard, "_supports_interactive_ui", lambda: True)

    class FakePrompt:
        @staticmethod
        def execute() -> str:
            return "openai"

    fake_inquirer = SimpleNamespace(select=lambda **_kwargs: FakePrompt())
    monkeypatch.setattr(init_wizard, "inquirer", fake_inquirer)

    choice = init_wizard._prompt_menu(
        "[3/5] AI provider?",
        [("anthropic", "Anthropic (Claude)"), ("openai", "OpenAI")],
        default_value="anthropic",
    )

    assert choice == "openai"


def test_prompt_menu_falls_back_to_click_when_inquirer_fails(monkeypatch) -> None:
    monkeypatch.setattr(init_wizard, "_supports_interactive_ui", lambda: True)

    class FailingPrompt:
        @staticmethod
        def execute() -> str:
            raise RuntimeError("no tty")

    fake_inquirer = SimpleNamespace(select=lambda **_kwargs: FailingPrompt())
    monkeypatch.setattr(init_wizard, "inquirer", fake_inquirer)
    monkeypatch.setattr(click, "prompt", lambda *_args, **_kwargs: 2)

    choice = init_wizard._prompt_menu(
        "What would you like to do?",
        [("run_file", "Run a file"), ("change_settings", "Change settings")],
        default_value="run_file",
    )

    assert choice == "change_settings"


def test_prompt_secret_uses_compact_mask_in_interactive_mode(monkeypatch) -> None:
    monkeypatch.setattr(init_wizard, "_supports_interactive_ui", lambda: True)
    captured: dict[str, object] = {}

    class FakePrompt:
        @staticmethod
        def execute() -> str:
            return "sk-test-123456"

    def fake_secret(**kwargs):
        captured.update(kwargs)
        return FakePrompt()

    fake_inquirer = SimpleNamespace(secret=fake_secret)
    monkeypatch.setattr(init_wizard, "inquirer", fake_inquirer)

    value = init_wizard._prompt_secret("[4/5] Paste your API key")

    assert value == "sk-test-123456"
    assert "transformer" in captured
    transformer = captured["transformer"]
    assert callable(transformer)
    assert transformer("any-length-key") == "********"
