from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from rfm_engine import config as app_config
from rfm_engine.cli.main import cli


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _isolated_global_config(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / ".config" / "retensios"
    config_file = config_dir / "config.toml"
    legacy_config_dir = tmp_path / ".config" / "rfm-engine"
    legacy_config_file = legacy_config_dir / "config.toml"

    monkeypatch.setattr(app_config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(app_config, "CONFIG_FILE", config_file)
    monkeypatch.setattr(app_config, "LEGACY_CONFIG_DIR", legacy_config_dir)
    monkeypatch.setattr(app_config, "LEGACY_CONFIG_FILE", legacy_config_file)


def test_cli_happy_path_writes_all_outputs() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [
                {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
                {"user_id": "u1", "order_date": "2026-01-10", "revenue": 80.0},
                {"user_id": "u2", "order_date": "2026-01-02", "revenue": 50.0},
            ],
        )

        result = runner.invoke(cli, ["run", str(input_path), "--no-ai"])

        assert result.exit_code == 0
        out_dir = Path("outputs/input")
        assert (out_dir / "customers_rfm_input.csv").exists()
        assert (out_dir / "segments_summary_input.csv").exists()
        assert (out_dir / "report_input.md").exists()


def test_cli_missing_file_returns_exit_code_1() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "does-not-exist.csv", "--no-ai"])
    assert result.exit_code == 1
    assert "Input error:" in result.output


def test_cli_bad_schema_returns_exit_code_1() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        bad_schema_path = _write_csv(
            Path("bad.csv"),
            [
                {"user_id": "u1", "order_date": "2026-01-01"},
            ],
        )
        result = runner.invoke(cli, ["run", str(bad_schema_path), "--no-ai"])

        assert result.exit_code == 1
        assert "Missing required columns" in result.output


def test_cli_today_flag_changes_recency_reference_date() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [{"user_id": "u1", "order_date": "2024-12-31", "revenue": 100.0}],
        )

        result = runner.invoke(
            cli,
            ["run", str(input_path), "--today", "2025-01-01", "--no-ai"],
        )
        assert result.exit_code == 0

        customers = pd.read_csv("outputs/input/customers_rfm_input.csv")
        assert int(customers.iloc[0]["recency_days"]) == 1


def test_cli_include_refunds_keeps_negative_revenue_rows() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [
                {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
                {"user_id": "u1", "order_date": "2026-01-05", "revenue": -20.0},
                {"user_id": "u2", "order_date": "2026-01-03", "revenue": 50.0},
            ],
        )

        result = runner.invoke(
            cli,
            ["run", str(input_path), "--include-refunds", "--no-ai"],
        )
        assert result.exit_code == 0

        customers = pd.read_csv("outputs/input/customers_rfm_input.csv").set_index("user_id")
        assert float(customers.loc["u1", "monetary"]) == 80.0


def test_cli_output_flag_writes_files_to_custom_directory() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [{"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0}],
        )
        result = runner.invoke(
            cli,
            ["run", str(input_path), "--output", "exports", "--no-ai"],
        )

        assert result.exit_code == 0
        out_dir = Path("exports/input")
        assert (out_dir / "customers_rfm_input.csv").exists()
        assert (out_dir / "segments_summary_input.csv").exists()
        assert (out_dir / "report_input.md").exists()


def test_cli_no_ai_skips_ai_even_when_api_key_is_set(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [{"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0}],
        )
        result = runner.invoke(cli, ["run", str(input_path), "--no-ai"])

        assert result.exit_code == 0
        assert "AI generation skipped (--no-ai)." in result.output


def test_cli_ai_timeout_like_failure_does_not_fail_pipeline(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr("rfm_engine.ai.generator.generate_playbooks", lambda *args, **kwargs: None)

    with runner.isolated_filesystem():
        input_path = _write_csv(
            Path("input.csv"),
            [{"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0}],
        )
        result = runner.invoke(cli, ["run", str(input_path)])

        assert result.exit_code == 0
        assert "AI generation in progress." in result.output
        assert "AI generation failed or produced no playbooks." in result.output


def test_cli_help_includes_all_flags() -> None:
    runner = CliRunner()

    root_help = runner.invoke(cli, ["--help"])
    run_help = runner.invoke(cli, ["run", "--help"])

    assert root_help.exit_code == 0
    assert run_help.exit_code == 0
    assert "run" in root_help.output
    assert "config" in root_help.output
    assert "--advanced" in run_help.output
    assert "--today" in run_help.output
    assert "--include-refunds" in run_help.output
    assert "--no-ai" in run_help.output
    assert "--output" in run_help.output
    assert "--api-key" in run_help.output
    assert "--provider" in run_help.output
