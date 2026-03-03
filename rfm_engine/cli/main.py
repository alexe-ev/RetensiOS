"""CLI entrypoint for the retensios command."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import click

from rfm_engine.cli.init import (
    process_file_now,
    prompt_file_path,
    run_repeat_menu,
    run_wizard,
    save_wizard_config,
    view_current_settings,
)
from rfm_engine.config import (
    CONFIG_FILE,
    SUPPORTED_PROVIDERS,
    get_output_path_from_config,
    get_provider_from_config,
    is_ai_enabled_from_config,
    mask_api_key,
    read_config,
    set_api_key_in_config,
    set_provider_in_config,
)
from rfm_engine.core.churn import calculate_churn_risk
from rfm_engine.core.data_loader import load_csv
from rfm_engine.core.profiler import profile_data
from rfm_engine.core.rfm import calculate_rfm
from rfm_engine.core.scoring import score_rfm
from rfm_engine.core.segmentation import assign_segments
from rfm_engine.reporting.csv_output import write_customer_csv, write_summary_csv
from rfm_engine.reporting.report_builder import build_report
from rfm_engine.reporting.revenue_analysis import analyze_revenue


def _echo_section(title: str) -> None:
    click.echo("")
    click.echo(click.style(f"== {title} ==", fg="cyan", bold=True))


def _echo_step(step: int, total: int, title: str) -> None:
    click.echo(click.style(f"Step {step}/{total}: {title}...", fg="bright_blue", bold=True))


def _format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _echo_status(tag: str, message: str) -> None:
    color = {
        "OK": "green",
        "WARN": "yellow",
        "AI": "magenta",
        "INFO": "white",
    }.get(tag, "white")
    click.echo(f"{click.style(f'[{tag}]', fg=color, bold=True)} {message}")


def _parse_today(today: str | None) -> datetime | None:
    if not today:
        return None
    try:
        return datetime.strptime(today, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Invalid --today value. Use YYYY-MM-DD.") from exc


def _write_report(report_markdown: str, output_dir: str, suffix: str = "") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"report_{suffix}.md" if suffix else "report.md"
    report_path = output_path / filename
    report_path.write_text(report_markdown, encoding="utf-8")
    return str(report_path)


def _maybe_run_ai(
    *,
    no_ai: bool,
    provider: str | None,
    api_key: str | None,
    segment_summary,
    output_dir: str,
    suffix: str = "",
) -> str | None:
    if no_ai:
        _echo_status("AI", "AI generation skipped (--no-ai).")
        return None

    from rfm_engine.ai.generator import detect_api_key, detect_provider, generate_playbooks

    resolved_provider = detect_provider(cli_provider=provider)
    resolved_api_key = detect_api_key(cli_api_key=api_key, provider=resolved_provider)
    if resolved_api_key is None:
        _echo_status("AI", "AI generation skipped (no API key detected).")
        return None

    _echo_status(
        "AI",
        "AI generation in progress. This may take up to a minute depending on segment count.",
    )
    _echo_status("AI", f"Provider: {resolved_provider}")
    ai_start = time.perf_counter()

    def _on_ai_progress(index: int, total: int, segment_name: str) -> None:
        ai_tag = click.style("[AI]", fg="magenta", bold=True)
        click.echo(f"{ai_tag} Segment {index}/{total}: {segment_name}")

    playbook_path = generate_playbooks(
        segment_summary,
        output_dir=output_dir,
        api_key=resolved_api_key,
        provider_name=resolved_provider,
        suffix=suffix,
        progress_callback=_on_ai_progress,
    )
    if playbook_path is None:
        _echo_status(
            "WARN",
            "AI generation failed or produced no playbooks. Continuing without AI output.",
        )
    else:
        _echo_status("OK", f"AI playbooks saved: {Path(playbook_path).name}")
    _echo_status("AI", f"AI stage finished in {_format_duration(time.perf_counter() - ai_start)}")
    return playbook_path


def _run_file_from_wizard_config(config: dict, csv_path: str) -> None:
    ai_section = config.get("ai", {})
    ai_enabled = True
    provider = "openai"
    if isinstance(ai_section, dict):
        ai_enabled = bool(ai_section.get("enabled", True))
        provider = str(ai_section.get("provider", "openai")).strip().lower() or "openai"

    output_path = str(config.get("output", {}).get("path", "outputs")).strip() or "outputs"
    process_file_now(
        csv_path,
        run_pipeline=run.callback,
        output_path=output_path,
        ai_enabled=ai_enabled,
        provider=provider,
    )


@click.group()
def cli() -> None:
    """RetensiOS retention decision framework."""


@cli.command()
@click.argument("input_csv", required=True)
@click.option("--advanced", is_flag=True, help="Enable advanced profiling")
@click.option("--today", type=str, default=None, help="Reference date (YYYY-MM-DD)")
@click.option("--include-refunds", is_flag=True, help="Include refund rows")
@click.option("--no-ai", is_flag=True, help="Disable AI generation")
@click.option("--output", type=click.Path(), default="outputs", help="Output root directory")
@click.option("--api-key", type=str, default=None, help="API key override for this run only")
@click.option(
    "--provider",
    type=click.Choice(SUPPORTED_PROVIDERS, case_sensitive=False),
    default=None,
    help="AI provider override for this run only",
)
def run(
    input_csv: str,
    advanced: bool,
    today: str | None,
    include_refunds: bool,
    no_ai: bool,
    output: str,
    api_key: str | None,
    provider: str | None,
) -> None:
    """Run the retention pipeline on input CSV data."""
    _echo_section("RetensiOS Pipeline")
    click.echo(f"Input file: {input_csv}")

    if output == "outputs":
        output = get_output_path_from_config()
    click.echo(f"Output root: {output}")

    try:
        today_date = _parse_today(today)
    except ValueError as exc:
        click.echo(f"Input error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc

    input_stem = Path(input_csv).stem
    effective_output_dir = str(Path(output) / input_stem)
    run_start = time.perf_counter()

    _echo_section("Core Processing")
    _echo_step(1, 9, "Loading input data")
    step_start = time.perf_counter()
    try:
        transactions = load_csv(input_csv, include_refunds=include_refunds)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Input error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc
    _echo_status(
        "OK",
        f"Loaded {len(transactions)} rows in {_format_duration(time.perf_counter() - step_start)}",
    )

    _echo_step(2, 9, "Profiling dataset")
    step_start = time.perf_counter()
    profile = profile_data(transactions)
    _echo_status(
        "OK",
        (
            f"Profile complete in {_format_duration(time.perf_counter() - step_start)} "
            f"(users={profile['unique_users']}, rows={profile['total_rows']})"
        ),
    )
    if advanced:
        click.echo(
            "Profile: "
            f"rows={profile['total_rows']}, users={profile['unique_users']}, "
            f"revenue_total={profile['revenue_total']:.2f}"
        )
        if profile["warnings"]:
            click.echo("Profile warnings:")
            for warning in profile["warnings"]:
                click.echo(f"- {warning}")

    try:
        _echo_step(3, 9, "Calculating RFM metrics")
        step_start = time.perf_counter()
        rfm = calculate_rfm(transactions, today_date=today_date)
        duration = _format_duration(time.perf_counter() - step_start)
        _echo_status("OK", f"RFM metrics ready in {duration}")

        _echo_step(4, 9, "Scoring RFM")
        step_start = time.perf_counter()
        scored = score_rfm(rfm)
        duration = _format_duration(time.perf_counter() - step_start)
        _echo_status("OK", f"Scoring complete in {duration}")

        _echo_step(5, 9, "Assigning segments")
        step_start = time.perf_counter()
        segmented = assign_segments(scored)
        _echo_status(
            "OK",
            f"Segmentation complete in {_format_duration(time.perf_counter() - step_start)}",
        )

        _echo_step(6, 9, "Calculating churn risk")
        step_start = time.perf_counter()
        customer_output = calculate_churn_risk(segmented)
        _echo_status(
            "OK",
            f"Churn scoring complete in {_format_duration(time.perf_counter() - step_start)}",
        )

        _echo_step(7, 9, "Analyzing revenue")
        step_start = time.perf_counter()
        segment_summary, concentration_index = analyze_revenue(customer_output)
        _echo_status(
            "OK",
            f"Revenue analysis complete in {_format_duration(time.perf_counter() - step_start)}",
        )

        _echo_step(8, 9, "Building report")
        step_start = time.perf_counter()
        report_markdown = build_report(profile, segment_summary, concentration_index)
        report_path = _write_report(report_markdown, effective_output_dir, suffix=input_stem)
        _echo_status("OK", f"Report built in {_format_duration(time.perf_counter() - step_start)}")

        _echo_step(9, 9, "Writing CSV outputs")
        step_start = time.perf_counter()
        customers_csv_path = write_customer_csv(
            customer_output, effective_output_dir, suffix=input_stem,
        )
        summary_csv_path = write_summary_csv(
            segment_summary, effective_output_dir, suffix=input_stem,
        )
        duration = _format_duration(time.perf_counter() - step_start)
        _echo_status("OK", f"CSV outputs written in {duration}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Processing error: {exc}", err=True)
        raise click.exceptions.Exit(2) from exc

    configured_ai_enabled = is_ai_enabled_from_config()
    effective_no_ai = no_ai or (not configured_ai_enabled and api_key is None and provider is None)
    if not no_ai and effective_no_ai:
        _echo_status("AI", "AI generation skipped (disabled in config).")

    _echo_section("AI Stage")
    playbook_path = _maybe_run_ai(
        no_ai=effective_no_ai,
        provider=provider,
        api_key=api_key,
        segment_summary=segment_summary,
        output_dir=effective_output_dir,
        suffix=input_stem,
    )

    _echo_section("Run Complete")
    click.echo(click.style("Pipeline completed successfully.", fg="green", bold=True))
    _echo_status("OK", f"Total runtime: {_format_duration(time.perf_counter() - run_start)}")
    click.echo(f"Output directory: {effective_output_dir}/")
    click.echo(f"- {Path(customers_csv_path).name}")
    click.echo(f"- {Path(summary_csv_path).name}")
    click.echo(f"- {Path(report_path).name}")
    if playbook_path:
        click.echo(f"- {Path(playbook_path).name}")


@cli.group()
def config() -> None:
    """Manage global CLI configuration."""


@config.command("path")
def config_path() -> None:
    """Print config file location."""
    click.echo(str(CONFIG_FILE))


@config.group("set")
def config_set() -> None:
    """Set config values."""


@config_set.command("api-key")
@click.argument("api_key", type=str, required=True)
def config_set_api_key(api_key: str) -> None:
    """Save API key in global config."""
    set_api_key_in_config(api_key.strip())
    click.echo(f"API key saved to config ({mask_api_key(api_key)}).")


@config_set.command("provider")
@click.argument(
    "provider",
    type=click.Choice(SUPPORTED_PROVIDERS, case_sensitive=False),
    required=True,
)
def config_set_provider(provider: str) -> None:
    """Save provider in global config."""
    normalized = provider.strip().lower()
    set_provider_in_config(normalized)
    click.echo(f"Provider saved to config ({normalized}).")


@config.group("get")
def config_get() -> None:
    """Get config values."""


@config_get.command("api-key")
def config_get_api_key() -> None:
    """Print resolved API key source and masked value."""
    from rfm_engine.ai.generator import detect_api_key_with_source, detect_provider

    provider = detect_provider()
    api_key, source = detect_api_key_with_source(provider=provider)
    if api_key is None or source is None:
        click.echo("API key: not set")
        return

    click.echo(f"Provider: {provider}")
    click.echo(f"API key source: {source}")
    click.echo(f"API key value: {mask_api_key(api_key)}")


@config_get.command("provider")
def config_get_provider() -> None:
    """Print configured provider value."""
    click.echo(get_provider_from_config())


@cli.command("init")
def init_command() -> None:
    """Launch the interactive setup wizard."""
    try:
        if not CONFIG_FILE.exists() or not read_config():
            config, csv_path = run_wizard()
            save_wizard_config(config)

            if csv_path:
                _run_file_from_wizard_config(config, csv_path)
            return

        action = run_repeat_menu()
        if action == "view_settings":
            view_current_settings()
            return

        if action == "change_settings":
            config, csv_path = run_wizard()
            save_wizard_config(config)
            if csv_path:
                _run_file_from_wizard_config(config, csv_path)
            return

        csv_path = prompt_file_path()
        output_path = get_output_path_from_config()
        ai_enabled = is_ai_enabled_from_config()
        provider = get_provider_from_config()
        process_file_now(
            csv_path,
            run_pipeline=run.callback,
            output_path=output_path,
            ai_enabled=ai_enabled,
            provider=provider,
        )
    except click.Abort:
        click.echo("\nSetup cancelled.")
