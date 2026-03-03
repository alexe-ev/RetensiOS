"""Interactive first-run setup wizard for retensios."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from rfm_engine.config import (
    CONFIG_FILE,
    SUPPORTED_PROVIDERS,
    get_output_path_from_config,
    is_ai_enabled_from_config,
    mask_api_key,
    read_config,
    write_config,
)

try:
    from InquirerPy import inquirer
except Exception:  # noqa: BLE001
    inquirer = None


def _supports_interactive_ui() -> bool:
    return bool(inquirer is not None and sys.stdin.isatty() and sys.stdout.isatty())


def _prompt_menu(title: str, options: list[tuple[str, str]], default_value: str) -> str:
    if _supports_interactive_ui():
        try:
            choices = [{"name": label, "value": value} for value, label in options]
            return str(
                inquirer.select(message=title, choices=choices, default=default_value).execute()
            ).strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise click.Abort() from exc
        except Exception:  # noqa: BLE001
            pass

    click.echo(title)
    for index, (_, label) in enumerate(options, start=1):
        click.echo(f"  {index}) {label}")
    default_index = 1
    for index, (value, _) in enumerate(options, start=1):
        if value == default_value:
            default_index = index
            break
    choice = click.prompt(
        "Choose option", type=click.IntRange(1, len(options)), default=default_index
    )
    return options[choice - 1][0]


def _prompt_text(message: str) -> str:
    if _supports_interactive_ui():
        try:
            return str(inquirer.text(message=message).execute()).strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise click.Abort() from exc
        except Exception:  # noqa: BLE001
            pass
    return click.prompt(message, type=str).strip()


def _prompt_secret(message: str) -> str:
    if _supports_interactive_ui():
        try:
            return str(
                inquirer.secret(
                    message=message,
                    transformer=lambda _result: "********",
                ).execute()
            ).strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise click.Abort() from exc
        except Exception:  # noqa: BLE001
            pass
    return click.prompt(message, hide_input=True, type=str).strip()


def _prompt_confirm(message: str, default: bool = True) -> bool:
    if _supports_interactive_ui():
        try:
            return bool(inquirer.confirm(message=message, default=default).execute())
        except (KeyboardInterrupt, EOFError) as exc:
            raise click.Abort() from exc
        except Exception:  # noqa: BLE001
            pass
    return bool(click.confirm(message, default=default))


def _render_api_key_preview(api_key: str, provider: str) -> str:
    value = api_key.strip()
    if len(value) <= 4:
        return "Saved: ****"
    if provider == "anthropic" and value.startswith("sk-ant-"):
        prefix = "sk-ant-"
    elif provider == "openai" and value.startswith("sk-proj-"):
        prefix = "sk-proj-"
    elif value.startswith("sk-"):
        prefix = "sk-"
    else:
        prefix = value[: min(4, len(value))]
    return f"Saved: {prefix}{'•' * 6}{value[-4:]}"


def _validate_api_key_prefix(provider: str, api_key: str) -> str:
    value = api_key.strip()
    if not value:
        return value

    expected_prefix = "sk-ant-" if provider == "anthropic" else "sk-"
    if not value.startswith(expected_prefix):
        provider_label = "Anthropic" if provider == "anthropic" else "OpenAI"
        click.echo(
            f"Warning: this doesn't look like an {provider_label} key "
            f"(expected prefix: {expected_prefix})."
        )
        if not _prompt_confirm("Continue anyway?", default=True):
            return _prompt_api_key(provider)
    return value


def _prompt_output_path() -> str:
    choice = _prompt_menu(
        "[1/5] Where should results be saved?",
        [
            ("default", "./output (default)"),
            ("custom", "Custom path..."),
        ],
        default_value="default",
    )
    if choice == "custom":
        custom = _prompt_text("Enter custom output path")
        return custom or "./output"
    return "./output"


def _prompt_ai_enabled() -> bool:
    choice = _prompt_menu(
        "[2/5] Enable AI-powered playbooks?",
        [("yes", "Yes"), ("no", "No")],
        default_value="yes",
    )
    return choice == "yes"


def _prompt_provider() -> str:
    return _prompt_menu(
        "[3/5] AI provider?",
        [("anthropic", "Anthropic (Claude)"), ("openai", "OpenAI")],
        default_value="anthropic",
    )


def _prompt_api_key(provider: str) -> str:
    value = _prompt_secret("[4/5] Paste your API key")
    click.echo(_render_api_key_preview(value, provider))
    return _validate_api_key_prefix(provider, value)


def _prompt_run_file_now() -> str | None:
    click.echo("[5/5] Process a file now?")
    click.echo("  Your CSV file should have these columns:")
    click.echo("    Required: user_id, order_date, revenue")
    click.echo("    Optional: order_id, is_refund")
    click.echo("")
    click.echo("  Example:")
    click.echo("  user_id,order_date,revenue")
    click.echo("  U001,2024-01-15,49.99")
    click.echo("  U001,2024-02-20,29.99")
    click.echo("  U002,2024-01-10,99.50")

    choice = _prompt_menu(
        "",
        [("run", "Enter path to CSV"), ("skip", "Skip for now")],
        default_value="skip",
    )
    if choice == "skip":
        return None

    while True:
        csv_path = _prompt_text("CSV path")
        if not csv_path:
            click.echo("No file path provided. Skipping file processing.")
            return None
        if Path(csv_path).exists():
            return csv_path

        click.echo(f"Input error: file does not exist: {csv_path}")
        next_action = _prompt_menu(
            "Path not found. What would you like to do?",
            [("retry", "Try another path"), ("skip", "Skip for now")],
            default_value="retry",
        )
        if next_action == "skip":
            return None


def _build_wizard_config() -> tuple[dict[str, Any], str | None]:
    output_path = _prompt_output_path()
    ai_enabled = _prompt_ai_enabled()

    provider = "anthropic"
    api_key = ""
    if ai_enabled:
        provider = _prompt_provider()
        api_key = _prompt_api_key(provider)

    run_path = _prompt_run_file_now()
    config: dict[str, Any] = {"output": {"path": output_path}, "ai": {"enabled": ai_enabled}}
    if ai_enabled:
        config["ai"]["provider"] = provider
        config["ai"]["api_key"] = api_key
    return config, run_path


def run_wizard() -> tuple[dict[str, Any], str | None]:
    """Run first-time setup wizard and return config and optional CSV path."""
    click.echo(click.style("+------------------------------+", fg="cyan"))
    click.echo(click.style("|       RETENSIOS SETUP        |", fg="cyan", bold=True))
    click.echo(click.style("+------------------------------+", fg="cyan"))
    click.echo(click.style("Welcome to RetensiOS!", fg="cyan", bold=True))
    click.echo("Guided setup in 5 steps.")
    click.echo("")
    return _build_wizard_config()


def run_repeat_menu() -> str:
    """Prompt for action when config already exists."""
    click.echo(f"Config found ({CONFIG_FILE})")
    click.echo("")
    return _prompt_menu(
        "What would you like to do?",
        [
            ("run_file", "Run a file"),
            ("change_settings", "Change settings"),
            ("view_settings", "View current settings"),
        ],
        default_value="run_file",
    )


def prompt_file_path() -> str:
    """Prompt user for CSV path."""
    return _prompt_text("Enter path to CSV")


def view_current_settings() -> None:
    """Print current settings with masked API key."""
    config = read_config()
    output_path = get_output_path_from_config()
    ai_enabled = is_ai_enabled_from_config()

    click.echo(click.style("Current settings:", bold=True))
    click.echo(f"- {click.style('Output path', bold=True)}: {output_path}")
    click.echo(f"- {click.style('AI enabled', bold=True)}: {'yes' if ai_enabled else 'no'}")

    ai_section = config.get("ai")
    provider = "not set"
    api_key = "not set"
    if isinstance(ai_section, dict):
        raw_provider = ai_section.get("provider")
        if isinstance(raw_provider, str) and raw_provider.strip().lower() in SUPPORTED_PROVIDERS:
            provider = raw_provider.strip().lower()
        raw_api_key = ai_section.get("api_key")
        if isinstance(raw_api_key, str) and raw_api_key.strip():
            api_key = mask_api_key(raw_api_key)

    click.echo(f"- {click.style('Provider', bold=True)}: {provider}")
    click.echo(f"- {click.style('API key', bold=True)}: {api_key}")


def save_wizard_config(config: dict[str, Any]) -> None:
    """Persist wizard config."""
    write_config(config)
    click.echo(f"Config saved to {CONFIG_FILE}")


def process_file_now(
    csv_path: str,
    *,
    run_pipeline,
    output_path: str,
    ai_enabled: bool,
    provider: str,
) -> None:
    """Run pipeline for a file path gathered from wizard prompts."""
    if not csv_path:
        click.echo("No file path provided. Skipping file processing.")
        return
    if not Path(csv_path).exists():
        click.echo(f"Input error: file does not exist: {csv_path}")
        return

    run_pipeline(
        input_csv=csv_path,
        advanced=False,
        today=None,
        include_refunds=False,
        no_ai=not ai_enabled,
        output=output_path,
        api_key=None,
        provider=provider,
    )
