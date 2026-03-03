"""Global configuration helpers for retensios."""

from __future__ import annotations

import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "retensios"
CONFIG_FILE = CONFIG_DIR / "config.toml"
LEGACY_CONFIG_DIR = Path.home() / ".config" / "rfm-engine"
LEGACY_CONFIG_FILE = LEGACY_CONFIG_DIR / "config.toml"
DEFAULT_PROVIDER = "openai"
SUPPORTED_PROVIDERS = ("openai", "anthropic")
DEFAULT_OUTPUT_PATH = "outputs"


def migrate_legacy_config() -> None:
    """Copy legacy config into the new location when needed."""
    if CONFIG_FILE.exists():
        return
    if not LEGACY_CONFIG_FILE.exists():
        return

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0700
        shutil.copy2(LEGACY_CONFIG_FILE, CONFIG_FILE)
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        print(
            "Notice: migrated config from "
            f"{LEGACY_CONFIG_FILE} to {CONFIG_FILE}. "
            "Legacy directory was left untouched."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to migrate legacy config from %s to %s: %s",
            LEGACY_CONFIG_FILE,
            CONFIG_FILE,
            exc,
        )


def _serialize_config(data: dict[str, Any]) -> str:
    """Serialize config dictionary into minimal TOML."""
    lines: list[str] = []

    output_section = data.get("output")
    if isinstance(output_section, dict):
        output_path = output_section.get("path")
        if isinstance(output_path, str) and output_path.strip():
            escaped_path = output_path.strip().replace("\\", "\\\\").replace('"', '\\"')
            lines.extend(["[output]", f'path = "{escaped_path}"', ""])

    ai_section = data.get("ai")
    if isinstance(ai_section, dict):
        ai_lines: list[str] = ["[ai]"]

        enabled = ai_section.get("enabled")
        if isinstance(enabled, bool):
            ai_lines.append(f"enabled = {'true' if enabled else 'false'}")

        provider = ai_section.get("provider")
        if isinstance(provider, str) and provider.strip():
            escaped_provider = provider.strip().lower().replace("\\", "\\\\").replace('"', '\\"')
            ai_lines.append(f'provider = "{escaped_provider}"')

        api_key = ai_section.get("api_key")
        if isinstance(api_key, str) and api_key:
            escaped_key = api_key.replace("\\", "\\\\").replace('"', '\\"')
            ai_lines.append(f'api_key = "{escaped_key}"')

        if len(ai_lines) > 1:
            lines.extend(ai_lines)

    if not lines:
        return ""
    if lines[-1] == "":
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


def _parse_config(raw: str) -> dict[str, Any]:
    """Parse minimal TOML format used by the project."""
    try:
        import tomllib

        return tomllib.loads(raw)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Malformed config file at %s: %s", CONFIG_FILE, exc)
        return {}

    # Fallback parser for Python versions without tomllib.
    section: str | None = None
    parsed: dict[str, Any] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            parsed.setdefault(section, {})
            continue

        if "=" not in stripped or section is None:
            logger.warning("Malformed config file at %s: unsupported TOML format.", CONFIG_FILE)
            return {}

        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()

        parsed.setdefault(section, {})
        if not isinstance(parsed[section], dict):
            logger.warning("Malformed config file at %s: invalid section type.", CONFIG_FILE)
            return {}
        if value in {"true", "false"}:
            parsed[section][key] = value == "true"
            continue

        if len(value) < 2 or value[0] != '"' or value[-1] != '"':
            logger.warning("Malformed config file at %s: unsupported TOML value.", CONFIG_FILE)
            return {}

        unquoted = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        parsed[section][key] = unquoted

    return parsed


def read_config() -> dict[str, Any]:
    """Read config TOML and return dictionary."""
    migrate_legacy_config()
    if not CONFIG_FILE.exists():
        return {}

    try:
        raw = CONFIG_FILE.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read config file at %s: %s", CONFIG_FILE, exc)
        return {}

    if not raw.strip():
        return {}

    return _parse_config(raw)


def write_config(data: dict[str, Any]) -> None:
    """Write config TOML with secure permissions."""
    serialized = _serialize_config(data)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0700

    with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
        handle.write(serialized)
    os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600


def get_api_key_from_config() -> str | None:
    """Read ai.api_key from global config."""
    config = read_config()
    ai_section = config.get("ai")
    if not isinstance(ai_section, dict):
        return None

    api_key = ai_section.get("api_key")
    if not isinstance(api_key, str):
        return None

    stripped = api_key.strip()
    return stripped or None


def set_api_key_in_config(api_key: str) -> None:
    """Persist API key under ai.api_key."""
    config = read_config()
    ai_section = config.get("ai")
    if not isinstance(ai_section, dict):
        ai_section = {}

    ai_section["api_key"] = api_key
    config["ai"] = ai_section
    write_config(config)


def get_provider_from_config() -> str:
    """Read ai.provider from global config, defaulting to openai."""
    config = read_config()
    ai_section = config.get("ai")
    if not isinstance(ai_section, dict):
        return DEFAULT_PROVIDER

    provider = ai_section.get("provider")
    if not isinstance(provider, str):
        return DEFAULT_PROVIDER

    normalized = provider.strip().lower()
    if normalized in SUPPORTED_PROVIDERS:
        return normalized

    logger.warning(
        "Unknown provider '%s' in config file at %s. Falling back to '%s'.",
        provider,
        CONFIG_FILE,
        DEFAULT_PROVIDER,
    )
    return DEFAULT_PROVIDER


def set_provider_in_config(provider: str) -> None:
    """Persist ai.provider in global config."""
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown provider '{provider}'. Supported providers: {supported}.")

    config = read_config()
    ai_section = config.get("ai")
    if not isinstance(ai_section, dict):
        ai_section = {}

    ai_section["provider"] = normalized
    config["ai"] = ai_section
    write_config(config)


def is_ai_enabled_from_config() -> bool:
    """Read ai.enabled from global config, defaulting to True."""
    config = read_config()
    ai_section = config.get("ai")
    if not isinstance(ai_section, dict):
        return True

    enabled = ai_section.get("enabled")
    if isinstance(enabled, bool):
        return enabled
    return True


def get_output_path_from_config() -> str:
    """Read output.path from global config, with default fallback."""
    config = read_config()
    output_section = config.get("output")
    if not isinstance(output_section, dict):
        return DEFAULT_OUTPUT_PATH

    path = output_section.get("path")
    if not isinstance(path, str):
        return DEFAULT_OUTPUT_PATH

    normalized = path.strip()
    return normalized or DEFAULT_OUTPUT_PATH


def mask_api_key(api_key: str) -> str:
    """Mask API key for safe display."""
    value = api_key.strip()
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}...{value[-3:]}"
