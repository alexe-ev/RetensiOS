"""AI output generation entrypoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from rfm_engine.ai.prompts import SYSTEM_PROMPT, build_segment_prompt
from rfm_engine.ai.providers import get_provider
from rfm_engine.config import (
    DEFAULT_PROVIDER,
    get_api_key_from_config,
    get_provider_from_config,
)

logger = logging.getLogger(__name__)


def detect_provider(cli_provider: Optional[str] = None) -> str:
    """Resolve provider from cli flag, config, then default."""
    if cli_provider:
        return cli_provider.strip().lower()
    return get_provider_from_config()


def _provider_env_var(provider: str) -> str:
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def detect_api_key_with_source(
    cli_api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve API key from cli flag, env, then global config."""
    if cli_api_key:
        return cli_api_key, "cli"

    provider_name = (provider or DEFAULT_PROVIDER).strip().lower()
    env_var = _provider_env_var(provider_name)
    env_key = os.environ.get(env_var)
    if env_key:
        return env_key, f"env:{env_var}"

    config_key = get_api_key_from_config()
    if config_key:
        return config_key, "config"

    return None, None


def detect_api_key(
    cli_api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[str]:
    """Resolve API key from cli flag, env, then global config."""
    api_key, _source = detect_api_key_with_source(cli_api_key=cli_api_key, provider=provider)
    return api_key


def generate_playbooks(
    segment_summary: pd.DataFrame,
    output_dir: str,
    api_key: str,
    provider_name: str = DEFAULT_PROVIDER,
    suffix: str = "",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Optional[str]:
    """Generate segment playbooks markdown file, returning file path or None."""
    if segment_summary.empty:
        logger.warning("AI generation skipped: segment summary is empty.")
        return None

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"segment_playbooks_{suffix}.md" if suffix else "segment_playbooks.md"
        playbook_path = output_path / filename

        provider = get_provider(provider_name)

        sections: list[str] = []
        success_count = 0

        total_segments = len(segment_summary)
        for index, (_, row) in enumerate(segment_summary.iterrows(), start=1):
            segment_name = str(row.get("segment", "Unknown Segment"))
            if progress_callback is not None:
                progress_callback(index, total_segments, segment_name)
            prompt = build_segment_prompt(
                segment_name=segment_name,
                user_count=int(row.get("user_count", 0)),
                revenue_share=float(row.get("revenue_share_pct", 0.0)),
                avg_churn_risk=float(row.get("avg_churn_risk", 0.0)),
                avg_recency=float(row.get("avg_recency_days", 0.0)),
                avg_frequency=float(row.get("avg_frequency", 0.0)),
            )

            try:
                response_text = provider.call(
                    api_key=api_key,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("AI generation failed for segment '%s': %s", segment_name, exc)
                response_text = "_Playbook generation failed for this segment._"

            sections.append(f"## {segment_name}\n\n{response_text}")

        if success_count == 0:
            logger.warning("AI generation failed for all segments; playbooks file was not created.")
            return None

        markdown = "# Segment Playbooks\n\n" + "\n\n".join(sections).strip() + "\n"
        playbook_path.write_text(markdown, encoding="utf-8")
        return str(playbook_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI generation failed due to unrecoverable error: %s", exc)
        return None
