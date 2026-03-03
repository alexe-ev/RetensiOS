"""CSV output writers for customer and segment deliverables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CUSTOMER_COLUMNS = [
    "user_id",
    "recency_days",
    "frequency",
    "monetary",
    "r_score",
    "f_score",
    "m_score",
    "segment",
    "churn_risk",
]

SUMMARY_COLUMNS = [
    "segment",
    "user_count",
    "total_revenue",
    "revenue_share_pct",
    "avg_recency_days",
    "avg_frequency",
    "avg_churn_risk",
    "revenue_at_risk",
]


def _ensure_columns(df: pd.DataFrame, required_columns: list[str], *, label: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for {label}: {missing}")


def _prepare_output_path(output_dir: str, filename: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / filename


def write_customer_csv(df: pd.DataFrame, output_dir: str, suffix: str = "") -> str:
    """Write customer-level retention output CSV and return file path."""
    _ensure_columns(df, CUSTOMER_COLUMNS, label="customers_rfm.csv")
    filename = f"customers_rfm_{suffix}.csv" if suffix else "customers_rfm.csv"
    file_path = _prepare_output_path(output_dir, filename)
    df.loc[:, CUSTOMER_COLUMNS].to_csv(file_path, index=False, encoding="utf-8")
    return str(file_path)


def write_summary_csv(segment_summary: pd.DataFrame, output_dir: str, suffix: str = "") -> str:
    """Write segment summary output CSV and return file path."""
    _ensure_columns(segment_summary, SUMMARY_COLUMNS, label="segments_summary.csv")
    filename = f"segments_summary_{suffix}.csv" if suffix else "segments_summary.csv"
    file_path = _prepare_output_path(output_dir, filename)
    segment_summary.loc[:, SUMMARY_COLUMNS].to_csv(file_path, index=False, encoding="utf-8")
    return str(file_path)
