"""Churn risk proxy calculations."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("recency_days", "frequency", "monetary")
WEIGHTS = {
    "recency": 0.55,
    "frequency": 0.30,
    "monetary": 0.15,
}


def _min_max_norm(series: pd.Series) -> pd.Series:
    """Normalize a numeric series with min-max scaling."""
    range_ = series.max() - series.min()
    if range_ == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / range_


def calculate_churn_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add an explainable churn risk score (0-100) to an RFM dataframe."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for churn risk: {missing}")

    scored = df.copy()
    if len(scored) == 1:
        scored["churn_risk"] = 0.0
        return scored

    recency_norm = _min_max_norm(scored["recency_days"])
    frequency_norm = _min_max_norm(scored["frequency"])
    monetary_norm = _min_max_norm(scored["monetary"])

    scored["churn_risk"] = 100 * (
        WEIGHTS["recency"] * recency_norm
        + WEIGHTS["frequency"] * (1 - frequency_norm)
        + WEIGHTS["monetary"] * (1 - monetary_norm)
    )
    scored["churn_risk"] = scored["churn_risk"].clip(0, 100).round(2)
    return scored
