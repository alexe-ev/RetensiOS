"""Revenue exposure and concentration analysis."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = (
    "user_id",
    "segment",
    "monetary",
    "churn_risk",
    "recency_days",
    "frequency",
)


def analyze_revenue(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Compute segment-level revenue metrics and concentration index."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for revenue analysis: {missing}")

    grouped = (
        df.groupby("segment", as_index=False)
        .agg(
            user_count=("user_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_churn_risk=("churn_risk", "mean"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
        )
        .astype({"user_count": int})
    )

    grand_total = float(grouped["total_revenue"].sum())
    if grand_total == 0:
        grouped["revenue_share_pct"] = 0.0
    else:
        grouped["revenue_share_pct"] = (grouped["total_revenue"] / grand_total) * 100

    grouped["avg_churn_risk"] = grouped["avg_churn_risk"].round(2)
    grouped["avg_recency_days"] = grouped["avg_recency_days"].round(1)
    grouped["avg_frequency"] = grouped["avg_frequency"].round(1)
    grouped["revenue_share_pct"] = grouped["revenue_share_pct"].round(2)
    grouped["revenue_at_risk"] = (
        grouped["total_revenue"] * (grouped["avg_churn_risk"] / 100)
    ).round(2)

    summary = grouped.sort_values("revenue_share_pct", ascending=False).reset_index(drop=True)
    concentration_index = float(summary["revenue_share_pct"].head(2).sum().round(2))
    return summary, concentration_index
