"""Dataset profiling utilities."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def profile_data(df: pd.DataFrame) -> dict[str, Any]:
    """Build a diagnostic profile for a loaded transaction dataset."""
    total_rows = int(len(df))
    unique_users = int(df["user_id"].nunique())

    date_min = df["order_date"].min()
    date_max = df["order_date"].max()
    date_span_days = int((date_max - date_min).days)

    revenue_series = df["revenue"]
    revenue_min = float(revenue_series.min())
    revenue_max = float(revenue_series.max())
    revenue_mean = float(revenue_series.mean())
    revenue_median = float(revenue_series.median())
    revenue_total = float(revenue_series.sum())

    missing_counts = {column: int(count) for column, count in df.isna().sum().items()}
    duplicate_rows = int(df.duplicated().sum())

    warnings: list[str] = []
    if unique_users == 1:
        warnings.append("Dataset contains only 1 user.")

    transactions_per_user = df.groupby("user_id").size()
    single_txn_ratio = (
        float((transactions_per_user == 1).mean())
        if not transactions_per_user.empty
        else 0.0
    )
    if single_txn_ratio > 0.8:
        warnings.append("Over 80% of users have only one transaction.")

    if date_span_days < 30:
        warnings.append("Date range is less than 30 days - RFM scores may be unreliable.")

    if (revenue_median > 0 and revenue_max > 100 * revenue_median) or (
        revenue_median == 0 and revenue_max > 0
    ):
        warnings.append("Extreme revenue outlier detected.")

    profile = {
        "total_rows": total_rows,
        "unique_users": unique_users,
        "date_min": date_min,
        "date_max": date_max,
        "date_span_days": date_span_days,
        "revenue_min": revenue_min,
        "revenue_max": revenue_max,
        "revenue_mean": revenue_mean,
        "revenue_median": revenue_median,
        "revenue_total": revenue_total,
        "missing_counts": missing_counts,
        "duplicate_rows": duplicate_rows,
        "warnings": warnings,
    }

    logger.info("Dataset profile: rows=%d, unique_users=%d", total_rows, unique_users)
    logger.info(
        "Date range: %s -> %s (%d days)",
        date_min.date(),
        date_max.date(),
        date_span_days,
    )
    logger.info(
        "Revenue stats: min=%.2f max=%.2f mean=%.2f median=%.2f total=%.2f",
        revenue_min,
        revenue_max,
        revenue_mean,
        revenue_median,
        revenue_total,
    )
    logger.info("Missing values by column: %s", missing_counts)
    logger.info("Duplicate rows: %d", duplicate_rows)

    if warnings:
        for warning in warnings:
            logger.warning(warning)
    else:
        logger.info("No data profile warnings detected.")

    return profile
