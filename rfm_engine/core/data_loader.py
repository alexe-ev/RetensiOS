"""Data loading utilities for transaction CSV input."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"user_id", "order_date", "revenue"}


def load_csv(path: str, include_refunds: bool = False) -> pd.DataFrame:
    """Load and validate transaction input data from a CSV file.

    Args:
        path: Path to input CSV.
        include_refunds: Keep negative revenue rows when True.

    Returns:
        Cleaned dataframe ready for the next pipeline stage.

    Raises:
        ValueError: On schema issues, empty input, or empty result after cleaning.
    """
    try:
        df = pd.read_csv(Path(path))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty.") from exc

    if df.empty:
        raise ValueError("Input CSV is empty.")

    df.columns = [column.strip() for column in df.columns]

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    parsed_dates = pd.to_datetime(df["order_date"], format="mixed", errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())
    if invalid_date_count:
        logger.warning("Dropped %d rows with invalid order_date values.", invalid_date_count)
        df = df.loc[parsed_dates.notna()].copy()
        parsed_dates = parsed_dates.loc[parsed_dates.notna()]

    df["order_date"] = parsed_dates

    revenue = pd.to_numeric(df["revenue"], errors="coerce")
    invalid_revenue_count = int(revenue.isna().sum())
    if invalid_revenue_count:
        logger.warning("Dropped %d rows with invalid revenue values.", invalid_revenue_count)
        df = df.loc[revenue.notna()].copy()
        revenue = revenue.loc[revenue.notna()]

    df["revenue"] = revenue.astype(float)

    if not include_refunds:
        negative_mask = df["revenue"] < 0
        negative_count = int(negative_mask.sum())
        if negative_count:
            logger.warning(
                "Dropped %d rows with negative revenue. Use include_refunds=True to keep them.",
                negative_count,
            )
            df = df.loc[~negative_mask].copy()

    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")

    return df
