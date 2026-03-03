"""RFM metric calculations."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def calculate_rfm(
    df: pd.DataFrame,
    today_date: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute raw Recency, Frequency, Monetary values per user."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    effective_today = (
        pd.Timestamp(today_date)
        if today_date is not None
        else df["order_date"].max() + pd.Timedelta(days=1)
    )

    grouped = df.groupby("user_id")
    recency_days = (effective_today - grouped["order_date"].max()).dt.days.astype(int)
    monetary = grouped["revenue"].sum()

    if "order_id" in df.columns:
        frequency = grouped["order_id"].nunique()
    else:
        frequency = grouped.size()

    rfm = pd.DataFrame(
        {
            "user_id": recency_days.index,
            "recency_days": recency_days.values,
            "frequency": frequency.values,
            "monetary": monetary.values,
        }
    )
    return rfm[["user_id", "recency_days", "frequency", "monetary"]]
