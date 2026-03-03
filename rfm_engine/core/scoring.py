"""RFM scoring logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("recency_days", "frequency", "monetary")


def _rank_based_score(values: pd.Series, *, invert: bool) -> pd.Series:
    """Create deterministic 1-5 scores using rank-based binning."""
    if len(values) == 1:
        return pd.Series([3], index=values.index, dtype=int)

    ranks = values.rank(method="first", ascending=True)
    percentile = ranks / len(values)
    scores = np.ceil(percentile * 5).clip(1, 5).astype(int)
    score_series = pd.Series(scores, index=values.index)

    if invert:
        score_series = 6 - score_series

    return score_series.astype(int)


def _score_metric(values: pd.Series, *, invert: bool) -> pd.Series:
    """Score a metric with qcut and deterministic fallback."""
    if len(values) == 1:
        return pd.Series([3], index=values.index, dtype=int)

    labels = [5, 4, 3, 2, 1] if invert else [1, 2, 3, 4, 5]
    try:
        binned = pd.qcut(values, q=5, labels=labels, duplicates="drop")
        if len(binned.cat.categories) == 5:
            return binned.astype(int)
    except ValueError:
        pass

    return _rank_based_score(values, invert=invert)


def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Add R/F/M 1-5 scores to an RFM dataframe."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for scoring: {missing}")

    scored = df.copy()
    scored["r_score"] = _score_metric(scored["recency_days"], invert=True).astype(int)
    scored["f_score"] = _score_metric(scored["frequency"], invert=False).astype(int)
    scored["m_score"] = _score_metric(scored["monetary"], invert=False).astype(int)
    return scored
