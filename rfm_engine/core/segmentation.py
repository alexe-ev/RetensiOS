"""Business segmentation mapping rules."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("r_score", "f_score", "m_score")


@dataclass(frozen=True)
class SegmentRule:
    """Single rule for mapping an RFM score tuple to a business segment."""

    segment: str
    r: tuple[int, int]
    f: tuple[int, int]
    m: tuple[int, int]


SEGMENT_RULES: tuple[SegmentRule, ...] = (
    SegmentRule(segment="Champions", r=(4, 5), f=(4, 5), m=(4, 5)),
    SegmentRule(segment="Loyal", r=(3, 5), f=(4, 5), m=(1, 5)),
    SegmentRule(segment="New", r=(5, 5), f=(1, 1), m=(1, 5)),
    SegmentRule(segment="Potential Loyalists", r=(4, 5), f=(1, 3), m=(1, 5)),
    SegmentRule(segment="Promising", r=(3, 3), f=(1, 3), m=(1, 5)),
    SegmentRule(segment="At Risk", r=(1, 2), f=(3, 5), m=(3, 5)),
    SegmentRule(segment="Lost", r=(1, 1), f=(1, 1), m=(1, 1)),
    SegmentRule(segment="Hibernating", r=(1, 2), f=(1, 5), m=(1, 5)),
)


def _in_range(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _segment_for_scores(r_score: int, f_score: int, m_score: int) -> str:
    for rule in SEGMENT_RULES:
        if (
            _in_range(r_score, rule.r)
            and _in_range(f_score, rule.f)
            and _in_range(m_score, rule.m)
        ):
            return rule.segment
    return "Unclassified"


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a business segment for each row based on RFM scores."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns for segmentation: {missing}")

    segmented = df.copy()
    segmented["segment"] = segmented.apply(
        lambda row: _segment_for_scores(
            int(row["r_score"]),
            int(row["f_score"]),
            int(row["m_score"]),
        ),
        axis=1,
    )

    unclassified_count = int((segmented["segment"] == "Unclassified").sum())
    if unclassified_count > 0:
        LOGGER.error(
            "Unmapped RFM combinations found during segmentation: %s row(s)",
            unclassified_count,
        )

    assert not segmented["segment"].isna().any(), "Segmentation produced NaN segment values."
    return segmented
