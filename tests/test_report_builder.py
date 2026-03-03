from __future__ import annotations

import pandas as pd

from rfm_engine.reporting.report_builder import build_report


def _profile() -> dict:
    return {
        "date_min": pd.Timestamp("2026-01-01"),
        "date_max": pd.Timestamp("2026-03-31"),
    }


def _segment_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": "Champions",
                "user_count": 120,
                "total_revenue": 12000.0,
                "revenue_share_pct": 48.0,
                "avg_churn_risk": 18.0,
                "avg_recency_days": 6.2,
                "avg_frequency": 7.0,
                "revenue_at_risk": 2160.0,
            },
            {
                "segment": "At Risk",
                "user_count": 80,
                "total_revenue": 9000.0,
                "revenue_share_pct": 36.0,
                "avg_churn_risk": 72.0,
                "avg_recency_days": 29.4,
                "avg_frequency": 2.1,
                "revenue_at_risk": 6480.0,
            },
            {
                "segment": "Potential Loyalists",
                "user_count": 100,
                "total_revenue": 4000.0,
                "revenue_share_pct": 16.0,
                "avg_churn_risk": 22.0,
                "avg_recency_days": 10.0,
                "avg_frequency": 3.8,
                "revenue_at_risk": 880.0,
            },
        ]
    )


def test_build_report_is_deterministic_for_same_input() -> None:
    profile = _profile()
    summary = _segment_summary()

    report_1 = build_report(profile, summary, concentration_index=84.0)
    report_2 = build_report(profile, summary, concentration_index=84.0)

    assert report_1 == report_2


def test_build_report_contains_all_required_sections() -> None:
    report = build_report(_profile(), _segment_summary(), concentration_index=84.0)

    assert "## Executive Summary" in report
    assert "## Revenue Distribution" in report
    assert "## Risk Exposure" in report
    assert "## Segment Overview Table" in report
    assert "## Strategic Priority" in report
    assert "## Quick Wins" in report


def test_build_report_segment_table_uses_markdown_table_syntax() -> None:
    report = build_report(_profile(), _segment_summary(), concentration_index=84.0)
    table_header = (
        "| Segment | Users | Revenue Share (%) | Avg Churn Risk (%) | "
        "Avg Recency (days) | Revenue at Risk |"
    )

    assert table_header in report
    assert "|---|---:|---:|---:|---:|---:|" in report
    assert "| Champions | 120 | 48.00 | 18.00 | 6.2 | $2,160.00 |" in report


def test_build_report_revenue_at_risk_values_match_summary() -> None:
    report = build_report(_profile(), _segment_summary(), concentration_index=84.0)

    assert "Total revenue at risk: $9,520.00." in report
    assert "- At Risk: $6,480.00 at risk (72.00% average risk)" in report


def test_build_report_contains_no_technical_jargon() -> None:
    report = build_report(_profile(), _segment_summary(), concentration_index=84.0).lower()

    forbidden_terms = ["quantile", "normalization", "r_score", "f_score", "m_score"]
    for term in forbidden_terms:
        assert term not in report


def test_build_report_handles_single_segment_dataset() -> None:
    summary = pd.DataFrame(
        [
            {
                "segment": "Only Segment",
                "user_count": 50,
                "total_revenue": 5000.0,
                "revenue_share_pct": 100.0,
                "avg_churn_risk": 30.0,
                "avg_recency_days": 15.0,
                "avg_frequency": 2.5,
                "revenue_at_risk": 1500.0,
            }
        ]
    )

    report = build_report(_profile(), summary, concentration_index=100.0)
    assert "Only Segment" in report
    assert "100.00% share" in report
