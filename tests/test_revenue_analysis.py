from __future__ import annotations

import pandas as pd
import pytest

from rfm_engine.reporting.revenue_analysis import analyze_revenue


def _make_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_analyze_revenue_known_dataset_share_sums_to_100() -> None:
    df = _make_df(
        [
            {
                "user_id": "u1",
                "segment": "A",
                "monetary": 8000.0,
                "churn_risk": 40.0,
                "recency_days": 10,
                "frequency": 6,
            },
            {
                "user_id": "u2",
                "segment": "A",
                "monetary": 2000.0,
                "churn_risk": 60.0,
                "recency_days": 20,
                "frequency": 4,
            },
            {
                "user_id": "u3",
                "segment": "B",
                "monetary": 1500.0,
                "churn_risk": 20.0,
                "recency_days": 8,
                "frequency": 5,
            },
            {
                "user_id": "u4",
                "segment": "B",
                "monetary": 500.0,
                "churn_risk": 40.0,
                "recency_days": 12,
                "frequency": 3,
            },
            {
                "user_id": "u5",
                "segment": "C",
                "monetary": 3000.0,
                "churn_risk": 0.0,
                "recency_days": 5,
                "frequency": 7,
            },
            {
                "user_id": "u6",
                "segment": "D",
                "monetary": 500.0,
                "churn_risk": 80.0,
                "recency_days": 30,
                "frequency": 1,
            },
        ]
    )

    summary, concentration_index = analyze_revenue(df)

    assert summary["revenue_share_pct"].sum() == pytest.approx(100.0, abs=0.05)
    assert concentration_index == pytest.approx(83.87, abs=0.01)


def test_analyze_revenue_revenue_at_risk_uses_avg_churn_risk() -> None:
    df = _make_df(
        [
            {
                "user_id": "u1",
                "segment": "Core",
                "monetary": 5000.0,
                "churn_risk": 40.0,
                "recency_days": 10,
                "frequency": 5,
            },
            {
                "user_id": "u2",
                "segment": "Core",
                "monetary": 5000.0,
                "churn_risk": 60.0,
                "recency_days": 15,
                "frequency": 4,
            },
        ]
    )

    summary, _ = analyze_revenue(df)
    core_row = summary.set_index("segment").loc["Core"]
    assert float(core_row["avg_churn_risk"]) == pytest.approx(50.0, abs=0.01)
    assert float(core_row["revenue_at_risk"]) == pytest.approx(5000.0, abs=0.01)


def test_analyze_revenue_concentration_index_reflects_high_concentration() -> None:
    df = _make_df(
        [
            {
                "user_id": "u1",
                "segment": "Dominant",
                "monetary": 800.0,
                "churn_risk": 10.0,
                "recency_days": 4,
                "frequency": 6,
            },
            {
                "user_id": "u2",
                "segment": "Long Tail",
                "monetary": 200.0,
                "churn_risk": 20.0,
                "recency_days": 10,
                "frequency": 2,
            },
        ]
    )

    _, concentration_index = analyze_revenue(df)
    assert concentration_index == pytest.approx(100.0, abs=0.01)


def test_analyze_revenue_single_segment_returns_full_share_without_error() -> None:
    df = _make_df(
        [
            {
                "user_id": "u1",
                "segment": "Only",
                "monetary": 100.0,
                "churn_risk": 15.0,
                "recency_days": 2,
                "frequency": 3,
            },
            {
                "user_id": "u2",
                "segment": "Only",
                "monetary": 300.0,
                "churn_risk": 25.0,
                "recency_days": 4,
                "frequency": 2,
            },
        ]
    )

    summary, concentration_index = analyze_revenue(df)
    row = summary.iloc[0]
    assert float(row["revenue_share_pct"]) == pytest.approx(100.0, abs=0.01)
    assert concentration_index == pytest.approx(100.0, abs=0.01)


def test_analyze_revenue_zero_churn_segment_has_zero_revenue_at_risk() -> None:
    df = _make_df(
        [
            {
                "user_id": "u1",
                "segment": "Safe",
                "monetary": 1000.0,
                "churn_risk": 0.0,
                "recency_days": 3,
                "frequency": 4,
            },
            {
                "user_id": "u2",
                "segment": "Safe",
                "monetary": 2000.0,
                "churn_risk": 0.0,
                "recency_days": 5,
                "frequency": 2,
            },
            {
                "user_id": "u3",
                "segment": "Risky",
                "monetary": 1000.0,
                "churn_risk": 50.0,
                "recency_days": 15,
                "frequency": 1,
            },
        ]
    )

    summary, _ = analyze_revenue(df)
    safe_row = summary.set_index("segment").loc["Safe"]
    assert float(safe_row["revenue_at_risk"]) == pytest.approx(0.0, abs=0.01)


def test_analyze_revenue_output_sorted_by_revenue_share_desc() -> None:
    df = _make_df(
        [
            {
                "user_id": "u1",
                "segment": "Mid",
                "monetary": 300.0,
                "churn_risk": 20.0,
                "recency_days": 8,
                "frequency": 3,
            },
            {
                "user_id": "u2",
                "segment": "High",
                "monetary": 500.0,
                "churn_risk": 30.0,
                "recency_days": 12,
                "frequency": 2,
            },
            {
                "user_id": "u3",
                "segment": "Low",
                "monetary": 200.0,
                "churn_risk": 10.0,
                "recency_days": 5,
                "frequency": 4,
            },
        ]
    )

    summary, _ = analyze_revenue(df)
    assert summary["revenue_share_pct"].tolist() == sorted(
        summary["revenue_share_pct"].tolist(),
        reverse=True,
    )
