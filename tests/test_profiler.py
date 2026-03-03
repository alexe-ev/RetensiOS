from __future__ import annotations

import logging

import pandas as pd

from rfm_engine.core.profiler import profile_data


def _make_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def test_profile_data_standard_dataset_returns_expected_metrics() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u1", "order_date": "2026-01-10", "revenue": 50.0},
            {"user_id": "u2", "order_date": "2026-02-10", "revenue": 200.0},
        ]
    )

    profile = profile_data(df)

    assert profile["total_rows"] == 3
    assert profile["unique_users"] == 2
    assert profile["date_min"] == pd.Timestamp("2026-01-01")
    assert profile["date_max"] == pd.Timestamp("2026-02-10")
    assert profile["date_span_days"] == 40
    assert profile["revenue_min"] == 50.0
    assert profile["revenue_max"] == 200.0
    assert profile["revenue_mean"] == 350.0 / 3.0
    assert profile["revenue_median"] == 100.0
    assert profile["revenue_total"] == 350.0
    assert profile["missing_counts"] == {"user_id": 0, "order_date": 0, "revenue": 0}
    assert profile["duplicate_rows"] == 0


def test_profile_data_warns_when_dataset_has_only_one_user() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u1", "order_date": "2026-03-05", "revenue": 120.0},
        ]
    )

    profile = profile_data(df)

    assert "Dataset contains only 1 user." in profile["warnings"]


def test_profile_data_warns_when_single_transaction_users_dominate() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 10.0},
            {"user_id": "u2", "order_date": "2026-01-02", "revenue": 20.0},
            {"user_id": "u3", "order_date": "2026-01-03", "revenue": 30.0},
            {"user_id": "u4", "order_date": "2026-01-04", "revenue": 40.0},
            {"user_id": "u5", "order_date": "2026-01-05", "revenue": 50.0},
        ]
    )

    profile = profile_data(df)

    assert "Over 80% of users have only one transaction." in profile["warnings"]


def test_profile_data_warns_on_short_date_range() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u2", "order_date": "2026-01-15", "revenue": 200.0},
        ]
    )

    profile = profile_data(df)

    assert "Date range is less than 30 days - RFM scores may be unreliable." in profile["warnings"]


def test_profile_data_warns_on_extreme_revenue_outlier() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 1.0},
            {"user_id": "u2", "order_date": "2026-02-15", "revenue": 1.0},
            {"user_id": "u3", "order_date": "2026-03-10", "revenue": 200.0},
        ]
    )

    profile = profile_data(df)

    assert "Extreme revenue outlier detected." in profile["warnings"]


def test_profile_data_no_warnings_for_clean_dataset(caplog) -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u1", "order_date": "2026-01-15", "revenue": 120.0},
            {"user_id": "u2", "order_date": "2026-02-20", "revenue": 130.0},
            {"user_id": "u2", "order_date": "2026-03-10", "revenue": 110.0},
            {"user_id": "u3", "order_date": "2026-02-01", "revenue": 90.0},
            {"user_id": "u3", "order_date": "2026-03-20", "revenue": 95.0},
            {"user_id": "u4", "order_date": "2026-01-28", "revenue": 80.0},
            {"user_id": "u4", "order_date": "2026-03-28", "revenue": 85.0},
            {"user_id": "u5", "order_date": "2026-02-12", "revenue": 105.0},
            {"user_id": "u5", "order_date": "2026-03-30", "revenue": 115.0},
        ]
    )

    with caplog.at_level(logging.INFO):
        profile = profile_data(df)

    assert profile["warnings"] == []
    assert "No data profile warnings detected." in caplog.text
