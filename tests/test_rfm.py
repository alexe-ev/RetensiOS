from __future__ import annotations

from datetime import datetime

import pandas as pd

from rfm_engine.core.rfm import calculate_rfm


def _make_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def test_calculate_rfm_standard_case_returns_expected_values() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_id": "ord-1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u1", "order_id": "ord-2", "order_date": "2026-01-10", "revenue": 50.0},
            {"user_id": "u2", "order_id": "ord-3", "order_date": "2026-01-05", "revenue": 200.0},
            {"user_id": "u2", "order_id": "ord-4", "order_date": "2026-01-05", "revenue": 30.0},
            {"user_id": "u3", "order_id": "ord-5", "order_date": "2026-01-11", "revenue": 10.0},
        ]
    )

    actual = calculate_rfm(df).sort_values("user_id").reset_index(drop=True)

    expected = pd.DataFrame(
        [
            {"user_id": "u1", "recency_days": 2, "frequency": 2, "monetary": 150.0},
            {"user_id": "u2", "recency_days": 7, "frequency": 2, "monetary": 230.0},
            {"user_id": "u3", "recency_days": 1, "frequency": 1, "monetary": 10.0},
        ]
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert list(actual.columns) == ["user_id", "recency_days", "frequency", "monetary"]


def test_calculate_rfm_uses_default_today_date_when_not_provided() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 10.0},
            {"user_id": "u2", "order_date": "2026-01-03", "revenue": 20.0},
        ]
    )

    result = calculate_rfm(df).set_index("user_id")

    assert int(result.loc["u2", "recency_days"]) == 1


def test_calculate_rfm_honors_today_date_override() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u2", "order_date": "2026-01-10", "revenue": 200.0},
        ]
    )

    result = calculate_rfm(df, today_date=datetime(2026, 1, 20)).set_index("user_id")

    assert int(result.loc["u1", "recency_days"]) == 19
    assert int(result.loc["u2", "recency_days"]) == 10


def test_calculate_rfm_frequency_uses_order_id_nunique_when_present() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_id": "ord-1", "order_date": "2026-01-01", "revenue": 40.0},
            {"user_id": "u1", "order_id": "ord-1", "order_date": "2026-01-02", "revenue": 60.0},
        ]
    )

    result = calculate_rfm(df).set_index("user_id")

    assert int(result.loc["u1", "frequency"]) == 1


def test_calculate_rfm_frequency_falls_back_to_row_count_without_order_id() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_date": "2026-01-01", "revenue": 40.0},
            {"user_id": "u1", "order_date": "2026-01-02", "revenue": 60.0},
        ]
    )

    result = calculate_rfm(df).set_index("user_id")

    assert int(result.loc["u1", "frequency"]) == 2


def test_calculate_rfm_single_transaction_user_and_same_day_orders() -> None:
    df = _make_df(
        [
            {"user_id": "u1", "order_id": "ord-1", "order_date": "2026-01-05", "revenue": 50.0},
            {"user_id": "u2", "order_id": "ord-2", "order_date": "2026-01-10", "revenue": 20.0},
            {"user_id": "u2", "order_id": "ord-3", "order_date": "2026-01-10", "revenue": 30.0},
        ]
    )

    result = calculate_rfm(df).set_index("user_id")

    assert int(result.loc["u1", "recency_days"]) == 6
    assert int(result.loc["u1", "frequency"]) == 1
    assert int(result.loc["u2", "recency_days"]) == 1
    assert int(result.loc["u2", "frequency"]) == 2
