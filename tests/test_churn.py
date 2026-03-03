from __future__ import annotations

import math

import pandas as pd
import pytest

from rfm_engine.core.churn import WEIGHTS, calculate_churn_risk


def _make_rfm_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_calculate_churn_risk_known_values_match_formula() -> None:
    df = _make_rfm_df(
        [
            {"user_id": "u1", "recency_days": 10, "frequency": 1, "monetary": 100.0},
            {"user_id": "u2", "recency_days": 5, "frequency": 5, "monetary": 500.0},
            {"user_id": "u3", "recency_days": 1, "frequency": 3, "monetary": 300.0},
        ]
    )

    result = calculate_churn_risk(df).set_index("user_id")

    assert float(result.loc["u1", "churn_risk"]) == pytest.approx(100.00, abs=0.01)
    assert float(result.loc["u2", "churn_risk"]) == pytest.approx(24.44, abs=0.01)
    assert float(result.loc["u3", "churn_risk"]) == pytest.approx(22.50, abs=0.01)


def test_calculate_churn_risk_scores_are_bounded_0_to_100() -> None:
    df = _make_rfm_df(
        [
            {"user_id": "u1", "recency_days": 1, "frequency": 10, "monetary": 900.0},
            {"user_id": "u2", "recency_days": 30, "frequency": 1, "monetary": 10.0},
            {"user_id": "u3", "recency_days": 15, "frequency": 4, "monetary": 250.0},
        ]
    )

    result = calculate_churn_risk(df)
    assert result["churn_risk"].between(0, 100).all()


def test_calculate_churn_risk_highest_and_lowest_risk_are_extremes() -> None:
    df = _make_rfm_df(
        [
            {"user_id": "high", "recency_days": 50, "frequency": 1, "monetary": 50.0},
            {"user_id": "mid", "recency_days": 20, "frequency": 2, "monetary": 200.0},
            {"user_id": "low", "recency_days": 1, "frequency": 10, "monetary": 1000.0},
        ]
    )

    result = calculate_churn_risk(df).set_index("user_id")
    assert float(result.loc["high", "churn_risk"]) == pytest.approx(100.0, abs=0.01)
    assert float(result.loc["low", "churn_risk"]) == pytest.approx(0.0, abs=0.01)


def test_calculate_churn_risk_zero_variance_recency_no_division_by_zero() -> None:
    df = _make_rfm_df(
        [
            {"user_id": "u1", "recency_days": 7, "frequency": 1, "monetary": 10.0},
            {"user_id": "u2", "recency_days": 7, "frequency": 2, "monetary": 20.0},
        ]
    )

    result = calculate_churn_risk(df).set_index("user_id")
    assert float(result.loc["u1", "churn_risk"]) == pytest.approx(45.0, abs=0.01)
    assert float(result.loc["u2", "churn_risk"]) == pytest.approx(0.0, abs=0.01)


def test_calculate_churn_risk_single_user_returns_zero_risk() -> None:
    df = _make_rfm_df(
        [{"user_id": "u1", "recency_days": 14, "frequency": 2, "monetary": 150.0}]
    )

    result = calculate_churn_risk(df)
    assert float(result.iloc[0]["churn_risk"]) == pytest.approx(0.0, abs=0.01)


def test_churn_weight_sanity_sum_is_one() -> None:
    weight_sum = WEIGHTS["recency"] + WEIGHTS["frequency"] + WEIGHTS["monetary"]
    assert math.isclose(weight_sum, 1.0)
