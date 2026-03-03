from __future__ import annotations

import pandas as pd

from rfm_engine.core.scoring import score_rfm


def _make_rfm_df(
    recency: list[int],
    frequency: list[int],
    monetary: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(1, len(recency) + 1)],
            "recency_days": recency,
            "frequency": frequency,
            "monetary": monetary,
        }
    )


def test_score_rfm_standard_case_scores_are_in_1_to_5_range() -> None:
    df = _make_rfm_df(
        recency=list(range(1, 21)),
        frequency=list(range(1, 21)),
        monetary=[float(value * 10) for value in range(1, 21)],
    )

    result = score_rfm(df)

    for column in ["r_score", "f_score", "m_score"]:
        assert pd.api.types.is_integer_dtype(result[column])
        assert result[column].between(1, 5).all()
        counts = result[column].value_counts()
        assert len(counts) == 5


def test_score_rfm_inverts_recency_direction() -> None:
    df = _make_rfm_df(
        recency=[1, 5, 10, 20, 40],
        frequency=[1, 2, 3, 4, 5],
        monetary=[10.0, 20.0, 30.0, 40.0, 50.0],
    )

    result = score_rfm(df).set_index("user_id")

    assert int(result.loc["u1", "r_score"]) == 5
    assert int(result.loc["u5", "r_score"]) == 1


def test_score_rfm_frequency_direction_high_value_gets_high_score() -> None:
    df = _make_rfm_df(
        recency=[10, 9, 8, 7, 6],
        frequency=[1, 2, 3, 4, 100],
        monetary=[10.0, 20.0, 30.0, 40.0, 50.0],
    )

    result = score_rfm(df).set_index("user_id")

    assert int(result.loc["u5", "f_score"]) == 5
    assert int(result.loc["u1", "f_score"]) == 1


def test_score_rfm_low_variance_uses_rank_fallback_without_crash() -> None:
    df = _make_rfm_df(
        recency=list(range(1, 11)),
        frequency=[1] * 10,
        monetary=[50.0] * 10,
    )

    result = score_rfm(df)

    assert result["f_score"].between(1, 5).all()
    assert result["m_score"].between(1, 5).all()


def test_score_rfm_is_deterministic_with_tied_values() -> None:
    df = _make_rfm_df(
        recency=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        frequency=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        monetary=[100.0, 100.0, 80.0, 80.0, 60.0, 60.0, 40.0, 40.0, 20.0, 20.0],
    )

    result_1 = score_rfm(df)
    result_2 = score_rfm(df)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_score_rfm_minimum_dataset_of_five_users_gets_all_score_bins() -> None:
    df = _make_rfm_df(
        recency=[1, 2, 3, 4, 5],
        frequency=[10, 20, 30, 40, 50],
        monetary=[100.0, 200.0, 300.0, 400.0, 500.0],
    )

    result = score_rfm(df)

    assert sorted(result["r_score"].tolist()) == [1, 2, 3, 4, 5]
    assert sorted(result["f_score"].tolist()) == [1, 2, 3, 4, 5]
    assert sorted(result["m_score"].tolist()) == [1, 2, 3, 4, 5]


def test_score_rfm_single_user_returns_neutral_scores() -> None:
    df = _make_rfm_df(
        recency=[7],
        frequency=[3],
        monetary=[120.0],
    )

    result = score_rfm(df)

    assert int(result.iloc[0]["r_score"]) == 3
    assert int(result.iloc[0]["f_score"]) == 3
    assert int(result.iloc[0]["m_score"]) == 3
