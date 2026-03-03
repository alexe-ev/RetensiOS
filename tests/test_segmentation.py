from __future__ import annotations

import pandas as pd

from rfm_engine.core.segmentation import SEGMENT_RULES, assign_segments


def _segment_for_score(r_score: int, f_score: int, m_score: int) -> str:
    df = pd.DataFrame(
        [{"user_id": "u1", "r_score": r_score, "f_score": f_score, "m_score": m_score}]
    )
    result = assign_segments(df)
    return str(result.iloc[0]["segment"])


def test_assign_segments_champions_case() -> None:
    assert _segment_for_score(5, 5, 5) == "Champions"


def test_assign_segments_lost_case() -> None:
    assert _segment_for_score(1, 1, 1) == "Lost"


def test_assign_segments_at_risk_case() -> None:
    assert _segment_for_score(2, 4, 4) == "At Risk"


def test_assign_segments_new_case() -> None:
    assert _segment_for_score(5, 1, 1) == "New"


def test_assign_segments_covers_all_125_combinations_without_unclassified() -> None:
    rows = [
        {
            "user_id": f"u-{r_score}{f_score}{m_score}",
            "r_score": r_score,
            "f_score": f_score,
            "m_score": m_score,
        }
        for r_score in range(1, 6)
        for f_score in range(1, 6)
        for m_score in range(1, 6)
    ]
    df = pd.DataFrame(rows)

    result = assign_segments(df)

    assert len(result) == 125
    assert not result["segment"].isna().any()
    assert not (result["segment"] == "Unclassified").any()


def test_assign_segments_is_deterministic() -> None:
    df = pd.DataFrame(
        [
            {"user_id": "u1", "r_score": 5, "f_score": 5, "m_score": 5},
            {"user_id": "u2", "r_score": 2, "f_score": 4, "m_score": 4},
            {"user_id": "u3", "r_score": 3, "f_score": 2, "m_score": 1},
            {"user_id": "u4", "r_score": 1, "f_score": 1, "m_score": 1},
        ]
    )

    result_1 = assign_segments(df)
    result_2 = assign_segments(df)

    pd.testing.assert_frame_equal(result_1, result_2)


def test_assign_segments_boundary_scores_are_consistent() -> None:
    assert _segment_for_score(3, 3, 3) == "Promising"
    assert _segment_for_score(4, 3, 3) == "Potential Loyalists"
    assert _segment_for_score(3, 4, 3) == "Loyal"


def test_segment_rules_table_has_all_expected_segment_names() -> None:
    expected = {
        "Champions",
        "Loyal",
        "Potential Loyalists",
        "New",
        "Promising",
        "At Risk",
        "Hibernating",
        "Lost",
    }
    actual = {rule.segment for rule in SEGMENT_RULES}
    assert actual == expected
