from __future__ import annotations

import pandas as pd

from rfm_engine.core.rfm import calculate_rfm
from rfm_engine.core.scoring import score_rfm
from rfm_engine.core.segmentation import assign_segments


def test_iteration2_pipeline_rfm_scoring_segmentation_end_to_end() -> None:
    transactions = pd.DataFrame(
        [
            {"user_id": "u1", "order_id": "o1", "order_date": "2026-01-01", "revenue": 100.0},
            {"user_id": "u1", "order_id": "o2", "order_date": "2026-01-10", "revenue": 80.0},
            {"user_id": "u1", "order_id": "o3", "order_date": "2026-01-12", "revenue": 120.0},
            {"user_id": "u2", "order_id": "o4", "order_date": "2026-01-12", "revenue": 25.0},
            {"user_id": "u3", "order_id": "o5", "order_date": "2025-11-01", "revenue": 20.0},
            {"user_id": "u4", "order_id": "o6", "order_date": "2025-10-01", "revenue": 10.0},
        ]
    )
    transactions["order_date"] = pd.to_datetime(transactions["order_date"])

    rfm = calculate_rfm(transactions)
    scored = score_rfm(rfm)
    segmented = assign_segments(scored)

    assert len(segmented) == 4
    assert list(segmented.columns) == [
        "user_id",
        "recency_days",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "segment",
    ]
    assert segmented["segment"].notna().all()
    assert not (segmented["segment"] == "Unclassified").any()
    assert segmented["segment"].nunique() >= 2
