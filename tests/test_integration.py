from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfm_engine.core.churn import calculate_churn_risk
from rfm_engine.core.data_loader import load_csv
from rfm_engine.core.rfm import calculate_rfm
from rfm_engine.core.scoring import score_rfm
from rfm_engine.core.segmentation import assign_segments
from rfm_engine.reporting.csv_output import write_customer_csv, write_summary_csv
from rfm_engine.reporting.revenue_analysis import analyze_revenue


def test_golden_dataset_pipeline_matches_expected_outputs(
    golden_input_path: Path,
    expected_customers: pd.DataFrame,
    expected_summary: pd.DataFrame,
    tmp_path: Path,
) -> None:
    transactions = load_csv(str(golden_input_path))
    rfm = calculate_rfm(transactions)
    scored = score_rfm(rfm)
    segmented = assign_segments(scored)
    customers = calculate_churn_risk(segmented).sort_values("user_id").reset_index(drop=True)
    summary, _ = analyze_revenue(customers)

    customers_path = write_customer_csv(customers, str(tmp_path))
    summary_path = write_summary_csv(summary, str(tmp_path))
    actual_customers = pd.read_csv(customers_path)
    actual_summary = pd.read_csv(summary_path)

    pd.testing.assert_frame_equal(
        actual_customers,
        expected_customers,
        check_dtype=False,
        atol=1e-2,
        rtol=0,
    )
    pd.testing.assert_frame_equal(
        actual_summary,
        expected_summary,
        check_dtype=False,
        atol=1e-2,
        rtol=0,
    )
