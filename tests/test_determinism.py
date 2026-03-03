from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfm_engine.core.churn import calculate_churn_risk
from rfm_engine.core.data_loader import load_csv
from rfm_engine.core.profiler import profile_data
from rfm_engine.core.rfm import calculate_rfm
from rfm_engine.core.scoring import score_rfm
from rfm_engine.core.segmentation import assign_segments
from rfm_engine.reporting.report_builder import build_report
from rfm_engine.reporting.revenue_analysis import analyze_revenue


def _run_full_pipeline(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    transactions = load_csv(str(input_csv))
    profile = profile_data(transactions)
    rfm = calculate_rfm(transactions)
    scored = score_rfm(rfm)
    segmented = assign_segments(scored)
    customers = calculate_churn_risk(segmented).sort_values("user_id").reset_index(drop=True)
    summary, concentration_index = analyze_revenue(customers)
    report = build_report(profile, summary, concentration_index)
    return customers, summary, report


def test_full_pipeline_is_deterministic_for_golden_dataset(golden_input_path: Path) -> None:
    customers_1, summary_1, report_1 = _run_full_pipeline(golden_input_path)
    customers_2, summary_2, report_2 = _run_full_pipeline(golden_input_path)

    pd.testing.assert_frame_equal(customers_1, customers_2)
    pd.testing.assert_frame_equal(summary_1, summary_2)
    assert report_1 == report_2
