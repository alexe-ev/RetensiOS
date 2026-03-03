from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfm_engine.reporting.csv_output import write_customer_csv, write_summary_csv


def _customer_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "recency_days": 2,
                "frequency": 5,
                "monetary": 1200.5,
                "r_score": 5,
                "f_score": 4,
                "m_score": 4,
                "segment": "Champions",
                "churn_risk": 12.34,
            },
            {
                "user_id": "u2",
                "recency_days": 20,
                "frequency": 1,
                "monetary": 150.0,
                "r_score": 1,
                "f_score": 1,
                "m_score": 2,
                "segment": "At Risk",
                "churn_risk": 78.9,
            },
        ]
    )


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": "Champions",
                "user_count": 1,
                "total_revenue": 1200.5,
                "revenue_share_pct": 88.89,
                "avg_recency_days": 2.0,
                "avg_frequency": 5.0,
                "avg_churn_risk": 12.34,
                "revenue_at_risk": 148.14,
            },
            {
                "segment": "At Risk",
                "user_count": 1,
                "total_revenue": 150.0,
                "revenue_share_pct": 11.11,
                "avg_recency_days": 20.0,
                "avg_frequency": 1.0,
                "avg_churn_risk": 78.9,
                "revenue_at_risk": 118.35,
            },
        ]
    )


def test_write_customer_csv_writes_expected_columns_rows_and_data(tmp_path: Path) -> None:
    df = _customer_df()
    file_path = Path(write_customer_csv(df, str(tmp_path)))
    reloaded = pd.read_csv(file_path)

    expected_columns = [
        "user_id",
        "recency_days",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "segment",
        "churn_risk",
    ]
    assert reloaded.columns.tolist() == expected_columns
    assert len(reloaded) == len(df)
    pd.testing.assert_frame_equal(reloaded, df, check_dtype=False)


def test_write_summary_csv_writes_expected_columns_rows_and_data(tmp_path: Path) -> None:
    summary = _summary_df()
    file_path = Path(write_summary_csv(summary, str(tmp_path)))
    reloaded = pd.read_csv(file_path)

    expected_columns = [
        "segment",
        "user_count",
        "total_revenue",
        "revenue_share_pct",
        "avg_recency_days",
        "avg_frequency",
        "avg_churn_risk",
        "revenue_at_risk",
    ]
    assert reloaded.columns.tolist() == expected_columns
    assert len(reloaded) == len(summary)
    pd.testing.assert_frame_equal(reloaded, summary, check_dtype=False)


def test_write_csv_uses_custom_output_directory(tmp_path: Path) -> None:
    custom_dir = tmp_path / "exports" / "snapshot"
    customer_path = Path(write_customer_csv(_customer_df(), str(custom_dir)))
    summary_path = Path(write_summary_csv(_summary_df(), str(custom_dir)))

    assert customer_path.parent == custom_dir
    assert summary_path.parent == custom_dir
    assert customer_path.exists()
    assert summary_path.exists()


def test_write_csv_overwrites_existing_files_without_error(tmp_path: Path) -> None:
    first_df = _customer_df()
    second_df = first_df.copy()
    second_df.loc[0, "monetary"] = 9999.0

    output_dir = str(tmp_path)
    customer_path = Path(write_customer_csv(first_df, output_dir))
    write_customer_csv(second_df, output_dir)

    reloaded = pd.read_csv(customer_path)
    assert float(reloaded.iloc[0]["monetary"]) == 9999.0


def test_written_csv_files_are_utf8_encoded(tmp_path: Path) -> None:
    df = _customer_df()
    df.loc[0, "segment"] = "Loyal Cafe"
    customer_path = Path(write_customer_csv(df, str(tmp_path)))

    file_bytes = customer_path.read_bytes()
    decoded = file_bytes.decode("utf-8")
    assert "Loyal Cafe" in decoded
