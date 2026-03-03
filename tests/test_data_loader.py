from __future__ import annotations

import logging

import pandas as pd
import pytest

from rfm_engine.core.data_loader import load_csv


def test_load_csv_valid_input_parses_types(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text(
        "user_id,order_date,revenue\n"
        "u1,2026-01-01,100\n"
        "u2,2026-01-02,200.5\n",
        encoding="utf-8",
    )

    df = load_csv(str(csv_path))

    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert pd.api.types.is_float_dtype(df["revenue"])
    assert len(df) == 2


def test_load_csv_raises_on_missing_required_columns(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "missing_user_id.csv"
    csv_path.write_text("order_date,revenue\n2026-01-01,100\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns: user_id"):
        load_csv(str(csv_path))


def test_load_csv_raises_on_empty_file(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Input CSV is empty"):
        load_csv(str(csv_path))


def test_load_csv_excludes_negative_revenue_by_default(
    tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "negative_revenue.csv"
    csv_path.write_text(
        "user_id,order_date,revenue\n"
        "u1,2026-01-01,100\n"
        "u2,2026-01-02,-50\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        df = load_csv(str(csv_path))

    assert len(df) == 1
    assert (df["revenue"] >= 0).all()
    assert "negative revenue" in caplog.text


def test_load_csv_keeps_negative_revenue_when_include_refunds(
    tmp_path: pytest.TempPathFactory,
) -> None:
    csv_path = tmp_path / "include_refunds.csv"
    csv_path.write_text(
        "user_id,order_date,revenue\n"
        "u1,2026-01-01,100\n"
        "u2,2026-01-02,-50\n",
        encoding="utf-8",
    )

    df = load_csv(str(csv_path), include_refunds=True)

    assert len(df) == 2
    assert (df["revenue"] < 0).any()


def test_load_csv_drops_invalid_dates_with_warning(
    tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "invalid_dates.csv"
    csv_path.write_text(
        "user_id,order_date,revenue\n"
        "u1,invalid-date,100\n"
        "u2,2026-01-02,200\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        df = load_csv(str(csv_path))

    assert len(df) == 1
    assert df.iloc[0]["user_id"] == "u2"
    assert "invalid order_date" in caplog.text


def test_load_csv_preserves_optional_columns_and_trims_whitespace(
    tmp_path: pytest.TempPathFactory,
) -> None:
    csv_path = tmp_path / "optional_columns.csv"
    csv_path.write_text(
        " user_id , order_date , revenue , order_id , is_refund \n"
        "u1,2026-01-01,100,ord-1,false\n",
        encoding="utf-8",
    )

    df = load_csv(str(csv_path))

    assert {"order_id", "is_refund"}.issubset(df.columns)
    assert {"user_id", "order_date", "revenue"}.issubset(df.columns)


def test_load_csv_raises_when_no_valid_rows_after_cleaning(
    tmp_path: pytest.TempPathFactory,
) -> None:
    csv_path = tmp_path / "no_valid_rows.csv"
    csv_path.write_text(
        "user_id,order_date,revenue\n"
        "u1,invalid-date,-100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No valid rows remain after cleaning"):
        load_csv(str(csv_path))
