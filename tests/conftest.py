from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def golden_dir() -> Path:
    return Path(__file__).parent / "golden"


@pytest.fixture()
def golden_input_path(golden_dir: Path) -> Path:
    return golden_dir / "golden_data.csv"


@pytest.fixture()
def golden_transactions(golden_input_path: Path) -> pd.DataFrame:
    return pd.read_csv(golden_input_path)


@pytest.fixture()
def expected_customers(golden_dir: Path) -> pd.DataFrame:
    return pd.read_csv(golden_dir / "expected_customers_rfm.csv")


@pytest.fixture()
def expected_summary(golden_dir: Path) -> pd.DataFrame:
    return pd.read_csv(golden_dir / "expected_segments_summary.csv")
