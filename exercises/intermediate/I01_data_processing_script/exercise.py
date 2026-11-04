"""I01 - Reusable data-processing script. Complete the functions (see README.md).

Usage once completed:
    python exercises/intermediate/I01_data_processing_script/exercise.py \
        --input datasets/raw/sales_raw.csv --output-dir outputs/I01 --by region
"""

from __future__ import annotations

import argparse  # noqa: F401  (you will need it in main)
import json  # noqa: F401
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"order_id", "date", "region", "product", "quantity", "unit_price"}


def load(path: str | Path) -> pd.DataFrame:
    raise NotImplementedError("TODO")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TODO")


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    raise NotImplementedError("TODO")


def save(clean_df: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    raise NotImplementedError("TODO")


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    raise SystemExit(main())
