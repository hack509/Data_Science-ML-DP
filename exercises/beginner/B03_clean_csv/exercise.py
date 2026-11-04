"""B03 - Clean a CSV file with pandas. Complete the functions (see README.md)."""

from __future__ import annotations

import pandas as pd


def parse_price(value) -> float:
    """'19,90 EUR' -> 19.9 ; '39.9' -> 39.9 ; NaN -> NaN."""
    raise NotImplementedError("TODO")


def clean_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw sales table."""
    raise NotImplementedError("TODO")


def cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Return rows_before, rows_after, duplicates_removed and total_revenue."""
    raise NotImplementedError("TODO")
