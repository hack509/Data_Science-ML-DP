"""I02 - Missing values and outliers. Complete the functions (see README.md)."""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd


def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    raise NotImplementedError("TODO")


def zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    raise NotImplementedError("TODO")


def clean_students(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raise NotImplementedError("TODO")
