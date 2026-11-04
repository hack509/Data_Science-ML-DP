"""I05 - Confidence interval and hypothesis test. Complete the functions (see README.md)."""

from __future__ import annotations

import numpy as np  # noqa: F401
import pandas as pd
from scipy import stats  # noqa: F401  (only stats.t.* functions are allowed)


def mean_confidence_interval(sample, confidence: float = 0.95) -> tuple[float, float, float]:
    raise NotImplementedError("TODO")


def bootstrap_ci(sample, n_boot: int = 5000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    raise NotImplementedError("TODO")


def welch_t_test(a, b) -> tuple[float, float, float]:
    raise NotImplementedError("TODO")


def cohens_d(a, b) -> float:
    raise NotImplementedError("TODO")


def compare_groups(df: pd.DataFrame, group_col: str, value_col: str, group_a, group_b, alpha: float = 0.05) -> dict:
    raise NotImplementedError("TODO")
