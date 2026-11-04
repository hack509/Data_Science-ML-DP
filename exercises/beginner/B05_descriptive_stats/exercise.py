"""B05 - Descriptive statistics from scratch. Only `math` and `collections` are allowed."""

from __future__ import annotations

import math  # noqa: F401
from collections import Counter  # noqa: F401


def mean(values) -> float:
    raise NotImplementedError("TODO")


def median(values) -> float:
    raise NotImplementedError("TODO")


def modes(values) -> list:
    raise NotImplementedError("TODO")


def variance(values, ddof: int = 0) -> float:
    raise NotImplementedError("TODO")


def std(values, ddof: int = 0) -> float:
    raise NotImplementedError("TODO")


def quantile(values, q: float) -> float:
    raise NotImplementedError("TODO")


def iqr(values) -> float:
    raise NotImplementedError("TODO")


def covariance(x, y, ddof: int = 1) -> float:
    raise NotImplementedError("TODO")


def correlation(x, y) -> float:
    raise NotImplementedError("TODO")


def zscores(values) -> list[float]:
    raise NotImplementedError("TODO")
