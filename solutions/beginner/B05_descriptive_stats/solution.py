"""Reference solution - B05 Descriptive statistics from scratch (no NumPy)."""

from __future__ import annotations

import math
from collections import Counter


def _check(values) -> list[float]:
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    return values


def mean(values) -> float:
    values = _check(values)
    return sum(values) / len(values)


def median(values) -> float:
    ordered = sorted(_check(values))
    n = len(ordered)
    middle = n // 2
    return ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def modes(values) -> list:
    counts = Counter(_check(values))
    best = max(counts.values())
    return sorted(v for v, c in counts.items() if c == best)


def variance(values, ddof: int = 0) -> float:
    values = _check(values)
    if len(values) - ddof <= 0:
        raise ValueError("not enough values for this ddof")
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - ddof)


def std(values, ddof: int = 0) -> float:
    return math.sqrt(variance(values, ddof))


def quantile(values, q: float) -> float:
    """Linear interpolation between closest ranks (NumPy's default method)."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(_check(values))
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def iqr(values) -> float:
    return quantile(values, 0.75) - quantile(values, 0.25)


def covariance(x, y, ddof: int = 1) -> float:
    x, y = _check(x), _check(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - ddof)


def correlation(x, y) -> float:
    return covariance(x, y, ddof=1) / (std(x, ddof=1) * std(y, ddof=1))


def zscores(values) -> list[float]:
    m, s = mean(values), std(values)
    if s == 0:
        raise ValueError("standard deviation is zero")
    return [(v - m) / s for v in values]
