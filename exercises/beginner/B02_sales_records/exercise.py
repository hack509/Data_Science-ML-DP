"""B02 - Sales records. Complete the functions (see README.md). Standard library only."""

from __future__ import annotations

from pathlib import Path


def load_sales(path: str | Path) -> list[dict]:
    """Read the CSV file; quantity -> int, unit_price -> float."""
    raise NotImplementedError("TODO")


def revenue_by(records: list[dict], key: str) -> dict[str, float]:
    """Total revenue per value of `key`, sorted by decreasing revenue."""
    raise NotImplementedError("TODO")


def top_products(records: list[dict], n: int = 3) -> list[tuple[str, int]]:
    """The n best-selling products by quantity."""
    raise NotImplementedError("TODO")


def monthly_revenue(records: list[dict]) -> dict[str, float]:
    """Revenue per month 'YYYY-MM', sorted by month."""
    raise NotImplementedError("TODO")
