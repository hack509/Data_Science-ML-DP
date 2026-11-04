"""B04 - Import and transform JSON data. Complete the functions (see README.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["customer_id", "city", "order_id", "date", "product", "quantity", "unit_price", "line_total"]


def load_customers(path: str | Path) -> list[dict]:
    raise NotImplementedError("TODO")


def flatten_orders(customers: list[dict]) -> pd.DataFrame:
    """One row per ordered item, columns = COLUMNS."""
    raise NotImplementedError("TODO")


def customer_summary(customers: list[dict]) -> pd.DataFrame:
    """One row per customer: n_orders, total_spent, favorite_product."""
    raise NotImplementedError("TODO")
