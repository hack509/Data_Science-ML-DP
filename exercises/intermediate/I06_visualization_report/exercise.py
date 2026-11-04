"""I06 - Automated visual report. Complete the functions (see README.md)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402,F401
import pandas as pd  # noqa: E402

FIGURES = ["monthly_revenue", "revenue_by_product", "basket_distribution", "price_vs_quantity"]


def prepare(sales: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TODO")


def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    raise NotImplementedError("TODO")


def build_report(sales: pd.DataFrame, output_dir: str | Path) -> dict[str, dict]:
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    result = build_report(pd.read_csv("datasets/generated/sales.csv"), "outputs/I06")
    for name, item in result.items():
        print(f"{name:<20} {item['path']}  |  {item['title']}")
