"""Reference solution - B02 Sales records (standard library only)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_sales(path: str | Path) -> list[dict]:
    """Read the CSV file and convert quantity to int and unit_price to float."""
    with open(path, encoding="utf-8", newline="") as handle:
        records = []
        for row in csv.DictReader(handle):
            row["quantity"] = int(row["quantity"])
            row["unit_price"] = float(row["unit_price"])
            records.append(row)
    return records


def line_revenue(record: dict) -> float:
    return record["quantity"] * record["unit_price"]


def revenue_by(records: list[dict], key: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        totals[record[key]] += line_revenue(record)
    # Sorted by decreasing revenue; rounding only at the end.
    return {k: round(v, 2) for k, v in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)}


def top_products(records: list[dict], n: int = 3) -> list[tuple[str, int]]:
    quantities: dict[str, int] = defaultdict(int)
    for record in records:
        quantities[record["product"]] += record["quantity"]
    ranking = sorted(quantities.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranking[:n]


def monthly_revenue(records: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        totals[record["date"][:7]] += line_revenue(record)  # "2024-03-15" -> "2024-03"
    return {month: round(totals[month], 2) for month in sorted(totals)}


if __name__ == "__main__":
    sales = load_sales("datasets/generated/sales.csv")
    print(revenue_by(sales, "region"))
    print(top_products(sales, 3))
