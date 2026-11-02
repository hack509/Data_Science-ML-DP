"""Reference solution - B04 Import and transform JSON data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = ["customer_id", "city", "order_id", "date", "product", "quantity", "unit_price", "line_total"]


def load_customers(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def flatten_orders(customers: list[dict]) -> pd.DataFrame:
    """One row per ordered item, with customer and order information."""
    rows = []
    for customer in customers:
        for order in customer["orders"]:
            for item in order["items"]:
                rows.append(
                    {
                        "customer_id": customer["customer_id"],
                        "city": customer["city"],
                        "order_id": order["order_id"],
                        "date": order["date"],
                        "product": item["product"],
                        "quantity": item["quantity"],
                        "unit_price": item["unit_price"],
                        "line_total": round(item["quantity"] * item["unit_price"], 2),
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def customer_summary(customers: list[dict]) -> pd.DataFrame:
    """One row per customer, including customers without any order."""
    items = flatten_orders(customers)
    ids = [c["customer_id"] for c in customers]
    n_orders = pd.Series({c["customer_id"]: len(c["orders"]) for c in customers})
    spent = items.groupby("customer_id")["line_total"].sum().reindex(ids, fill_value=0.0).round(2)
    favorite = (
        items.groupby(["customer_id", "product"])["quantity"].sum()
        .reset_index()
        .sort_values(["customer_id", "quantity", "product"], ascending=[True, False, True])
        .drop_duplicates("customer_id")
        .set_index("customer_id")["product"]
        .reindex(ids)
    )
    summary = pd.DataFrame({"n_orders": n_orders, "total_spent": spent, "favorite_product": favorite})
    summary.index.name = "customer_id"
    return summary


if __name__ == "__main__":
    customers = load_customers("datasets/raw/orders.json")
    print(customer_summary(customers).sort_values("total_spent", ascending=False).head())
