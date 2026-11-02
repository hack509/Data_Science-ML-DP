"""Reference solution - B03 Clean a CSV file with pandas."""

from __future__ import annotations

import pandas as pd


def parse_price(value) -> float:
    """'19,90 EUR' -> 19.9 ; 39.9 -> 39.9 ; missing -> NaN."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().removesuffix("EUR").strip().replace(",", ".")
    return float(text)


def clean_sales(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()  # never modify the caller's data
    df["product"] = df["product"].str.strip().str.capitalize()
    df["unit_price"] = df["unit_price"].map(parse_price)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.drop_duplicates(subset="order_id")
    df = df.dropna(subset=["date", "quantity", "unit_price"])
    df = df[df["quantity"] > 0]

    df["quantity"] = df["quantity"].astype(int)
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    return df.sort_values("order_id").reset_index(drop=True)


def cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    return {
        "rows_before": len(raw),
        "rows_after": len(clean),
        "duplicates_removed": int(raw.duplicated(subset="order_id").sum()),
        "total_revenue": round(float(clean["revenue"].sum()), 2),
    }


if __name__ == "__main__":
    raw = pd.read_csv("datasets/raw/sales_raw.csv")
    clean = clean_sales(raw)
    print(cleaning_report(raw, clean))
