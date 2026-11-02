"""Reference solution - I01 Reusable data-processing script.

Usage:
    python solutions/intermediate/I01_data_processing_script/solution.py \
        --input datasets/raw/sales_raw.csv --output-dir outputs/I01 --by region
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("sales_pipeline")
REQUIRED_COLUMNS = {"order_id", "date", "region", "product", "quantity", "unit_price"}


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def _parse_price(value) -> float:
    if pd.isna(value):
        return float("nan")
    return float(str(value).strip().removesuffix("EUR").strip().replace(",", "."))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["product"] = out["product"].str.strip().str.capitalize()
    out["unit_price"] = out["unit_price"].map(_parse_price)
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    before = len(out)
    out = out.drop_duplicates(subset="order_id").dropna(subset=["date", "quantity", "unit_price"])
    out = out[out["quantity"] > 0].copy()
    out["quantity"] = out["quantity"].astype(int)
    out["revenue"] = (out["quantity"] * out["unit_price"]).round(2)
    logger.info("cleaning: %d -> %d rows", before, len(out))
    return out.sort_values("order_id").reset_index(drop=True)


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in df.columns:
        raise ValueError(f"cannot group by unknown column {by!r}")
    summary = (
        df.groupby(by)
        .agg(n_orders=("order_id", "count"), quantity=("quantity", "sum"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
    )
    summary["revenue"] = summary["revenue"].round(2)
    summary["share"] = (summary["revenue"] / summary["revenue"].sum()).round(4)
    return summary.reset_index()


def save(clean_df: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"clean": output_dir / "clean_sales.csv", "summary": output_dir / "summary.json"}
    clean_df.to_csv(paths["clean"], index=False, date_format="%Y-%m-%d")
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, ensure_ascii=False, indent=2)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean a sales CSV file and summarize it.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--by", default="region")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        clean_df = clean(load(args.input))
        summary = summarize(clean_df, args.by)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    paths = save(clean_df, summary, args.output_dir)
    print(f"{len(clean_df)} clean rows -> {paths['clean']}")
    print(f"summary by {args.by} -> {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
