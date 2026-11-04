"""Reference solution - I06 Automated visual report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lab.plotting import CATEGORICAL, apply_style  # noqa: E402

FIGURES = ["monthly_revenue", "revenue_by_product", "basket_distribution", "price_vs_quantity"]


def prepare(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["month"] = df["date"].dt.month
    return df


def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    return df.groupby("month")["revenue"].sum()


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def build_report(sales: pd.DataFrame, output_dir: str | Path) -> dict[str, dict]:
    """Create the four figures; return {name: {"path": Path, "title": str}}."""
    apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(sales)
    report: dict[str, dict] = {}

    monthly = monthly_revenue(df) / 1000
    share = monthly.loc[[11, 12]].sum() / monthly.sum()
    title = f"Novembre et décembre font {share:.0%} du chiffre d'affaires annuel"
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(monthly.index, monthly.values, marker="o", color=CATEGORICAL[0])
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Mois")
    ax.set_ylabel("Chiffre d'affaires (k€)")
    ax.set_title(title)
    report["monthly_revenue"] = {"path": _save(fig, output_dir / "monthly_revenue.png"), "title": title}

    by_product = (df.groupby("product")["revenue"].sum() / 1000).sort_values()
    title = f"{by_product.index[-1]} : {by_product.iloc[-1] / by_product.sum():.0%} du chiffre d'affaires"
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(by_product.index, by_product.values, color=CATEGORICAL[0])
    ax.bar_label(bars, fmt="%.0f k€", padding=3)
    ax.set_xlabel("Chiffre d'affaires (k€)")
    ax.set_title(title)
    report["revenue_by_product"] = {"path": _save(fig, output_dir / "revenue_by_product.png"), "title": title}

    median_line = df["revenue"].median()
    title = f"La moitié des lignes de vente valent moins de {median_line:.0f} €"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["revenue"], bins=40, color=CATEGORICAL[0], edgecolor="white")
    ax.axvline(median_line, color="#2b2b29", linestyle="--", linewidth=1)
    ax.set_xlabel("Montant d'une ligne de vente (€)")
    ax.set_ylabel("Nombre de lignes")
    ax.set_title(title)
    report["basket_distribution"] = {"path": _save(fig, output_dir / "basket_distribution.png"), "title": title}

    corr = df["unit_price"].corr(df["quantity"])
    title = f"Prix et quantité ne sont presque pas liés (r = {corr:.2f})"
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(df["unit_price"], df["quantity"] + (df.index % 7 - 3) * 0.04, s=6, alpha=0.3, color=CATEGORICAL[0])
    ax.set_xlabel("Prix unitaire (€)")
    ax.set_ylabel("Quantité (légèrement dispersée)")
    ax.set_title(title)
    report["price_vs_quantity"] = {"path": _save(fig, output_dir / "price_vs_quantity.png"), "title": title}
    return report


if __name__ == "__main__":
    result = build_report(pd.read_csv("datasets/generated/sales.csv"), "outputs/I06")
    for name, item in result.items():
        print(f"{name:<20} {item['path']}  |  {item['title']}")
