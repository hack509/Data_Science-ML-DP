import re

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.fixture(scope="module")
def sales():
    return load_csv("sales")


def test_prepare_and_monthly(sales):
    df = ex.prepare(sales)
    assert "revenue" in df and "month" in df
    assert "revenue" not in sales.columns, "prepare must not modify its input"
    monthly = ex.monthly_revenue(df)
    assert list(monthly.index) == list(range(1, 13))
    assert monthly.sum() == pytest.approx((sales["quantity"] * sales["unit_price"]).sum())


def test_report(sales, tmp_path):
    report = ex.build_report(sales, tmp_path / "report")
    assert set(report) == set(ex.FIGURES)
    assert plt.get_fignums() == [], "all figures must be closed"
    for name, item in report.items():
        with Image.open(item["path"]) as image:
            assert image.format == "PNG" and image.width >= 400, name
        assert re.search(r"\d", item["title"]), f"title of {name} should contain a computed number"
    assert "29" in report["monthly_revenue"]["title"]
    assert "Ecran" in report["revenue_by_product"]["title"]
    assert "0.0" in report["price_vs_quantity"]["title"] or "-0.0" in report["price_vs_quantity"]["title"]
