import math

import pandas as pd
import pytest

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.fixture(scope="module")
def raw():
    return load_csv("sales_raw")


@pytest.fixture(scope="module")
def clean(raw):
    return ex.clean_sales(raw)


@pytest.mark.parametrize("value, expected", [("19,90 EUR", 19.9), ("39.9", 39.9), (39.9, 39.9), (" 24,90 EUR ", 24.9)])
def test_parse_price(value, expected):
    assert ex.parse_price(value) == pytest.approx(expected)


def test_parse_price_missing():
    assert math.isnan(ex.parse_price(float("nan")))


def test_raw_not_modified(raw):
    before = raw.copy()
    ex.clean_sales(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_types_and_values(clean):
    assert pd.api.types.is_integer_dtype(clean["quantity"])
    assert pd.api.types.is_float_dtype(clean["unit_price"])
    assert pd.api.types.is_datetime64_any_dtype(clean["date"])
    assert (clean["quantity"] > 0).all()
    assert clean[["date", "quantity", "unit_price"]].notna().all().all()
    assert sorted(clean["product"].unique()) == ["Casque", "Clavier", "Ecran", "Souris", "Webcam"]
    assert clean["order_id"].is_unique
    assert clean["order_id"].is_monotonic_increasing
    assert list(clean.index) == list(range(len(clean)))
    assert (clean["revenue"] - clean["quantity"] * clean["unit_price"]).abs().max() < 0.006


def test_report(raw, clean):
    report = ex.cleaning_report(raw, clean)
    assert report == {"rows_before": 410, "rows_after": 379, "duplicates_removed": 10, "total_revenue": 64939.08}
