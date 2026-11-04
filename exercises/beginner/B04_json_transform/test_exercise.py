import pandas as pd
import pytest

from lab.data import dataset_path
from lab.exercise_loader import load_target

ex = load_target(__file__)

TINY = [
    {"customer_id": "A", "city": "Lyon", "orders": [
        {"order_id": "W1", "date": "2024-01-01", "items": [
            {"product": "Souris", "quantity": 2, "unit_price": 24.9},
            {"product": "Clavier", "quantity": 2, "unit_price": 39.9},
        ]},
    ]},
    {"customer_id": "B", "city": "Lille", "orders": []},
]


@pytest.fixture(scope="module")
def customers():
    return ex.load_customers(dataset_path("orders_json"))


def test_flatten_tiny():
    items = ex.flatten_orders(TINY)
    assert list(items.columns) == ex.COLUMNS
    assert len(items) == 2
    assert items["line_total"].tolist() == [49.8, 79.8]


def test_summary_tiny_keeps_customers_without_orders():
    summary = ex.customer_summary(TINY)
    assert list(summary.index) == ["A", "B"]
    assert summary.index.name == "customer_id"
    assert summary.loc["A", "n_orders"] == 1
    assert summary.loc["A", "total_spent"] == pytest.approx(129.6)
    assert summary.loc["A", "favorite_product"] == "Clavier"  # tie broken alphabetically
    assert summary.loc["B", "n_orders"] == 0
    assert summary.loc["B", "total_spent"] == 0.0
    assert pd.isna(summary.loc["B", "favorite_product"])


def test_on_dataset(customers):
    assert len(customers) == 40
    assert len(ex.flatten_orders(customers)) == 124
    summary = ex.customer_summary(customers)
    assert len(summary) == 40
    assert int((summary["n_orders"] == 0).sum()) == 14
    best = summary["total_spent"].idxmax()
    assert best == "C024"
    assert summary.loc[best, "total_spent"] == pytest.approx(1331.1)
    assert summary.loc[best, "favorite_product"] == "Clavier"
