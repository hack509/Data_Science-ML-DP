import pytest

from lab.data import dataset_path, load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)

SMALL = [
    {"date": "2024-01-03", "region": "Nord", "product": "Souris", "quantity": 2, "unit_price": 10.0},
    {"date": "2024-01-20", "region": "Sud", "product": "Clavier", "quantity": 1, "unit_price": 40.0},
    {"date": "2024-02-02", "region": "Nord", "product": "Clavier", "quantity": 1, "unit_price": 40.0},
    {"date": "2024-02-10", "region": "Sud", "product": "Ecran", "quantity": 1, "unit_price": 100.0},
]


@pytest.fixture(scope="module")
def sales():
    return ex.load_sales(dataset_path("sales"))


def test_load_types(sales):
    assert len(sales) == 3000
    assert isinstance(sales[0]["quantity"], int)
    assert isinstance(sales[0]["unit_price"], float)


def test_revenue_by_small():
    assert ex.revenue_by(SMALL, "region") == {"Sud": 140.0, "Nord": 60.0}
    assert list(ex.revenue_by(SMALL, "region")) == ["Sud", "Nord"]


def test_top_products_tie_breaking():
    assert ex.top_products(SMALL, 2) == [("Clavier", 2), ("Souris", 2)]


def test_monthly_small():
    assert ex.monthly_revenue(SMALL) == {"2024-01": 60.0, "2024-02": 140.0}


def test_on_dataset(sales):
    assert ex.revenue_by(sales, "region") == {"Nord": 153939.0, "Sud": 134403.01, "Ouest": 128111.64, "Est": 114852.25}
    assert ex.top_products(sales, 3) == [("Souris", 2591), ("Clavier", 2323), ("Casque", 1819)]
    monthly = ex.monthly_revenue(sales)
    assert list(monthly) == sorted(monthly) and len(monthly) == 12
    df = load_csv("sales")
    assert sum(monthly.values()) == pytest.approx((df["quantity"] * df["unit_price"]).sum(), abs=0.1)
