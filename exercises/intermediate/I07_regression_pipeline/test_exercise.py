import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.fixture(scope="module")
def data():
    return ex.split(load_csv("house_prices"))


def test_split(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) == 640 and len(X_test) == 160
    assert ex.TARGET not in X_train.columns


def test_pipeline_and_metrics(data):
    X_train, X_test, y_train, y_test = data
    pipeline = ex.build_pipeline(LinearRegression()).fit(X_train, y_train)
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    metrics = ex.evaluate(pipeline, X_test, y_test)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["r2"] > 0.9
    assert metrics["rmse"] >= metrics["mae"]


def test_pipeline_is_robust(data):
    X_train, _, y_train, _ = data
    pipeline = ex.build_pipeline(LinearRegression()).fit(X_train, y_train)
    odd = pd.DataFrame([{
        "surface_m2": 75.0, "rooms": np.nan, "age_years": 20, "distance_center_km": 3.0,
        "has_garden": 0, "neighborhood": "Montagne", "energy_class": "C",
    }])
    assert np.isfinite(pipeline.predict(odd)).all()


def test_energy_order(data):
    X_train, _, y_train, _ = data
    pipeline = ex.build_pipeline(LinearRegression()).fit(X_train, y_train)
    encoder = pipeline.named_steps["preprocess"].named_transformers_["ord"]
    encoded = encoder.transform(pd.DataFrame({"energy_class": ["A", "D", "G"]}))
    assert np.asarray(encoded).ravel().tolist() == [0.0, 3.0, 6.0]


def test_compare_models(data):
    X_train, _, y_train, _ = data
    table = ex.compare_models(X_train, y_train, cv=3)
    assert set(table.index) == {"linear", "ridge", "random_forest"}
    assert list(table.columns) == ["mean_mae", "std_mae"]
    assert table["mean_mae"].is_monotonic_increasing
    assert table["mean_mae"].iloc[0] < 30
