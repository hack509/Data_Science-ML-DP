import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from house_prices.modeling import predict, run  # noqa: E402


def test_quick_run_and_prediction(tmp_path):
    result = run(tmp_path, quick=True)
    metrics = result["metrics"]
    assert set(result["cv_table"].index) == {"ridge_log", "random_forest", "gradient_boosting_log"}
    assert metrics["test_mae"] < 35 and metrics["test_r2"] > 0.9
    assert metrics["test_mae"] < metrics["baseline_mae"] / 3
    low, high = metrics["test_mae_ci95"]
    assert low < metrics["test_mae"] < high
    assert all(p.exists() for p in result["paths"].values())

    houses = pd.DataFrame([
        {"surface_m2": 70, "rooms": 3, "age_years": 15, "distance_center_km": 4.0, "has_garden": 1, "neighborhood": "Residentiel", "energy_class": "C"},
        {"surface_m2": 900, "rooms": 3, "age_years": 15, "distance_center_km": 4.0, "has_garden": 1, "neighborhood": "Inconnu", "energy_class": "C"},
    ])
    predictions, warnings = predict(tmp_path, houses)
    assert 150 < predictions.loc[0, "predicted_price_keur"] < 400
    assert any("surface_m2" in w for w in warnings)
