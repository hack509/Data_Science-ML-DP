import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from e2e_pipeline.config import load_config  # noqa: E402
from e2e_pipeline.data import DataQualityError, clean, quality_checks  # noqa: E402
from e2e_pipeline.pipeline import predict, train  # noqa: E402

from lab.data import load_csv  # noqa: E402

CONFIG = PROJECT_DIR / "config.json"


def test_config_validation():
    config = load_config(CONFIG)
    assert "math" not in config.features
    with pytest.raises(ValueError):
        load_config(CONFIG, numeric_features=["math"])
    with pytest.raises(ValueError):
        load_config(CONFIG, unknown_key=1)


def test_cleaning_keeps_missing_values_for_the_pipeline():
    cleaned, log = clean(load_csv("students_raw"))
    assert log["duplicates_removed"] == 8 and log["impossible_values_set_to_missing"] == 3
    assert cleaned["study_hours_week"].isna().sum() == 21


def test_quality_checks_detect_problems():
    config = load_config(CONFIG)
    with pytest.raises(DataQualityError, match="columns present"):
        quality_checks(pd.DataFrame({"student_id": ["S1"]}), config)
    dataset = pd.DataFrame({c: [1.0, 1.0] for c in config.features + ["passed"]}).assign(student_id=["S1", "S1"])
    with pytest.raises(DataQualityError, match="unique"):
        quality_checks(dataset, config)


def test_train_and_predict(tmp_path):
    config = load_config(CONFIG, n_bootstrap=200, cv_folds=3)
    result = train(config, tmp_path)
    metrics = result["metrics"]
    assert result["data_log"]["rows_labelled"] == 285  # 300 - 6 absent in math - 9 missing physics
    assert result["chosen"] != "baseline_majority"
    assert result["cv_table"].loc["baseline_majority", "cv_auc_mean"] == pytest.approx(0.5)
    assert metrics["roc_auc"] > 0.75
    low, high = metrics["ci95"]["roc_auc"]
    assert low <= metrics["roc_auc"] <= high
    assert set(metrics["subgroups"]) == {"A", "B", "C"}
    for name in ("model.joblib", "model_info.json", "metrics.json", "report.md", "model_card.md"):
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "model_info.json").read_text(encoding="utf-8"))["features"] == config.features

    new = pd.DataFrame([{"study_hours_week": 20, "attendance_rate": 0.95, "sleep_hours": 7.5, "previous_score": 15,
                         "volunteer_hours_year": 10, "group": "B", "club": "musique"},
                        {"study_hours_week": None, "attendance_rate": 0.6, "sleep_hours": 6, "previous_score": 5,
                         "volunteer_hours_year": 0, "group": "A", "club": "nouveau club"}])
    predictions = predict(tmp_path, new)
    assert predictions["pass_probability"].between(0, 1).all()
    assert predictions.loc[0, "pass_probability"] > predictions.loc[1, "pass_probability"]
