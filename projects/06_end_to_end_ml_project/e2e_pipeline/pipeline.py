"""Orchestration: train (data -> model -> evaluation -> artefacts) and predict."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from .config import Config
from .data import load_dataset
from .evaluation import evaluate
from .modeling import select_model
from .reporting import write_model_card, write_report


def train(config: Config, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, data_log = load_dataset(config)
    X, y = dataset[config.features], dataset["passed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )
    chosen, model, cv_table = select_model(X_train, y_train, config)
    metrics = evaluate(model, X_test, y_test, config)
    versions = {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__}
    joblib.dump(model, output_dir / "model.joblib")
    artefact = {"model": chosen, "features": config.features, "decision_threshold": config.decision_threshold, "versions": versions,
                "config": config.to_dict()}
    (output_dir / "model_info.json").write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"data": data_log, "cv": cv_table.reset_index().to_dict(orient="records"),
                                                         "test": metrics}, indent=2), encoding="utf-8")
    report = write_report(output_dir, config, data_log, cv_table, chosen, metrics)
    card = write_model_card(output_dir, config, chosen, metrics, versions)
    return {"chosen": chosen, "cv_table": cv_table, "metrics": metrics, "data_log": data_log, "report": report, "model_card": card}


def predict(model_dir: Path, records: pd.DataFrame) -> pd.DataFrame:
    info = json.loads((Path(model_dir) / "model_info.json").read_text(encoding="utf-8"))
    missing = [c for c in info["features"] if c not in records.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    model = joblib.load(Path(model_dir) / "model.joblib")  # only load artefacts you produced yourself
    proba = model.predict_proba(records[info["features"]])[:, 1]
    return records.assign(pass_probability=proba.round(3), predicted_pass=(proba >= info["decision_threshold"]).astype(int))
