"""Model selection, evaluation and persistence for house prices."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lab.data import load_csv

TARGET = "price_keur"
NUMERIC = ["surface_m2", "rooms", "age_years", "distance_center_km", "has_garden"]
NOMINAL = ["neighborhood"]
ORDINAL = ["energy_class"]
FEATURES = NUMERIC + NOMINAL + ORDINAL
SEED = 0


def load_and_split(test_size: float = 0.2):
    df = load_csv("house_prices")
    return train_test_split(df[FEATURES], df[TARGET], test_size=test_size, random_state=SEED)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("nom", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), NOMINAL),
        ("ord", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=[list("ABCDEFG")]))]), ORDINAL),
    ])


def log_target(regressor) -> TransformedTargetRegressor:
    """Train on log(price) and predict in k EUR (the errors are multiplicative, see lesson 5.2)."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def candidates(quick: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    return {
        "ridge_log": (
            Pipeline([("prep", preprocessor()), ("model", log_target(Ridge()))]),
            {"model__regressor__alpha": [0.1, 1.0] if quick else [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline([("prep", preprocessor()), ("model", RandomForestRegressor(n_estimators=100 if quick else 300, random_state=SEED, n_jobs=-1))]),
            {"model__min_samples_leaf": [2] if quick else [1, 3, 8]},
        ),
        "gradient_boosting_log": (
            Pipeline([("prep", preprocessor()), ("model", log_target(HistGradientBoostingRegressor(random_state=SEED)))]),
            {"model__regressor__learning_rate": [0.1] if quick else [0.03, 0.1], "model__regressor__max_leaf_nodes": [15] if quick else [15, 31]},
        ),
    }


def select_model(X_train, y_train, quick: bool = False) -> tuple[str, Pipeline, pd.DataFrame]:
    folds = KFold(n_splits=3 if quick else 5, shuffle=True, random_state=SEED)
    rows, best = [], None
    for name, (pipeline, grid) in candidates(quick).items():
        search = GridSearchCV(pipeline, grid, cv=folds, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train, y_train)
        std = search.cv_results_["std_test_score"][search.best_index_]
        rows.append({"model": name, "cv_mae": -search.best_score_, "cv_mae_std": std, "best_params": search.best_params_})
        if best is None or -search.best_score_ < best[1]:
            best = (name, -search.best_score_, search.best_estimator_)
    table = pd.DataFrame(rows).set_index("model").sort_values("cv_mae")
    return best[0], best[2], table


def bootstrap_interval(y_true, y_pred, metric, n_boot: int = 1000, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    values = [metric(y_true[i], y_pred[i]) for i in rng.integers(0, len(y_true), size=(n_boot, len(y_true)))]
    return [round(float(v), 3) for v in np.percentile(values, [2.5, 97.5])]


def evaluate(model, X_test, y_test, X_train, y_train) -> dict:
    pred = model.predict(X_test)
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train).predict(X_test)
    by_hood = (
        pd.DataFrame({"hood": X_test["neighborhood"], "error": np.abs(y_test - pred) / y_test})
        .groupby("hood")["error"].agg(["size", "mean"]).round(3)
    )
    return {
        "test_mae": round(float(mean_absolute_error(y_test, pred)), 3),
        "test_mae_ci95": bootstrap_interval(y_test, pred, mean_absolute_error),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "baseline_mae": round(float(mean_absolute_error(y_test, baseline)), 3),
        "relative_error_by_neighborhood": {k: {"n": int(v["size"]), "mean": float(v["mean"])} for k, v in by_hood.iterrows()},
    }


def save(model, name: str, metrics: dict, cv_table: pd.DataFrame, X_train: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"model": output_dir / "model.joblib", "metadata": output_dir / "metadata.json", "metrics": output_dir / "metrics.json"}
    joblib.dump(model, paths["model"])
    metadata = {
        "model": name,
        "features": FEATURES,
        "numeric_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERIC},
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "data": "datasets/generated/house_prices.csv (synthetic, not for real decisions)",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    content = {"selected_model": name, **metrics, "cross_validation": cv_table.reset_index().to_dict(orient="records")}
    paths["metrics"].write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    return paths


def load_model(model_dir: Path):
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return joblib.load(model_dir / "model.joblib"), metadata


def predict(model_dir: Path, records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    model, metadata = load_model(model_dir)
    missing = [c for c in metadata["features"] if c not in records.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    warnings = []
    for column, (low, high) in metadata["numeric_ranges"].items():
        for i in records.index[~records[column].between(low, high)]:
            warnings.append(f"row {i}: {column}={records.at[i, column]} outside [{low}, {high}]")
    result = records.assign(predicted_price_keur=model.predict(records[metadata["features"]]).round(1))
    return result, warnings


def run(output_dir: Path, quick: bool = False) -> dict:
    X_train, X_test, y_train, y_test = load_and_split()
    name, model, table = select_model(X_train, y_train, quick)
    metrics = evaluate(model, X_test, y_test, X_train, y_train)
    paths = save(model, name, metrics, table, X_train, Path(output_dir))
    return {"selected": name, "cv_table": table, "metrics": metrics, "paths": paths}
