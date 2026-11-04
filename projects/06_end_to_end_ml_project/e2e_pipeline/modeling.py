"""Preprocessing and model selection with cross-validation."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config


def preprocessor(config: Config) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), config.numeric_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), config.categorical_features),
    ])


def candidates(config: Config) -> dict[str, Pipeline]:
    classifiers = {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=config.seed, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=config.seed),
    }
    return {name: Pipeline([("prep", preprocessor(config)), ("clf", clf)]) for name, clf in classifiers.items()}


def select_model(X: pd.DataFrame, y: pd.Series, config: Config) -> tuple[str, Pipeline, pd.DataFrame]:
    """Compare candidates by cross-validated ROC-AUC on the training set; prefer the simpler model on near-ties."""
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    rows = []
    for name, pipeline in candidates(config).items():
        scores = cross_val_score(pipeline, X, y, cv=folds, scoring="roc_auc")
        rows.append({"model": name, "cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()})
    table = pd.DataFrame(rows).set_index("model").sort_values("cv_auc_mean", ascending=False)
    best_score = table["cv_auc_mean"].iloc[0]
    # Within one standard error of the best, keep the first (simplest) model in candidate order.
    tolerance = table["cv_auc_std"].iloc[0] / config.cv_folds**0.5
    for name in candidates(config):
        if name != "baseline_majority" and table.loc[name, "cv_auc_mean"] >= best_score - tolerance:
            chosen = name
            break
    model = candidates(config)[chosen].fit(X, y)
    return chosen, model, table
