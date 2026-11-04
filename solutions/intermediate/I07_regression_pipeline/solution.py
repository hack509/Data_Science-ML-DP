"""Reference solution - I07 Regression pipeline for house prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "price_keur"
NUMERIC = ["surface_m2", "rooms", "age_years", "distance_center_km", "has_garden"]
NOMINAL = ["neighborhood"]
ORDINAL = ["energy_class"]
ENERGY_ORDER = list("ABCDEFG")


def split(df: pd.DataFrame, seed: int = 0):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def build_pipeline(regressor) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ("nom", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), NOMINAL),
        ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ordinal", OrdinalEncoder(categories=[ENERGY_ORDER]))]), ORDINAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = pipeline.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def candidate_models() -> dict:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1),
    }


def compare_models(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> pd.DataFrame:
    folds = KFold(n_splits=cv, shuffle=True, random_state=0)
    rows = []
    for name, regressor in candidate_models().items():
        scores = -cross_val_score(build_pipeline(regressor), X_train, y_train, cv=folds, scoring="neg_mean_absolute_error")
        rows.append({"model": name, "mean_mae": scores.mean(), "std_mae": scores.std()})
    return pd.DataFrame(rows).set_index("model").sort_values("mean_mae")
