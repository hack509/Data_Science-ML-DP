"""I07 - Regression pipeline for house prices. Complete the functions (see README.md)."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

TARGET = "price_keur"
NUMERIC = ["surface_m2", "rooms", "age_years", "distance_center_km", "has_garden"]
NOMINAL = ["neighborhood"]
ORDINAL = ["energy_class"]
ENERGY_ORDER = list("ABCDEFG")


def split(df: pd.DataFrame, seed: int = 0):
    raise NotImplementedError("TODO")


def build_pipeline(regressor) -> Pipeline:
    raise NotImplementedError("TODO")


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    raise NotImplementedError("TODO")


def candidate_models() -> dict:
    raise NotImplementedError("TODO")


def compare_models(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> pd.DataFrame:
    raise NotImplementedError("TODO")
