"""A05 - Cross-validation from scratch and data leakage. Complete the functions (see README.md)."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone  # noqa: F401
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.pipeline import make_pipeline  # noqa: F401


def kfold_indices(n: int, k: int, shuffle: bool = False, seed: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    raise NotImplementedError("TODO")


def cross_validate(model, X: np.ndarray, y: np.ndarray, k: int = 5, shuffle: bool = False, seed: int | None = None) -> np.ndarray:
    raise NotImplementedError("TODO")


def selection_cv(X: np.ndarray, y: np.ndarray, n_features: int = 20, k: int = 5, leak: bool = False) -> float:
    raise NotImplementedError("TODO")
