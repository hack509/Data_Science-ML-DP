"""Reference solution - A05 Cross-validation from scratch and data leakage."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def kfold_indices(n: int, k: int, shuffle: bool = False, seed: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Same folds as sklearn.model_selection.KFold(n_splits=k, shuffle=shuffle, random_state=seed)."""
    if not 2 <= k <= n:
        raise ValueError("k must be between 2 and n")
    indices = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)  # same generator as scikit-learn
    fold_sizes = np.full(k, n // k)
    fold_sizes[: n % k] += 1  # the first n % k folds get one extra sample
    folds, start = [], 0
    for size in fold_sizes:
        validation = indices[start : start + size]
        train = np.concatenate([indices[:start], indices[start + size :]])
        folds.append((train, validation))
        start += size
    return folds


def cross_validate(model, X: np.ndarray, y: np.ndarray, k: int = 5, shuffle: bool = False, seed: int | None = None) -> np.ndarray:
    """Accuracy of a fresh copy of `model` on each validation fold."""
    scores = []
    for train, validation in kfold_indices(len(X), k, shuffle, seed):
        fitted = clone(model).fit(X[train], y[train])
        scores.append(fitted.score(X[validation], y[validation]))
    return np.array(scores)


def selection_cv(X: np.ndarray, y: np.ndarray, n_features: int = 20, k: int = 5, leak: bool = False) -> float:
    """Mean CV accuracy with univariate feature selection done wrongly (leak=True) or correctly."""
    classifier = LogisticRegression(max_iter=1000)
    if leak:
        X_selected = SelectKBest(f_classif, k=n_features).fit_transform(X, y)  # uses the validation labels!
        return float(cross_validate(classifier, X_selected, y, k).mean())
    pipeline = make_pipeline(SelectKBest(f_classif, k=n_features), classifier)
    return float(cross_validate(pipeline, X, y, k).mean())
