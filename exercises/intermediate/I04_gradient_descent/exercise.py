"""I04 - Gradient descent for multivariate linear regression (NumPy only)."""

from __future__ import annotations

import numpy as np


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("TODO")


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    raise NotImplementedError("TODO")


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    raise NotImplementedError("TODO")


def fit_gradient_descent(
    X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, n_iter: int = 1000, tol: float | None = None
) -> tuple[np.ndarray, float, list[float]]:
    raise NotImplementedError("TODO")


def to_original_units(w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
    raise NotImplementedError("TODO")
