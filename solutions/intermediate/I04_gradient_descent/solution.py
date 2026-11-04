"""Reference solution - I04 Gradient descent for multivariate linear regression."""

from __future__ import annotations

import numpy as np


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    if np.any(std == 0):
        raise ValueError("a column is constant and cannot be standardized")
    return (X - mean) / std, mean, std


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y - y_pred) ** 2))


def gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """Gradient of the MSE loss with respect to w and b."""
    error = predict(X, w, b) - y
    n = len(y)
    grad_w = 2 / n * X.T @ error
    grad_b = 2 / n * error.sum()
    return grad_w, float(grad_b)


def fit_gradient_descent(
    X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, n_iter: int = 1000, tol: float | None = None
) -> tuple[np.ndarray, float, list[float]]:
    w = np.zeros(X.shape[1])
    b = 0.0
    history: list[float] = []
    for _ in range(n_iter):
        history.append(mse(y, predict(X, w, b)))
        grad_w, grad_b = gradients(X, y, w, b)  # compute both gradients first
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        if not np.isfinite(history[-1]):
            raise FloatingPointError("the loss diverged: decrease the learning rate")
        if tol is not None and len(history) > 1 and history[-2] - history[-1] < tol:
            break
    return w, b, history


def to_original_units(w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert coefficients learned on standardized data back to the original scale."""
    w_original = w / std
    b_original = b - float(np.sum(w * mean / std))
    return w_original, b_original
