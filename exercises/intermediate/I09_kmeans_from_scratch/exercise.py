"""I09 - k-means from scratch with NumPy. Complete the functions (see README.md)."""

from __future__ import annotations

import numpy as np


def init_centroids(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    raise NotImplementedError("TODO")


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def update(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    raise NotImplementedError("TODO")


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, tol: float = 1e-6, seed: int = 0):
    raise NotImplementedError("TODO")


def best_of(X: np.ndarray, k: int, n_init: int = 10, seed: int = 0):
    raise NotImplementedError("TODO")
