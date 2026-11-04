"""Reference solution - I09 k-means from scratch with NumPy."""

from __future__ import annotations

import numpy as np


def init_centroids(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Pick k distinct observations at random as initial centroids."""
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of observations")
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), size=k, replace=False)].astype(float)


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the closest centroid for each observation (vectorized)."""
    distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def update(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Mean of each cluster; an empty cluster keeps its previous centroid."""
    new = centroids.copy()
    for j in range(len(centroids)):
        members = X[labels == j]
        if len(members):
            new[j] = members.mean(axis=0)
    return new


def inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    return float(((X - centroids[labels]) ** 2).sum())


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, tol: float = 1e-6, seed: int = 0):
    """Return (centroids, labels, inertia, n_iterations)."""
    centroids = init_centroids(X, k, seed)
    for iteration in range(1, n_iter + 1):
        labels = assign(X, centroids)
        new_centroids = update(X, labels, centroids)
        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < tol:
            break
    labels = assign(X, centroids)
    return centroids, labels, inertia(X, centroids, labels), iteration


def best_of(X: np.ndarray, k: int, n_init: int = 10, seed: int = 0):
    """Run k-means n_init times with different seeds and keep the lowest inertia."""
    runs = [kmeans(X, k, seed=seed + i) for i in range(n_init)]
    return min(runs, key=lambda run: run[2])
