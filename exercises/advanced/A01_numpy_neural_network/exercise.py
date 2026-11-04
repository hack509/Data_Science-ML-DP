"""A01 - Multilayer perceptron in NumPy with backpropagation. Complete the code (see README.md)."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO")


def binary_cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("TODO")


class MLP:
    """Binary classifier: ReLU hidden layers, one sigmoid output unit."""

    def __init__(self, layer_sizes: list[int], seed: int = 0):
        raise NotImplementedError("TODO")

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError("TODO")

    def backward(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("TODO")

    def step(self, grads_W, grads_b, learning_rate: float) -> None:
        raise NotImplementedError("TODO")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.5) -> list[float]:
        raise NotImplementedError("TODO")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")


def numerical_gradient(model, X: np.ndarray, y: np.ndarray, layer: int, index: tuple, h: float = 1e-6) -> float:
    raise NotImplementedError("TODO")
