"""Reference solution - A01 Multilayer perceptron in NumPy with backpropagation."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class MLP:
    """Binary classifier: ReLU hidden layers, one sigmoid output unit."""

    def __init__(self, layer_sizes: list[int], seed: int = 0):
        if len(layer_sizes) < 2 or layer_sizes[-1] != 1:
            raise ValueError("layer_sizes must look like [n_inputs, ..., 1]")
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out)) for n_in, n_out in zip(layer_sizes, layer_sizes[1:])]
        self.biases = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._activations = [X]  # inputs of each layer
        self._pre_activations = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            self._pre_activations.append(z)
            a = sigmoid(z) if i == len(self.weights) - 1 else np.maximum(0, z)
            self._activations.append(a)
        return a

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return binary_cross_entropy(self.forward(X), y.reshape(-1, 1))

    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        P = self.forward(X)
        delta = (P - y.reshape(-1, 1)) / len(X)  # gradient wrt the output pre-activation
        grads_W, grads_b = [], []
        for i in reversed(range(len(self.weights))):
            grads_W.insert(0, self._activations[i].T @ delta)
            grads_b.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self._pre_activations[i - 1] > 0)
        return grads_W, grads_b

    def step(self, grads_W, grads_b, learning_rate: float) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_W[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.5) -> list[float]:
        history = []
        for _ in range(epochs):
            history.append(self.loss(X, y))
            self.step(*self.backward(X, y), learning_rate)
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def numerical_gradient(model: MLP, X: np.ndarray, y: np.ndarray, layer: int, index: tuple, h: float = 1e-6) -> float:
    """Central finite difference of the loss with respect to one weight."""
    original = model.weights[layer][index]
    model.weights[layer][index] = original + h
    plus = model.loss(X, y)
    model.weights[layer][index] = original - h
    minus = model.loss(X, y)
    model.weights[layer][index] = original
    return (plus - minus) / (2 * h)
