"""A04 - 2D convolution and max pooling in NumPy. Complete the functions (see README.md)."""

from __future__ import annotations

import numpy as np


def output_size(n: int, kernel: int, padding: int = 0, stride: int = 1) -> int:
    raise NotImplementedError("TODO")


def conv2d(image: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    raise NotImplementedError("TODO")


def conv2d_multi(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    raise NotImplementedError("TODO")


def max_pool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    raise NotImplementedError("TODO")
