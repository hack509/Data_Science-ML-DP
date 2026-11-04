"""Reference solution - A04 2D convolution and max pooling in NumPy."""

from __future__ import annotations

import numpy as np


def output_size(n: int, kernel: int, padding: int = 0, stride: int = 1) -> int:
    size = (n + 2 * padding - kernel) // stride + 1
    if size <= 0:
        raise ValueError("kernel larger than the padded input")
    return size


def conv2d(image: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    """Single-channel cross-correlation, vectorized with sliding windows."""
    padded = np.pad(image, padding)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)[::stride, ::stride]
    return np.einsum("ijkl,kl->ij", windows, kernel)


def conv2d_multi(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    """Multi-channel convolution like torch.nn.functional.conv2d for one image.

    x: (C_in, H, W) ; weights: (C_out, C_in, k, k) ; bias: (C_out,) -> (C_out, H_out, W_out)
    """
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    k = weights.shape[-1]
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(1, 2))[:, ::stride, ::stride]
    # windows: (C_in, H_out, W_out, k, k)
    return np.einsum("chwij,ocij->ohw", windows, weights) + bias[:, None, None]


def max_pool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    """Non-overlapping max pooling on the last two axes; leftover rows/columns are dropped."""
    h, w = x.shape[-2] // size * size, x.shape[-1] // size * size
    cropped = x[..., :h, :w]
    blocks = cropped.reshape(*x.shape[:-2], h // size, size, w // size, size)
    return blocks.max(axis=(-3, -1))
