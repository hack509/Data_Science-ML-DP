"""A06 - Scaled dot-product and multi-head attention in NumPy. Complete the functions (see README.md)."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    raise NotImplementedError("TODO")


def causal_mask(n: int) -> np.ndarray:
    raise NotImplementedError("TODO")


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None):
    raise NotImplementedError("TODO")


def multi_head_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
                         n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    raise NotImplementedError("TODO")
