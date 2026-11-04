"""Reference solution - A06 Scaled dot-product and multi-head attention in NumPy."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)  # avoids overflow in exp
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_mask(n: int) -> np.ndarray:
    """True where attention is allowed (position j <= i)."""
    return np.tril(np.ones((n, n), dtype=bool))


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None):
    """Return (output, weights). Q: (n, d_k), K: (m, d_k), V: (m, d_v), mask: (n, m) booleans."""
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
                         n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Self-attention with n_heads heads. X: (n, d_model); every W: (d_model, d_model)."""
    n, d_model = X.shape
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    d_head = d_model // n_heads
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    heads = []
    for h in range(n_heads):
        part = slice(h * d_head, (h + 1) * d_head)
        output, _ = scaled_dot_product_attention(Q[:, part], K[:, part], V[:, part], mask)
        heads.append(output)
    return np.concatenate(heads, axis=1) @ W_o
