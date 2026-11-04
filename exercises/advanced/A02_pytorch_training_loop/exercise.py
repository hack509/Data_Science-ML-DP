"""A02 - Reusable PyTorch training loop with early stopping. Complete the functions (see README.md)."""

from __future__ import annotations

import copy  # noqa: F401
from pathlib import Path

import numpy as np
import torch  # noqa: F401
from torch import nn
from torch.utils.data import DataLoader, TensorDataset  # noqa: F401


def make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 32, val_fraction: float = 0.2, seed: int = 0):
    raise NotImplementedError("TODO")


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer) -> float:
    raise NotImplementedError("TODO")


def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    raise NotImplementedError("TODO")


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs: int = 100, patience: int | None = 10) -> dict:
    raise NotImplementedError("TODO")


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    raise NotImplementedError("TODO")


def load_checkpoint(model: nn.Module, path: str | Path) -> nn.Module:
    raise NotImplementedError("TODO")
