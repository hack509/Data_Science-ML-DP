"""Reference solution - A02 Reusable PyTorch training loop with early stopping."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = 32, val_fraction: float = 0.2, seed: int = 0):
    """Shuffle, split, standardize with training statistics, and wrap in DataLoaders.

    Returns (train_loader, val_loader, (mean, std)).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_val = int(round(len(X) * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std[std == 0] = 1.0

    def to_dataset(idx):
        features = torch.tensor((X[idx] - mean) / std, dtype=torch.float32)
        return TensorDataset(features, torch.tensor(y[idx], dtype=torch.int64))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(to_dataset(train_idx), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(to_dataset(val_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, (mean, std)


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn, optimizer) -> float:
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()
    total, correct, count = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        total += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        count += len(xb)
    return total / count, correct / count


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs: int = 100, patience: int | None = 10) -> dict:
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "best_epoch": 0}
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        history["train_loss"].append(train_one_epoch(model, train_loader, loss_fn, optimizer))
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        if val_loss < best_loss:
            best_loss, best_state, history["best_epoch"] = val_loss, copy.deepcopy(model.state_dict()), epoch
        elif patience is not None and epoch - history["best_epoch"] >= patience:
            break
    model.load_state_dict(best_state)
    return history


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model: nn.Module, path: str | Path) -> nn.Module:
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
