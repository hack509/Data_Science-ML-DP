"""Load the PNG images, train a CNN with augmentation and early stopping, evaluate, save, predict."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch import nn  # noqa: E402

from lab.data import dataset_path  # noqa: E402
from lab.paths import DATASETS_DIR  # noqa: E402
from lab.plotting import CATEGORICAL, apply_style  # noqa: E402

SEED = 0
SIZE = 28


def load_images() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read every PNG listed in the index (the same files a learner would add by hand)."""
    index = pd.read_csv(dataset_path("shapes_index"))
    class_names = sorted(index["label"].unique())
    images = np.stack([np.asarray(Image.open(DATASETS_DIR / p).convert("L").resize((SIZE, SIZE))) for p in index["path"]])
    labels = index["label"].map({name: i for i, name in enumerate(class_names)}).to_numpy()
    return images, labels, class_names


def make_model(n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n_classes),
    )


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Label-preserving transforms in pixel space [0, 1]: flips, 90-degree rotations, shifts, scaling of contrast, noise."""
    out = batch.clone()
    n = len(out)
    for dim in (-1, -2):
        mask = torch.rand(n, generator=generator) < 0.5
        out[mask] = out[mask].flip(dim)
    turns = torch.randint(0, 4, (n,), generator=generator)
    for k in range(1, 4):
        out[turns == k] = torch.rot90(out[turns == k], k, dims=(-2, -1))
    shifts = torch.randint(-3, 4, (n, 2), generator=generator)
    for i in range(n):
        out[i] = torch.roll(out[i], (int(shifts[i, 0]), int(shifts[i, 1])), dims=(-2, -1))
    contrast = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=generator)
    return (out * contrast + 0.08 * torch.randn(out.shape, generator=generator)).clamp(0, 1)


def train(model, X_train, y_train, X_val, y_val, mean, std, epochs=60, patience=12, lr=3e-3, use_augmentation=True):
    generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best = (float("inf"), None, 0)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(X_train), generator=generator).split(32):
            batch = augment(X_train[idx], generator) if use_augmentation else X_train[idx]
            optimizer.zero_grad()
            loss = loss_fn(model((batch - mean) / std), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model((X_val - mean) / std)
            val_loss = loss_fn(logits, y_val).item()
            val_acc = (logits.argmax(1) == y_val).float().mean().item()
        history["train_loss"].append(total / len(X_train))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(model.state_dict()), epoch)
        elif epoch - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    history["best_epoch"] = best[2]
    return history


def plot_history(history: dict, path: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(history["train_loss"], color=CATEGORICAL[0], label="entraînement (augmenté)")
    axes[0].plot(history["val_loss"], color=CATEGORICAL[1], label="validation")
    axes[0].axvline(history["best_epoch"], color="#8a8a85", linestyle="--", linewidth=1)
    axes[0].set_title("Perte")
    axes[0].legend()
    axes[1].plot(history["val_accuracy"], color=CATEGORICAL[1])
    axes[1].set_title("Précision de validation")
    for ax in axes:
        ax.set_xlabel("époque")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run(output_dir: Path, quick: bool = False) -> dict:
    torch.manual_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels, class_names = load_images()
    X_tmp, X_test, y_tmp, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=SEED)
    to_tensor = lambda a: torch.tensor(a[:, None] / 255.0, dtype=torch.float32)  # noqa: E731
    Xtr, Xva, Xte = to_tensor(X_train), to_tensor(X_val), to_tensor(X_test)
    ytr, yva, yte = (torch.tensor(a, dtype=torch.int64) for a in (y_train, y_val, y_test))
    mean, std = Xtr.mean().item(), Xtr.std().item()

    model = make_model(len(class_names))
    start = time.perf_counter()
    history = train(model, Xtr, ytr, Xva, yva, mean, std, epochs=8 if quick else 60, patience=4 if quick else 12)
    seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        test_pred = model((Xte - mean) / std).argmax(1).numpy()
    report = classification_report(y_test, test_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "epochs_run": len(history["val_loss"]),
        "best_epoch": history["best_epoch"],
        "training_seconds": round(seconds, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_accuracy": round(float((test_pred == y_test).mean()), 4),
        "per_class_recall": {c: round(report[c]["recall"], 3) for c in class_names},
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    torch.save(model.state_dict(), output_dir / "weights.pt")
    (output_dir / "metadata.json").write_text(json.dumps({"classes": class_names, "size": SIZE, "mean": mean, "std": std}, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_history(history, output_dir / "learning_curves.png")
    return {"metrics": metrics, "history": history}


def predict_files(paths: list[Path], model_dir: Path) -> list[dict]:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = make_model(len(metadata["classes"]))
    model.load_state_dict(torch.load(model_dir / "weights.pt", weights_only=True))
    model.eval()
    results = []
    for path in paths:
        image = Image.open(path).convert("L").resize((metadata["size"], metadata["size"]))
        x = torch.tensor(np.asarray(image) / 255.0, dtype=torch.float32).view(1, 1, metadata["size"], metadata["size"])
        with torch.no_grad():
            probabilities = torch.softmax(model((x - metadata["mean"]) / metadata["std"]), dim=1)[0]
        best = int(probabilities.argmax())
        results.append({"file": str(path), "class": metadata["classes"][best], "confidence": round(float(probabilities[best]), 3)})
    return results
