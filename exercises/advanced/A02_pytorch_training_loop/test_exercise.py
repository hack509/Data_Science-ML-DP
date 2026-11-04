import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from lab.data import load_csv  # noqa: E402
from lab.exercise_loader import load_target  # noqa: E402

ex = load_target(__file__)


@pytest.fixture(scope="module")
def data():
    df = load_csv("multiclass")
    return df.drop(columns="target").to_numpy(), df["target"].to_numpy()


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 3))


def test_loaders(data):
    X, y = data
    train_loader, val_loader, (mean, std) = ex.make_loaders(X, y, batch_size=64, val_fraction=0.2, seed=0)
    assert len(train_loader.dataset) == 720 and len(val_loader.dataset) == 180
    xb, yb = next(iter(train_loader))
    assert xb.dtype == torch.float32 and yb.dtype == torch.int64 and xb.shape[1] == 5
    train_features = train_loader.dataset.tensors[0]
    np.testing.assert_allclose(train_features.mean(dim=0).numpy(), 0, atol=1e-5)
    assert not np.allclose(val_loader.dataset.tensors[0].mean(dim=0).numpy(), 0, atol=1e-5), "validation must use training statistics"


def test_fit_and_early_stopping(data):
    X, y = data
    train_loader, val_loader, _ = ex.make_loaders(X, y, seed=0)
    model = make_model()
    history = ex.fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-2), epochs=200, patience=5)
    n_epochs = len(history["val_loss"])
    assert len(history["train_loss"]) == n_epochs == len(history["val_accuracy"])
    assert n_epochs < 200, "early stopping should trigger"
    assert n_epochs - 1 - history["best_epoch"] == 5
    best_val = min(history["val_loss"])
    loss, accuracy = ex.evaluate(model, val_loader, nn.CrossEntropyLoss())
    assert loss == pytest.approx(best_val, rel=1e-5), "the best weights must be restored"
    assert accuracy > 0.85


def test_no_patience_runs_all_epochs(data):
    X, y = data
    train_loader, val_loader, _ = ex.make_loaders(X, y, seed=1)
    model = make_model()
    history = ex.fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-2), epochs=4, patience=None)
    assert len(history["train_loss"]) == 4


def test_checkpoint_roundtrip(data, tmp_path):
    X, y = data
    train_loader, val_loader, _ = ex.make_loaders(X, y, seed=0)
    model = make_model()
    ex.fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-2), epochs=5, patience=None)
    path = ex.save_checkpoint(model, tmp_path / "nested" / "model.pt")
    restored = ex.load_checkpoint(make_model(), path)
    assert not restored.training
    batch = val_loader.dataset.tensors[0]
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(batch), restored(batch))
