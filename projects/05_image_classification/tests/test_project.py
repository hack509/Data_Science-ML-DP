import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from shape_classifier.training import augment, load_images, predict_files, run  # noqa: E402

from lab.paths import DATASETS_DIR  # noqa: E402


def test_load_images():
    images, labels, classes = load_images()
    assert images.shape == (300, 28, 28) and classes == ["circle", "square", "triangle"]
    assert sorted(set(labels.tolist())) == [0, 1, 2]


def test_augment_keeps_range_and_shape():
    batch = torch.rand(5, 1, 28, 28)
    out = augment(batch, torch.Generator().manual_seed(0))
    assert out.shape == batch.shape and out.min() >= 0 and out.max() <= 1


def test_quick_training_and_prediction(tmp_path):
    result = run(tmp_path, quick=True)
    metrics = result["metrics"]
    assert metrics["epochs_run"] <= 8
    assert metrics["test_accuracy"] > 0.5  # well above chance (0.33) even after a few epochs
    assert (tmp_path / "weights.pt").exists() and (tmp_path / "learning_curves.png").exists()
    rows = predict_files([DATASETS_DIR / "generated/shapes/circle/circle_000.png"], tmp_path)
    assert rows[0]["class"] in {"circle", "square", "triangle"} and 0 < rows[0]["confidence"] <= 1
