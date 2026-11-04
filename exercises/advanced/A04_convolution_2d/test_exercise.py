import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from lab.data import dataset_path  # noqa: E402
from lab.exercise_loader import load_target  # noqa: E402

ex = load_target(__file__)


def test_output_size():
    assert ex.output_size(28, 3) == 26
    assert ex.output_size(28, 3, padding=1, stride=2) == 14
    assert ex.output_size(32, 5) == 28
    with pytest.raises(ValueError):
        ex.output_size(2, 5)


@pytest.mark.parametrize("padding, stride", [(0, 1), (1, 1), (1, 2), (2, 3)])
def test_conv2d_matches_pytorch(padding, stride):
    rng = np.random.default_rng(0)
    image, kernel = rng.normal(size=(28, 28)), rng.normal(size=(3, 3))
    ours = ex.conv2d(image, kernel, padding=padding, stride=stride)
    reference = F.conv2d(torch.tensor(image).view(1, 1, 28, 28), torch.tensor(kernel).view(1, 1, 3, 3), padding=padding, stride=stride)
    np.testing.assert_allclose(ours, reference[0, 0].numpy(), atol=1e-5)


def test_conv2d_on_shape_image_detects_edges():
    images = np.load(dataset_path("shapes_npz"))["images"]
    square = images[100].astype(float) / 255
    response = ex.conv2d(square, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float))
    assert response.shape == (26, 26)
    assert np.abs(response).max() > 1.0


def test_conv2d_multi_matches_pytorch():
    rng = np.random.default_rng(1)
    x, w, b = rng.normal(size=(3, 12, 12)), rng.normal(size=(5, 3, 3, 3)), rng.normal(size=5)
    for padding, stride in [(0, 1), (1, 2)]:
        ours = ex.conv2d_multi(x, w, b, padding=padding, stride=stride)
        reference = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), padding=padding, stride=stride)[0]
        np.testing.assert_allclose(ours, reference.numpy(), atol=1e-5)


def test_max_pool_matches_pytorch():
    rng = np.random.default_rng(2)
    for shape in [(8, 8), (4, 9, 7)]:
        x = rng.normal(size=shape)
        reference = F.max_pool2d(torch.tensor(x)[None] if x.ndim == 2 else torch.tensor(x), 2)
        reference = reference[0] if x.ndim == 2 else reference
        np.testing.assert_allclose(ex.max_pool2d(x, 2), reference.numpy())


def test_conv2d_is_vectorized():
    image, kernel = np.random.default_rng(3).normal(size=(256, 256)), np.ones((5, 5))
    start = time.perf_counter()
    ex.conv2d(image, kernel, padding=2)
    assert time.perf_counter() - start < 0.5, "conv2d looks like it loops over pixels in Python"
