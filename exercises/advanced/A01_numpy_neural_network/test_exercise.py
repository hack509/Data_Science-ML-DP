import numpy as np
import pytest
from sklearn.datasets import make_moons

from lab.exercise_loader import load_target

ex = load_target(__file__)


def test_shapes_and_errors():
    model = ex.MLP([3, 5, 4, 1], seed=0)
    assert [w.shape for w in model.weights] == [(3, 5), (5, 4), (4, 1)]
    assert [b.shape for b in model.biases] == [(5,), (4,), (1,)]
    assert model.forward(np.zeros((7, 3))).shape == (7, 1)
    with pytest.raises(ValueError):
        ex.MLP([3, 2])
    with pytest.raises(ValueError):
        ex.MLP([3])


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.integers(0, 2, 20)
    model = ex.MLP([3, 5, 4, 1], seed=1)
    grads_W, grads_b = model.backward(X, y)
    assert [g.shape for g in grads_W] == [w.shape for w in model.weights]
    assert [g.shape for g in grads_b] == [b.shape for b in model.biases]
    before = [w.copy() for w in model.weights]
    for layer in range(3):
        for index in [(0, 0), (1, 0), (2, 0)]:
            analytic = grads_W[layer][index]
            numeric = ex.numerical_gradient(model, X, y, layer, index)
            assert abs(analytic - numeric) <= 1e-5 * max(1.0, abs(analytic) + abs(numeric))
    for w, original in zip(model.weights, before):
        np.testing.assert_array_equal(w, original)


def test_solves_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = ex.MLP([2, 8, 1], seed=0)
    history = model.fit(X, y, epochs=3000, learning_rate=0.5)
    assert len(history) == 3000 and history[-1] < history[0]
    assert model.predict(X).tolist() == [0, 1, 1, 0]
    assert model.predict_proba(X).shape == (4,)


def test_moons():
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    model = ex.MLP([2, 16, 16, 1], seed=0)
    model.fit(X, y, epochs=1500, learning_rate=0.5)
    assert (model.predict(X) == y).mean() > 0.9
