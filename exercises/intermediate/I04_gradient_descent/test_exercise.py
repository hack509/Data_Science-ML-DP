import numpy as np
import pytest

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)
FEATURES = ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score"]


@pytest.fixture(scope="module")
def data():
    df = load_csv("students")
    return df[FEATURES].to_numpy(), df["final_score"].to_numpy()


def test_standardize(data):
    X, _ = data
    Xs, mean, std = ex.standardize(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=0), 1)
    with pytest.raises(ValueError):
        ex.standardize(np.ones((5, 2)))


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    w = rng.normal(size=3)
    b = 0.5
    grad_w, grad_b = ex.gradients(X, y, w, b)
    h = 1e-6
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        numeric = (ex.mse(y, ex.predict(X, w + step, b)) - ex.mse(y, ex.predict(X, w - step, b))) / (2 * h)
        assert grad_w[j] == pytest.approx(numeric, rel=1e-5)
    numeric_b = (ex.mse(y, ex.predict(X, w, b + h)) - ex.mse(y, ex.predict(X, w, b - h))) / (2 * h)
    assert grad_b == pytest.approx(numeric_b, rel=1e-5)


def test_fit_matches_least_squares(data):
    X, y = data
    Xs, mean, std = ex.standardize(X)
    w, b, history = ex.fit_gradient_descent(Xs, y, learning_rate=0.1, n_iter=1000)
    assert len(history) == 1000
    assert all(later <= earlier + 1e-12 for earlier, later in zip(history, history[1:]))
    assert history[-1] == pytest.approx(3.02, abs=0.01)
    w_orig, b_orig = ex.to_original_units(w, b, mean, std)
    design = np.column_stack([np.ones(len(X)), X])
    exact = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(np.r_[b_orig, w_orig], exact, atol=1e-4)


def test_tolerance_stops_early(data):
    X, y = data
    Xs, _, _ = ex.standardize(X)
    _, _, history = ex.fit_gradient_descent(Xs, y, learning_rate=0.1, n_iter=5000, tol=1e-8)
    assert len(history) < 5000


def test_divergence_is_reported(data):
    X, y = data
    Xs, _, _ = ex.standardize(X)
    with pytest.raises(FloatingPointError):
        ex.fit_gradient_descent(Xs, y, learning_rate=5.0, n_iter=2000)
