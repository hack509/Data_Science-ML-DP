import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


def test_init_distinct_and_errors():
    X = np.arange(20, dtype=float).reshape(10, 2)
    c = ex.init_centroids(X, 4, seed=1)
    assert c.shape == (4, 2) and len({tuple(row) for row in c}) == 4
    with pytest.raises(ValueError):
        ex.init_centroids(X, 11)


def test_assign_update_inertia():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    centroids = np.array([[0.0, 0.5], [10.0, 10.5]])
    labels = ex.assign(X, centroids)
    assert labels.tolist() == [0, 0, 1, 1]
    assert ex.inertia(X, centroids, labels) == pytest.approx(1.0)
    empty = ex.update(X, np.array([0, 0, 0, 0]), np.array([[1.0, 1.0], [5.0, 5.0]]))
    np.testing.assert_allclose(empty[1], [5.0, 5.0])  # empty cluster keeps its centroid
    np.testing.assert_allclose(empty[0], X.mean(axis=0))


def test_converges_on_customers():
    customers = load_csv("customers")
    truth = load_csv("customers_true_segments")["true_segment"]
    X = StandardScaler().fit_transform(customers.drop(columns="customer_id"))
    centroids, labels, value, n_iterations = ex.best_of(X, 4, n_init=10)
    assert centroids.shape == (4, 6)
    assert n_iterations <= 100
    assert adjusted_rand_score(truth, labels) > 0.95
    reference = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X).inertia_
    assert value == pytest.approx(reference, rel=0.01)
