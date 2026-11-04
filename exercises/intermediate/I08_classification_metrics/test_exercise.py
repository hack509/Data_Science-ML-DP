import time

import numpy as np
import pytest
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


def test_small_examples():
    y_true, y_pred = [1, 0, 1, 1, 0], [1, 1, 0, 1, 0]
    assert ex.confusion_counts(y_true, y_pred) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert ex.accuracy(y_true, y_pred) == pytest.approx(0.6)
    assert ex.precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert ex.recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert ex.f1(y_true, y_pred) == pytest.approx(2 / 3)
    assert ex.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert ex.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_edge_cases():
    assert ex.precision([1, 1], [0, 0]) == 0.0
    assert ex.f1([0, 0], [0, 0]) == 0.0
    with pytest.raises(ValueError):
        ex.confusion_counts([1], [1, 0])
    with pytest.raises(ValueError):
        ex.accuracy([], [])
    with pytest.raises(ValueError):
        ex.roc_auc([1, 1], [0.2, 0.3])


def test_string_labels_and_matrix():
    y_true = ["spam", "ham", "spam", "ham", "ham"]
    y_pred = ["spam", "spam", "ham", "ham", "ham"]
    assert ex.precision(y_true, y_pred, positive="spam") == pytest.approx(0.5)
    assert ex.confusion_matrix(y_true, y_pred, ["ham", "spam"]) == [[2, 1], [1, 1]]


def test_against_sklearn():
    df = load_csv("binary")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="target"), df["target"], random_state=0)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    pred = model.predict(X_test).tolist()
    proba = model.predict_proba(X_test)[:, 1].tolist()
    truth = y_test.tolist()
    assert ex.accuracy(truth, pred) == pytest.approx(metrics.accuracy_score(truth, pred))
    assert ex.precision(truth, pred) == pytest.approx(metrics.precision_score(truth, pred))
    assert ex.recall(truth, pred) == pytest.approx(metrics.recall_score(truth, pred))
    assert ex.f1(truth, pred) == pytest.approx(metrics.f1_score(truth, pred))
    assert ex.confusion_matrix(truth, pred, [0, 1]) == metrics.confusion_matrix(truth, pred).tolist()
    assert ex.roc_auc(truth, proba) == pytest.approx(metrics.roc_auc_score(truth, proba))


def test_auc_is_fast_enough():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 10_000).tolist()
    scores = np.round(rng.random(10_000), 3).tolist()  # rounding creates ties
    start = time.perf_counter()
    value = ex.roc_auc(y, scores)
    assert time.perf_counter() - start < 5
    assert value == pytest.approx(metrics.roc_auc_score(y, scores))
