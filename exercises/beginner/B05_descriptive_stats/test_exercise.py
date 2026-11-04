import numpy as np
import pytest

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


def test_simple_values():
    assert ex.mean([1, 2, 3, 4]) == pytest.approx(2.5)
    assert ex.median([3, 1, 2]) == 2
    assert ex.median([4, 1, 3, 2]) == pytest.approx(2.5)
    assert ex.modes([1, 2, 2, 3, 3]) == [2, 3]
    assert ex.modes(["a", "b", "a"]) == ["a"]
    assert ex.variance([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(4.0)
    assert ex.std([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.0)
    assert ex.quantile([1, 2, 3, 4], 0.25) == pytest.approx(1.75)


def test_errors():
    for function in (ex.mean, ex.median, ex.modes, ex.variance, ex.iqr):
        with pytest.raises(ValueError):
            function([])
    with pytest.raises(ValueError):
        ex.variance([5], ddof=1)
    with pytest.raises(ValueError):
        ex.quantile([1, 2], 1.5)
    with pytest.raises(ValueError):
        ex.covariance([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        ex.zscores([3, 3, 3])


def test_input_not_modified():
    data = [3, 1, 2]
    ex.median(data)
    ex.quantile(data, 0.5)
    assert data == [3, 1, 2]


def test_against_numpy_on_dataset():
    df = load_csv("students")
    x = df["study_hours_week"].tolist()
    y = df["final_score"].tolist()
    assert ex.mean(x) == pytest.approx(np.mean(x))
    assert ex.median(x) == pytest.approx(np.median(x))
    assert ex.variance(x, ddof=1) == pytest.approx(np.var(x, ddof=1))
    assert ex.std(x) == pytest.approx(np.std(x))
    for q in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        assert ex.quantile(x, q) == pytest.approx(np.quantile(x, q))
    assert ex.iqr(x) == pytest.approx(np.subtract(*np.percentile(x, [75, 25])))
    assert ex.covariance(x, y) == pytest.approx(np.cov(x, y)[0, 1])
    assert ex.correlation(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    z = ex.zscores(y)
    assert np.mean(z) == pytest.approx(0, abs=1e-9) and np.std(z) == pytest.approx(1)
