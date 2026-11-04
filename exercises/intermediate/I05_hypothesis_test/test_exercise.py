import numpy as np
import pytest
from scipy import stats

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.fixture(scope="module")
def students():
    return load_csv("students")


def test_confidence_interval_matches_scipy():
    rng = np.random.default_rng(1)
    sample = rng.normal(10, 2, size=40)
    mean, low, high = ex.mean_confidence_interval(sample)
    expected = stats.t.interval(0.95, df=39, loc=sample.mean(), scale=stats.sem(sample))
    assert mean == pytest.approx(sample.mean())
    assert (low, high) == pytest.approx(expected)
    with pytest.raises(ValueError):
        ex.mean_confidence_interval([1.0])


def test_bootstrap_is_reproducible_and_close():
    rng = np.random.default_rng(2)
    sample = rng.normal(10, 2, size=200)
    first = ex.bootstrap_ci(sample, seed=3)
    assert first == ex.bootstrap_ci(sample, seed=3)
    _, low, high = ex.mean_confidence_interval(sample)
    assert first[0] == pytest.approx(low, abs=0.1) and first[1] == pytest.approx(high, abs=0.1)


def test_welch_matches_scipy():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 50)
    t, dof, p = ex.welch_t_test(a, b)
    reference = stats.ttest_ind(a, b, equal_var=False)
    assert t == pytest.approx(reference.statistic)
    assert p == pytest.approx(reference.pvalue)
    assert 30 < dof < 80


def test_cohens_d():
    assert ex.cohens_d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert ex.cohens_d([2, 3, 4], [1, 2, 3]) == pytest.approx(1.0)


def test_compare_groups_on_students(students):
    result = ex.compare_groups(students, "group", "final_score", "C", "A")
    assert result["difference"] == pytest.approx(0.70, abs=0.01)
    assert result["t"] == pytest.approx(1.80, abs=0.01)
    assert result["p_value"] == pytest.approx(0.074, abs=0.001)
    assert result["reject_h0"] is False
    assert set(result) == {"mean_a", "mean_b", "difference", "t", "df", "p_value", "cohens_d", "reject_h0"}
