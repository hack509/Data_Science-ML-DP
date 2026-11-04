import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.mark.parametrize("n, k, shuffle, seed", [(10, 3, False, None), (23, 5, False, None), (50, 4, True, 7)])
def test_same_folds_as_sklearn(n, k, shuffle, seed):
    ours = ex.kfold_indices(n, k, shuffle=shuffle, seed=seed)
    reference = list(KFold(n_splits=k, shuffle=shuffle, random_state=seed).split(np.zeros(n)))
    assert len(ours) == k
    for (train, val), (ref_train, ref_val) in zip(ours, reference):
        assert sorted(train.tolist()) == sorted(ref_train.tolist())
        assert sorted(val.tolist()) == sorted(ref_val.tolist())


def test_errors():
    with pytest.raises(ValueError):
        ex.kfold_indices(5, 1)
    with pytest.raises(ValueError):
        ex.kfold_indices(5, 6)


def test_scores_match_cross_val_score():
    df = load_csv("binary")
    X, y = df.drop(columns="target").to_numpy(), df["target"].to_numpy()
    model = LogisticRegression(max_iter=1000)
    ours = ex.cross_validate(model, X, y, k=5, shuffle=True, seed=0)
    reference = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    np.testing.assert_allclose(ours, reference)
    assert not hasattr(model, "coef_"), "the original model must not be fitted (use clone)"


def test_leakage_inflates_scores():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2000))
    y = rng.integers(0, 2, size=100)
    leaky = ex.selection_cv(X, y, leak=True)
    honest = ex.selection_cv(X, y, leak=False)
    assert leaky > 0.75
    assert 0.3 < honest < 0.7
