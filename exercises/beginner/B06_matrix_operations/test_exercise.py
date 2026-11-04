import numpy as np
import pytest

from lab.exercise_loader import load_target

ex = load_target(__file__)


def test_small_examples():
    assert ex.shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    assert ex.dot([1, 2, 3], [4, 5, 6]) == 32
    assert ex.norm([3, 4]) == pytest.approx(5.0)
    assert ex.transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert ex.matvec([[1, 2], [3, 4]], [1, -1]) == [-1, -1]
    assert ex.matmul([[2, 1], [1, -1]], [[1, 2], [0, 1]]) == [[2, 5], [1, 1]]
    assert ex.identity(2) == [[1.0, 0.0], [0.0, 1.0]]
    assert ex.solve_2x2([[2, 1], [1, -1]], [7, -1]) == pytest.approx([2.0, 3.0])


def test_errors():
    with pytest.raises(ValueError):
        ex.shape([[1, 2], [3]])
    with pytest.raises(ValueError):
        ex.dot([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        ex.matvec([[1, 2], [3, 4]], [1, 2, 3])
    with pytest.raises(ValueError):
        ex.matmul([[1, 2, 3]], [[1, 2, 3]])
    with pytest.raises(ValueError):
        ex.solve_2x2([[1, 2], [2, 4]], [1, 2])


def test_against_numpy():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(3, 5))
    v = rng.normal(size=3)
    np.testing.assert_allclose(ex.matmul(A.tolist(), B.tolist()), A @ B)
    np.testing.assert_allclose(ex.matvec(A.tolist(), v.tolist()), A @ v)
    np.testing.assert_allclose(ex.transpose(A.tolist()), A.T)
    I3 = ex.identity(3)
    np.testing.assert_allclose(ex.matmul(A.tolist(), I3), A)


def test_arguments_not_modified():
    A = [[1, 2], [3, 4]]
    ex.transpose(A)
    ex.matmul(A, A)
    assert A == [[1, 2], [3, 4]]
