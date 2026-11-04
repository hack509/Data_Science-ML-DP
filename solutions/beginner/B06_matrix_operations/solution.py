"""Reference solution - B06 Matrix operations with plain Python lists."""

from __future__ import annotations

import math

Vector = list[float]
Matrix = list[list[float]]


def shape(A: Matrix) -> tuple[int, int]:
    if not A or any(len(row) != len(A[0]) for row in A):
        raise ValueError("matrix must be non-empty and rectangular")
    return len(A), len(A[0])


def dot(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise ValueError("vectors must have the same length")
    return sum(a * b for a, b in zip(u, v))


def norm(v: Vector) -> float:
    return math.sqrt(dot(v, v))


def transpose(A: Matrix) -> Matrix:
    rows, cols = shape(A)
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def matvec(A: Matrix, v: Vector) -> Vector:
    _, cols = shape(A)
    if cols != len(v):
        raise ValueError(f"incompatible shapes: {shape(A)} and ({len(v)},)")
    return [dot(row, v) for row in A]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    (m, n), (n2, _) = shape(A), shape(B)
    if n != n2:
        raise ValueError(f"incompatible shapes: {shape(A)} and {shape(B)}")
    columns = transpose(B)
    return [[dot(row, col) for col in columns] for row in A]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def solve_2x2(A: Matrix, b: Vector) -> Vector:
    """Cramer's rule for a 2x2 system."""
    if shape(A) != (2, 2) or len(b) != 2:
        raise ValueError("A must be 2x2 and b of length 2")
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-12:
        raise ValueError("singular matrix: no unique solution")
    x = (b[0] * A[1][1] - A[0][1] * b[1]) / det
    y = (A[0][0] * b[1] - b[0] * A[1][0]) / det
    return [x, y]
