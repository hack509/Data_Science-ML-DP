"""B06 - Matrix operations with plain Python lists (no NumPy)."""

from __future__ import annotations

import math  # noqa: F401

Vector = list[float]
Matrix = list[list[float]]


def shape(A: Matrix) -> tuple[int, int]:
    raise NotImplementedError("TODO")


def dot(u: Vector, v: Vector) -> float:
    raise NotImplementedError("TODO")


def norm(v: Vector) -> float:
    raise NotImplementedError("TODO")


def transpose(A: Matrix) -> Matrix:
    raise NotImplementedError("TODO")


def matvec(A: Matrix, v: Vector) -> Vector:
    raise NotImplementedError("TODO")


def matmul(A: Matrix, B: Matrix) -> Matrix:
    raise NotImplementedError("TODO")


def identity(n: int) -> Matrix:
    raise NotImplementedError("TODO")


def solve_2x2(A: Matrix, b: Vector) -> Vector:
    raise NotImplementedError("TODO")
