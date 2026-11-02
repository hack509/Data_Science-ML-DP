"""B01 - Student averages. Complete the functions (see README.md)."""

from __future__ import annotations

SUBJECTS = ("math", "physics", "french")


def average(scores: list[float]) -> float:
    """Return the arithmetic mean. Raise ValueError if scores is empty."""
    raise NotImplementedError("TODO")


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Return sum(w * x) / sum(w) over the subjects present in scores."""
    raise NotImplementedError("TODO")


def student_averages(records: list[dict], weights: dict[str, float] | None = None) -> dict[str, float]:
    """Return {student_id: average rounded to 2 decimals}."""
    raise NotImplementedError("TODO")


def mention(avg: float) -> str:
    """Return the French grade label for an average between 0 and 20."""
    raise NotImplementedError("TODO")


def class_report(records: list[dict], weights: dict[str, float] | None = None) -> dict:
    """Return averages, best_student, pass_rate and mentions."""
    raise NotImplementedError("TODO")
