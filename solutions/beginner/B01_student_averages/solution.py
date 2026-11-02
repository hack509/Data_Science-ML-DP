"""Reference solution - B01 Student averages.

Key ideas:
* validate inputs early and raise explicit errors;
* a weighted average is sum(w * x) / sum(w);
* return values instead of printing them, so they can be tested.
"""

from __future__ import annotations

SUBJECTS = ("math", "physics", "french")


def average(scores: list[float]) -> float:
    if not scores:
        raise ValueError("scores must not be empty")
    return sum(scores) / len(scores)


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    missing = set(scores) - set(weights)
    if missing:
        raise KeyError(f"no weight for: {sorted(missing)}")
    total_weight = sum(weights[subject] for subject in scores)
    if total_weight <= 0:
        raise ValueError("the sum of the weights must be positive")
    return sum(weights[subject] * score for subject, score in scores.items()) / total_weight


def student_averages(records: list[dict], weights: dict[str, float] | None = None) -> dict[str, float]:
    if weights is None:
        weights = {subject: 1.0 for subject in SUBJECTS}
    result = {}
    for record in records:
        scores = {subject: float(record[subject]) for subject in SUBJECTS}
        result[record["student_id"]] = round(weighted_average(scores, weights), 2)
    return result


def mention(avg: float) -> str:
    if not 0 <= avg <= 20:
        raise ValueError(f"average out of range: {avg}")
    if avg < 10:
        return "ajourné"
    if avg < 12:
        return "passable"
    if avg < 14:
        return "assez bien"
    if avg < 16:
        return "bien"
    return "très bien"


def class_report(records: list[dict], weights: dict[str, float] | None = None) -> dict:
    averages = student_averages(records, weights)
    best = max(averages, key=averages.get)
    passed = sum(1 for value in averages.values() if value >= 10)
    return {
        "averages": averages,
        "best_student": best,
        "pass_rate": round(passed / len(averages), 3),
        "mentions": {student: mention(value) for student, value in averages.items()},
    }


if __name__ == "__main__":
    import csv

    with open("datasets/generated/students.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = class_report(rows, {"math": 4, "physics": 3, "french": 3})
    print("best student:", report["best_student"], "| pass rate:", report["pass_rate"])
