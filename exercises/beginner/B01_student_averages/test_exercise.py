import csv

import pytest

from lab.data import dataset_path
from lab.exercise_loader import load_target

ex = load_target(__file__)

RECORDS = [
    {"student_id": "S01", "math": "14", "physics": "12.5", "french": "11"},
    {"student_id": "S02", "math": "8.5", "physics": "9", "french": "13.5"},
    {"student_id": "S03", "math": "0", "physics": "10", "french": "12"},
]


def test_average():
    assert ex.average([12, 14, 16]) == pytest.approx(14.0)
    with pytest.raises(ValueError):
        ex.average([])


def test_weighted_average():
    assert ex.weighted_average({"math": 10, "french": 16}, {"math": 2, "french": 1}) == pytest.approx(12.0)
    with pytest.raises(KeyError):
        ex.weighted_average({"math": 10, "art": 15}, {"math": 1})
    with pytest.raises(ValueError):
        ex.weighted_average({"math": 10}, {"math": 0})


def test_student_averages_equal_weights_and_zero_score():
    result = ex.student_averages(RECORDS)
    assert result == {"S01": 12.5, "S02": 10.33, "S03": 7.33}


@pytest.mark.parametrize(
    "value, expected",
    [(0, "ajourné"), (9.99, "ajourné"), (10, "passable"), (13.99, "assez bien"), (14, "bien"), (16, "très bien"), (20, "très bien")],
)
def test_mention(value, expected):
    assert ex.mention(value) == expected


def test_mention_out_of_range():
    with pytest.raises(ValueError):
        ex.mention(21)


def test_class_report_on_dataset():
    with open(dataset_path("students"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    weights = {"math": 4, "physics": 3, "french": 3}
    report = ex.class_report(rows, weights)
    assert report["best_student"] == "S0081"
    assert report["pass_rate"] == pytest.approx(0.44)
    for row in rows:
        assert report["averages"][row["student_id"]] == pytest.approx(float(row["final_score"]), abs=0.011)
    assert set(report["mentions"].values()) <= {"ajourné", "passable", "assez bien", "bien", "très bien"}
