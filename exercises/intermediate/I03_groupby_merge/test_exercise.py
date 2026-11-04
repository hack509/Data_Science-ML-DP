import pandas as pd
import pytest

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.fixture(scope="module")
def tables():
    return load_csv("students"), load_csv("student_activities")


def test_orphans(tables):
    students, activities = tables
    assert ex.orphan_ids(students, activities) == [f"S{i:04d}" for i in range(901, 911)]


def test_merge(tables):
    students, activities = tables
    merged = ex.merge_activities(students, activities)
    assert len(merged) == 300
    assert merged["student_id"].is_unique
    assert (merged["club"] == "inconnu").sum() == 50
    assert merged["club"].notna().all()


def test_merge_rejects_duplicated_keys(tables):
    students, activities = tables
    broken = pd.concat([activities, activities.head(1)], ignore_index=True)
    with pytest.raises(Exception):
        ex.merge_activities(students, broken)


def test_summary(tables):
    merged = ex.merge_activities(*tables)
    summary = ex.summary_by_club(merged)
    assert list(summary.columns) == ["n", "pass_rate", "mean_final", "mean_study_hours"]
    assert summary.index[0] == "aucun"
    assert summary.loc["aucun", "n"] == 61
    assert summary.loc["aucun", "pass_rate"] == pytest.approx(0.574, abs=1e-3)
    assert summary["pass_rate"].is_monotonic_decreasing
    assert summary["n"].sum() == 300


def test_pivot(tables):
    merged = ex.merge_activities(*tables)
    pivot = ex.pivot_pass_rate(merged)
    assert list(pivot.index) == ["A", "B", "C"]
    assert "inconnu" in pivot.columns
    assert pivot.to_numpy().min() >= 0 and pivot.to_numpy().max() <= 1
