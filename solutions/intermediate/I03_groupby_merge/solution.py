"""Reference solution - I03 Merging tables and aggregations."""

from __future__ import annotations

import pandas as pd


def orphan_ids(students: pd.DataFrame, activities: pd.DataFrame) -> list[str]:
    """Activity ids that do not exist in the students table, sorted."""
    known = set(students["student_id"])
    return sorted(set(activities["student_id"]) - known)


def merge_activities(students: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    merged = students.merge(activities, on="student_id", how="left", validate="one_to_one")
    if len(merged) != len(students):
        raise AssertionError("the merge changed the number of students")
    merged["club"] = merged["club"].fillna("inconnu")
    return merged


def summary_by_club(merged: pd.DataFrame) -> pd.DataFrame:
    summary = merged.groupby("club").agg(
        n=("student_id", "size"),
        pass_rate=("passed", "mean"),
        mean_final=("final_score", "mean"),
        mean_study_hours=("study_hours_week", "mean"),
    )
    return summary.round(3).sort_values(["pass_rate", "n"], ascending=[False, False])


def pivot_pass_rate(merged: pd.DataFrame) -> pd.DataFrame:
    return merged.pivot_table(index="group", columns="club", values="passed", aggfunc="mean").round(3)
