"""I03 - Merging tables and aggregations. Complete the functions (see README.md)."""

from __future__ import annotations

import pandas as pd


def orphan_ids(students: pd.DataFrame, activities: pd.DataFrame) -> list[str]:
    raise NotImplementedError("TODO")


def merge_activities(students: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TODO")


def summary_by_club(merged: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TODO")


def pivot_pass_rate(merged: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("TODO")
