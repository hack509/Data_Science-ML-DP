"""Load raw files, clean them, build the target and run data-quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lab.data import load_csv

from .config import Config

WEIGHTS = {"math": 0.4, "physics": 0.3, "french": 0.3}


class DataQualityError(ValueError):
    """Raised when the cleaned data violates a quality rule."""


def _parse_rate(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return float(text.removesuffix("%")) / 100 if text.endswith("%") else float(text)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw student export. Missing feature values are left as NaN: imputation belongs to the model pipeline."""
    log = {"rows_raw": len(raw)}
    df = raw.drop_duplicates().copy()
    log["duplicates_removed"] = log["rows_raw"] - len(df)
    df["group"] = df["group"].str.strip().str.upper()
    df["attendance_rate"] = df["attendance_rate"].map(_parse_rate)
    df["math"] = pd.to_numeric(df["math"], errors="coerce")
    impossible = df["study_hours_week"].notna() & ~df["study_hours_week"].between(0, 80)
    log["impossible_values_set_to_missing"] = int(impossible.sum())
    df.loc[impossible, "study_hours_week"] = np.nan
    return df.reset_index(drop=True), log


def build_target(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict]:
    """Target = final score >= pass mark.

    Students without a math or physics score cannot be labelled: their target is unknown, and imputing
    a score would invent the very quantity we want to predict, so these rows are excluded.
    """
    final = sum(df[subject] * weight for subject, weight in WEIGHTS.items())
    labelled = df.assign(final_score=final.round(2))
    unlabelled = int(labelled["final_score"].isna().sum())
    labelled = labelled[labelled["final_score"].notna()].copy()
    labelled["passed"] = (labelled["final_score"] >= config.pass_mark).astype(int)
    return labelled, {"rows_without_label": unlabelled, "rows_labelled": len(labelled), "pass_rate": round(float(labelled["passed"].mean()), 4)}


def add_activities(df: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(activities, on="student_id", how="left", validate="one_to_one")
    merged["club"] = merged["club"].fillna("inconnu")
    return merged


def quality_checks(df: pd.DataFrame, config: Config) -> list[str]:
    """Return the list of passed checks; raise DataQualityError on the first failure."""
    checks = {  # evaluated lazily and in order, so a missing column is reported as a quality error
        "all feature and target columns present": lambda: set(config.features + ["passed"]) <= set(df.columns),
        "student_id is unique": lambda: df["student_id"].is_unique,
        "group in {A, B, C}": lambda: df["group"].isin(["A", "B", "C"]).all(),
        "attendance in [0, 1]": lambda: df["attendance_rate"].dropna().between(0, 1).all(),
        "study hours in [0, 80]": lambda: df["study_hours_week"].dropna().between(0, 80).all(),
        "target is binary": lambda: set(df["passed"].unique()) <= {0, 1},
        "at least 100 labelled rows": lambda: len(df) >= 100,
    }
    for name, check in checks.items():
        if not check():
            raise DataQualityError(f"data-quality check failed: {name}")
    return list(checks)


def load_dataset(config: Config) -> tuple[pd.DataFrame, dict]:
    raw = load_csv("students_raw")
    cleaned, clean_log = clean(raw)
    labelled, target_log = build_target(cleaned, config)
    dataset = add_activities(labelled, load_csv("student_activities"))
    checks = quality_checks(dataset, config)
    return dataset, {**clean_log, **target_log, "quality_checks_passed": checks}
