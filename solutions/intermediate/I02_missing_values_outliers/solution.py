"""Reference solution - I02 Missing values and outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAUSIBLE_STUDY_HOURS = (0, 80)


def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def zscore_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    z = (series - series.mean()) / series.std()
    return z.abs() > threshold


def _parse_rate(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith("%"):
        return float(text.removesuffix("%")) / 100
    return float(text)


def clean_students(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_before": len(raw)}
    df = raw.drop_duplicates().copy()
    report["duplicates_removed"] = report["rows_before"] - len(df)

    df["group"] = df["group"].str.strip().str.upper()
    df["attendance_rate"] = df["attendance_rate"].map(_parse_rate)
    df["math_absent"] = df["math"].eq("absent")
    df["math"] = pd.to_numeric(df["math"], errors="coerce")
    report["math_absent"] = int(df["math_absent"].sum())

    low, high = PLAUSIBLE_STUDY_HOURS
    impossible = df["study_hours_week"].notna() & ~df["study_hours_week"].between(low, high)
    report["impossible_values"] = int(impossible.sum())
    df.loc[impossible, "study_hours_week"] = np.nan

    imputed = {}
    for column in ["study_hours_week", "sleep_hours", "physics"]:
        flag = df[column].isna()
        df[f"{column}_was_missing"] = flag
        imputed[column] = int(flag.sum())
    df["study_hours_week"] = df["study_hours_week"].fillna(df.groupby("group")["study_hours_week"].transform("median"))
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
    df["physics"] = df["physics"].fillna(df["physics"].median())
    report["imputed"] = imputed
    report["rows_after"] = len(df)

    df = df.sort_values("student_id").reset_index(drop=True)
    return df, report
