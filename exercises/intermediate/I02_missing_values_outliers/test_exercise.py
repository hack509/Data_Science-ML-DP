import numpy as np
import pandas as pd
import pytest

from lab.data import load_csv
from lab.exercise_loader import load_target

ex = load_target(__file__)


@pytest.fixture(scope="module")
def raw():
    return load_csv("students_raw")


@pytest.fixture(scope="module")
def result(raw):
    return ex.clean_students(raw)


def test_detectors():
    values = pd.Series([10, 11, 12, 13, 12, 11, 100])
    assert ex.iqr_outliers(values).tolist() == [False] * 6 + [True]
    normal = pd.Series(np.random.default_rng(0).normal(0, 1, 1000))
    with_extreme = pd.concat([normal, pd.Series([12.0])], ignore_index=True)
    flags = ex.zscore_outliers(with_extreme)
    assert flags.iloc[-1] and flags.sum() < 10


def test_report(result):
    _, report = result
    assert report == {
        "rows_before": 308, "duplicates_removed": 8, "math_absent": 6, "impossible_values": 3,
        "imputed": {"study_hours_week": 21, "sleep_hours": 15, "physics": 9}, "rows_after": 300,
    }


def test_clean_frame(raw, result):
    df, _ = result
    assert raw.shape == (308, 9), "raw must not be modified"
    assert df["student_id"].is_unique and df["student_id"].is_monotonic_increasing
    assert sorted(df["group"].unique()) == ["A", "B", "C"]
    assert df["attendance_rate"].between(0, 1).all()
    assert df[["study_hours_week", "sleep_hours", "physics"]].notna().all().all()
    assert df["math"].isna().sum() == 6 and df["math_absent"].sum() == 6
    assert df["study_hours_week"].between(0, 80).all()


def test_matches_clean_dataset_where_not_imputed(result):
    df, _ = result
    truth = load_csv("students").set_index("student_id")
    cleaned = df.set_index("student_id")
    kept = ~cleaned["study_hours_week_was_missing"]
    pd.testing.assert_series_equal(
        cleaned.loc[kept, "study_hours_week"], truth.loc[kept[kept].index, "study_hours_week"], check_names=False
    )
    np.testing.assert_allclose(cleaned["attendance_rate"], truth["attendance_rate"], atol=6e-4)


def test_group_median_imputation(result):
    df, _ = result
    imputed = df[df["study_hours_week_was_missing"]]
    observed = df[~df["study_hours_week_was_missing"]]
    medians = observed.groupby("group")["study_hours_week"].median()
    for _, row in imputed.iterrows():
        assert row["study_hours_week"] == pytest.approx(medians[row["group"]])
