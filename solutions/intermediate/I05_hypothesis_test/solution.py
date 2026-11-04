"""Reference solution - I05 Confidence interval and hypothesis test.

Reflection questions (README):

* compare_groups(students, "passed", "study_hours_week", 1, 0) gives a
  difference of about 2.1 hours per week, t = 3.33, p = 0.001 and a Cohen's d
  of 0.40 (moderate effect): the difference is statistically significant.
* It is NOT a proof that studying more *causes* success. The test shows an
  association in observational data. In this synthetic dataset the generator
  does make study hours influence scores, but with real data other variables
  (motivation, prior level, free time) could explain both. Only a randomized
  experiment, or careful causal reasoning, can support a causal claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_confidence_interval(sample, confidence: float = 0.95) -> tuple[float, float, float]:
    """Student t interval for the mean: (mean, low, high)."""
    x = np.asarray(sample, dtype=float)
    if len(x) < 2:
        raise ValueError("at least two values are needed")
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=len(x) - 1)
    return float(mean), float(mean - t_crit * se), float(mean + t_crit * se)


def bootstrap_ci(sample, n_boot: int = 5000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    x = np.asarray(sample, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def welch_t_test(a, b) -> tuple[float, float, float]:
    """Two-sided Welch t-test computed by hand: (t, degrees of freedom, p-value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))  # Welch-Satterthwaite
    p_value = 2 * stats.t.sf(abs(t), df)
    return float(t), float(df), float(p_value)


def cohens_d(a, b) -> float:
    """Effect size: difference of means divided by the pooled standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


def compare_groups(df: pd.DataFrame, group_col: str, value_col: str, group_a, group_b, alpha: float = 0.05) -> dict:
    a = df.loc[df[group_col] == group_a, value_col].dropna().to_numpy()
    b = df.loc[df[group_col] == group_b, value_col].dropna().to_numpy()
    t, dof, p_value = welch_t_test(a, b)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "difference": float(a.mean() - b.mean()),
        "t": t,
        "df": dof,
        "p_value": p_value,
        "cohens_d": cohens_d(a, b),
        "reject_h0": bool(p_value < alpha),
    }
