"""Test-set metrics with bootstrap intervals and subgroup analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Config


def _bootstrap(y, proba, threshold, config: Config) -> dict[str, list[float]]:
    rng = np.random.default_rng(config.seed)
    y, proba = np.asarray(y), np.asarray(proba)
    values = {"roc_auc": [], "accuracy": []}
    for idx in rng.integers(0, len(y), size=(config.n_bootstrap, len(y))):
        if len(np.unique(y[idx])) < 2:
            continue
        values["roc_auc"].append(roc_auc_score(y[idx], proba[idx]))
        values["accuracy"].append(accuracy_score(y[idx], proba[idx] >= threshold))
    return {k: [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])] for k, vals in values.items()}


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, config: Config) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= config.decision_threshold).astype(int)
    metrics = {
        "n_test": int(len(y_test)),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "ci95": _bootstrap(y_test, proba, config.decision_threshold, config),
    }
    frame = pd.DataFrame({"group": X_test[config.subgroup_column].to_numpy(), "y": y_test.to_numpy(), "pred": pred, "proba": proba})
    subgroups = {}
    for group, part in frame.groupby("group"):
        subgroups[str(group)] = {
            "n": int(len(part)),
            "accuracy": round(float((part["y"] == part["pred"]).mean()), 4),
            "selection_rate": round(float(part["pred"].mean()), 4),
            "true_positive_rate": round(float(part.loc[part["y"] == 1, "pred"].mean()), 4) if (part["y"] == 1).any() else None,
        }
    metrics["subgroups"] = subgroups
    return metrics
