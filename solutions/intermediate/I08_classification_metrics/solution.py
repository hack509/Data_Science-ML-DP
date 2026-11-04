"""Reference solution - I08 Classification metrics from scratch (no scikit-learn)."""

from __future__ import annotations


def confusion_counts(y_true, y_pred, positive=1) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for truth, pred in zip(y_true, y_pred):
        if pred == positive:
            counts["tp" if truth == positive else "fp"] += 1
        else:
            counts["fn" if truth == positive else "tn"] += 1
    return counts


def _safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        raise ValueError("empty input")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def precision(y_true, y_pred, positive=1) -> float:
    c = confusion_counts(y_true, y_pred, positive)
    return _safe_divide(c["tp"], c["tp"] + c["fp"])


def recall(y_true, y_pred, positive=1) -> float:
    c = confusion_counts(y_true, y_pred, positive)
    return _safe_divide(c["tp"], c["tp"] + c["fn"])


def f1(y_true, y_pred, positive=1) -> float:
    p, r = precision(y_true, y_pred, positive), recall(y_true, y_pred, positive)
    return _safe_divide(2 * p * r, p + r)


def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    """Rows = true labels, columns = predicted labels, in the order of `labels`."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        matrix[index[truth]][index[pred]] += 1
    return matrix


def roc_auc(y_true, scores, positive=1) -> float:
    """Probability that a random positive gets a higher score than a random negative (ties count 1/2).

    Computed with ranks (Mann-Whitney U statistic), in O(n log n).
    """
    n_pos = sum(1 for t in y_true if t == positive)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes are needed to compute the AUC")
    ordered = sorted((s, t == positive) for t, s in zip(y_true, scores))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2  # ranks start at 1; tied scores share the mean rank
        rank_sum += average_rank * sum(1 for k in range(i, j) if ordered[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
