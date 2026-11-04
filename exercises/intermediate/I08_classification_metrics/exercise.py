"""I08 - Classification metrics from scratch. Standard library only."""

from __future__ import annotations


def confusion_counts(y_true, y_pred, positive=1) -> dict:
    raise NotImplementedError("TODO")


def accuracy(y_true, y_pred) -> float:
    raise NotImplementedError("TODO")


def precision(y_true, y_pred, positive=1) -> float:
    raise NotImplementedError("TODO")


def recall(y_true, y_pred, positive=1) -> float:
    raise NotImplementedError("TODO")


def f1(y_true, y_pred, positive=1) -> float:
    raise NotImplementedError("TODO")


def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    raise NotImplementedError("TODO")


def roc_auc(y_true, scores, positive=1) -> float:
    raise NotImplementedError("TODO")
