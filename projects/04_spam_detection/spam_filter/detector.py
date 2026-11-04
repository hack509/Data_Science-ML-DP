"""Leakage-free spam detector: group split, model comparison, threshold, error analysis."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from lab.data import load_csv

SEED = 0
MIN_PRECISION = 0.98  # a legitimate message lost in the spam folder is costly


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"https?://\S+", " urltoken ", text)
    text = re.sub(r"\b0\d(?:[ .]?\d{2}){4}\b", " phonetoken ", text)
    return re.sub(r"\d+(?:[.,]\d+)?", " numtoken ", text)


def split(df: pd.DataFrame):
    """Train / validation / test (60/20/20) with identical texts kept together."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    rest_idx, test_idx = next(outer.split(df, groups=df["text"]))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, val_idx = next(inner.split(rest, groups=rest["text"]))
    return rest.iloc[train_idx], rest.iloc[val_idx], df.iloc[test_idx]


def candidates() -> dict[str, Pipeline]:
    """Ordered by preference: on a validation tie, the first (explainable logistic model) wins."""
    return {
        "logistic_words": Pipeline([("vec", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
                                    ("clf", LogisticRegression(max_iter=2000, C=10))]),
        "naive_bayes_words": Pipeline([("vec", CountVectorizer(preprocessor=normalize, ngram_range=(1, 2))), ("clf", MultinomialNB())]),
        "logistic_chars": Pipeline([("vec", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
                                    ("clf", LogisticRegression(max_iter=2000, C=10))]),
    }


def is_spam(labels) -> np.ndarray:
    return (np.asarray(labels) == "spam").astype(int)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, min_precision: float = MIN_PRECISION) -> float:
    """Lowest threshold whose precision reaches min_precision (maximizes recall under that constraint)."""
    for threshold in np.unique(np.round(scores, 4)):
        pred = scores >= threshold
        if pred.sum() and precision_score(y_true, pred) >= min_precision:
            return float(threshold)
    return 0.5


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"precision": round(precision_score(y_true, pred, zero_division=0), 4), "recall": round(recall_score(y_true, pred), 4),
            "f1": round(f1_score(y_true, pred), 4), "false_positives": int(fp), "false_negatives": int(fn), "n": int(len(y_true))}


def explain(model: Pipeline, text: str, top: int = 5) -> list[tuple[str, float]]:
    """Terms of the message with the largest contribution (tf-idf x coefficient) to the spam score."""
    vector = model.named_steps["vec"].transform([text])
    coefficients = model.named_steps["clf"].coef_[0]
    names = model.named_steps["vec"].get_feature_names_out()
    contributions = {names[j]: float(vector[0, j] * coefficients[j]) for j in vector.nonzero()[1]}
    return sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]


def run(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_csv("sms_spam")
    train, val, test = split(df)
    overlap = len(set(train["text"]) & set(test["text"]))

    validation = {}
    for name, model in candidates().items():
        model.fit(train["text"], train["label"])
        validation[name] = f1_score(is_spam(val["label"]), is_spam(model.predict(val["text"])))
    best_name = max(validation, key=validation.get)
    model = candidates()[best_name].fit(train["text"], train["label"])
    spam_column = list(model.classes_).index("spam")
    threshold = choose_threshold(is_spam(val["label"]), model.predict_proba(val["text"])[:, spam_column])

    test_scores = model.predict_proba(test["text"])[:, spam_column]
    test_pred = (test_scores >= threshold).astype(int)
    results = {
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "train_test_text_overlap": overlap,
        "validation_f1": {k: round(v, 4) for k, v in validation.items()},
        "selected_model": best_name,
        "threshold": round(threshold, 4),
        "test_default_threshold": metrics(is_spam(test["label"]), (test_scores >= 0.5).astype(int)),
        "test_chosen_threshold": metrics(is_spam(test["label"]), test_pred),
    }
    errors = test.assign(spam_probability=test_scores.round(3), predicted=np.where(test_pred, "spam", "ham"))
    errors = errors[errors["predicted"] != errors["label"]]
    errors.to_csv(output_dir / "errors.csv", index=False)
    joblib.dump({"model": model, "threshold": threshold}, output_dir / "spam_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**results, "errors": errors, "model": model}


def classify(messages: list[str], model_path: Path) -> list[dict]:
    bundle = joblib.load(model_path)
    model, threshold = bundle["model"], bundle["threshold"]
    spam_column = list(model.classes_).index("spam")
    probabilities = model.predict_proba(messages)[:, spam_column]
    return [{"text": m, "spam_probability": round(float(p), 3), "label": "spam" if p >= threshold else "ham"} for m, p in zip(messages, probabilities)]
