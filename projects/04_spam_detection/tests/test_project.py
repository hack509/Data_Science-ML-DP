import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from spam_filter.detector import choose_threshold, classify, normalize, run  # noqa: E402


def test_normalize():
    assert normalize("GAGNÉ 500 EUR http://x.example 08 99 12 34 56") == "gagne  numtoken  eur  urltoken   phonetoken "


def test_choose_threshold():
    y = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.55, 0.8, 0.9])
    assert choose_threshold(y, scores, min_precision=1.0) == 0.8


def test_run_and_classify(tmp_path):
    result = run(tmp_path)
    assert result["train_test_text_overlap"] == 0
    chosen = result["test_chosen_threshold"]
    assert chosen["f1"] > 0.9 and chosen["precision"] >= 0.95
    assert (tmp_path / "errors.csv").exists() and (tmp_path / "metrics.json").exists()
    rows = classify(["FÉLICITATIONS ! Vous avez gagné 1000 EUR, appelez le 08 99 11 22 33", "On se retrouve au café à 18h ?"],
                    tmp_path / "spam_model.joblib")
    assert [r["label"] for r in rows] == ["spam", "ham"]
