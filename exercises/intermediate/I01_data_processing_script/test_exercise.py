import json

import pandas as pd
import pytest

from lab.data import dataset_path
from lab.exercise_loader import load_target

ex = load_target(__file__)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("order_id,date\nO1,2024-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ex.load(bad)


def test_summarize_columns_and_share():
    df = ex.clean(ex.load(dataset_path("sales_raw")))
    summary = ex.summarize(df, "product")
    assert list(summary.columns) == ["product", "n_orders", "quantity", "revenue", "share"]
    assert summary["revenue"].is_monotonic_decreasing
    assert summary["share"].sum() == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError):
        ex.summarize(df, "unknown_column")


def test_main_end_to_end(tmp_path, capsys):
    code = ex.main(["--input", str(dataset_path("sales_raw")), "--output-dir", str(tmp_path), "--by", "region"])
    assert code == 0
    clean = pd.read_csv(tmp_path / "clean_sales.csv")
    assert len(clean) == 379
    with open(tmp_path / "summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary[0] == {"region": "Nord", "n_orders": 122, "quantity": 358, "revenue": 20446.06, "share": 0.3148}
    assert "379" in capsys.readouterr().out


def test_main_reports_errors_with_exit_code(tmp_path, capsys):
    code = ex.main(["--input", str(tmp_path / "nope.csv"), "--output-dir", str(tmp_path)])
    assert code == 1
    assert "not found" in capsys.readouterr().out.lower()
