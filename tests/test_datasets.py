"""Dataset generation: reproducibility, content, loaders."""

import hashlib
import importlib.util
import json

import pandas as pd
import pytest

from lab import data
from lab.paths import DATASETS_DIR, ROOT

spec = importlib.util.spec_from_file_location("generate_datasets", ROOT / "scripts" / "generate_datasets.py")
generate_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_datasets)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    folder = tmp_path_factory.mktemp("datasets")
    generate_datasets.generate_all(folder, verbose=False)
    return folder


def test_all_registered_datasets_exist():
    assert data.missing_datasets() == []


@pytest.mark.parametrize("name", [n for n, i in data.DATASETS.items() if i.relative_path.endswith((".csv", ".json"))])
def test_regeneration_is_identical(regenerated, name):
    relative = data.DATASETS[name].relative_path
    expected = (DATASETS_DIR / relative).read_bytes().replace(b"\r\n", b"\n")
    assert (regenerated / relative).read_bytes() == expected, f"{relative} is not reproducible"


def test_expected_sizes():
    sizes = {"students": 300, "students_raw": 308, "sales": 3000, "sales_raw": 410, "house_prices": 800,
             "customers": 500, "sms_spam": 1200, "reviews": 800, "binary": 600, "multiclass": 900, "regression": 500, "shapes_index": 300}
    for name, rows in sizes.items():
        assert len(data.load_csv(name)) == rows, name


def test_json_datasets():
    assert len(data.load_json("orders_json")) == 40
    documents = data.load_json("knowledge_base")
    assert len(documents) == 14 and {"doc_id", "title", "text"} <= set(documents[0])


def test_students_consistency():
    df = data.load_csv("students")
    assert (df[["math", "physics", "french"]].stack().between(0, 20)).all()
    assert ((df["final_score"] >= 10).astype(int) == df["passed"]).all()


def test_raw_files_contain_the_documented_problems():
    raw = data.load_csv("students_raw")
    assert raw.duplicated().sum() == 8
    assert raw["math"].eq("absent").any()
    assert raw["attendance_rate"].astype(str).str.endswith("%").any()
    sales = data.load_csv("sales_raw")
    assert sales["unit_price"].astype(str).str.contains("EUR").any()


def test_missing_dataset_message(monkeypatch, tmp_path):
    monkeypatch.setattr(data.paths, "DATASETS_DIR", tmp_path)
    with pytest.raises(data.MissingDatasetError, match="generate_datasets"):
        data.dataset_path("students")
    with pytest.raises(KeyError):
        data.dataset_path("does_not_exist")


def test_utf8_accents_survive():
    df = pd.read_csv(data.dataset_path("sms_spam"), encoding="utf-8")
    assert df["text"].str.contains("é").any()
    assert "é" in json.dumps(data.load_json("knowledge_base"), ensure_ascii=False)
