"""Registry and loaders for the local datasets.

All datasets are created by ``scripts/generate_datasets.py``. Loaders never
download anything: when a file is missing they raise
:class:`MissingDatasetError` with the exact command to regenerate it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lab import paths

REGENERATE_HINT = "python scripts/generate_datasets.py"


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    relative_path: str
    description: str

    @property
    def path(self) -> Path:
        return paths.DATASETS_DIR / self.relative_path


DATASETS: dict[str, DatasetInfo] = {
    info.name: info
    for info in [
        DatasetInfo("students", "generated/students.csv", "Student scores and study habits (clean)"),
        DatasetInfo("students_raw", "raw/students_raw.csv", "Student scores with quality problems"),
        DatasetInfo("student_activities", "generated/student_activities.csv", "Extracurricular activities per student"),
        DatasetInfo("sales", "generated/sales.csv", "Daily sales records (clean)"),
        DatasetInfo("sales_raw", "raw/sales_raw.csv", "Sales records with quality problems"),
        DatasetInfo("orders_json", "raw/orders.json", "Nested JSON customer orders"),
        DatasetInfo("house_prices", "generated/house_prices.csv", "Synthetic house prices"),
        DatasetInfo("customers", "generated/customers.csv", "Customer behaviour for segmentation"),
        DatasetInfo("customers_true_segments", "generated/customers_true_segments.csv", "Hidden generating segments (evaluation only)"),
        DatasetInfo("sms_spam", "generated/sms_spam.csv", "Synthetic French SMS labelled ham/spam"),
        DatasetInfo("reviews", "generated/reviews_fr.csv", "Synthetic French product reviews with sentiment"),
        DatasetInfo("binary", "generated/binary_classification.csv", "Two-class tabular classification"),
        DatasetInfo("multiclass", "generated/multiclass_classification.csv", "Three-class tabular classification"),
        DatasetInfo("regression", "generated/regression.csv", "Tabular regression with a non-linear term"),
        DatasetInfo("shapes_index", "generated/shapes/index.csv", "Index of the 28x28 shape images"),
        DatasetInfo("shapes_npz", "generated/shapes.npz", "Shape images as NumPy arrays"),
        DatasetInfo("knowledge_base", "generated/knowledge_base.json", "Small French document collection for RAG"),
    ]
}


class MissingDatasetError(FileNotFoundError):
    """A dataset file is missing; the message explains how to regenerate it."""


def dataset_path(name: str) -> Path:
    """Return the path of a registered dataset, checking that it exists."""
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {known}")
    path = DATASETS[name].path
    if not path.exists():
        raise MissingDatasetError(
            f"Dataset '{name}' is missing (datasets/{DATASETS[name].relative_path}).\n"
            f"Generate it locally (no Internet needed) with:\n    {REGENERATE_HINT}"
        )
    return path


def load_csv(name: str, **read_csv_kwargs):
    """Load a registered CSV dataset into a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(dataset_path(name), **read_csv_kwargs)


def load_json(name: str):
    """Load a registered JSON dataset as Python objects."""
    with dataset_path(name).open(encoding="utf-8") as handle:
        return json.load(handle)


def missing_datasets() -> list[str]:
    """Names of the registered datasets that are not present on disk."""
    return [name for name, info in DATASETS.items() if not info.path.exists()]
