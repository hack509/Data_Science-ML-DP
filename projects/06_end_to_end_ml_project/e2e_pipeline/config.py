"""Typed configuration loaded from a JSON file."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 0
    test_size: float = 0.2
    cv_folds: int = 5
    numeric_features: list[str] = field(default_factory=lambda: ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score", "volunteer_hours_year"])
    categorical_features: list[str] = field(default_factory=lambda: ["group", "club"])
    subgroup_column: str = "group"
    pass_mark: float = 10.0
    decision_threshold: float = 0.5
    n_bootstrap: int = 1000

    @property
    def features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def validate(self) -> None:
        if not 0 < self.test_size < 0.5:
            raise ValueError("test_size must be between 0 and 0.5")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("decision_threshold must be between 0 and 1")
        leaky = {"math", "physics", "french", "final_score", "passed"} & set(self.features)
        if leaky:
            raise ValueError(f"these columns leak the target and cannot be features: {sorted(leaky)}")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None, **overrides) -> Config:
    values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    values.update(overrides)
    unknown = set(values) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    config = Config(**values)
    config.validate()
    return config
