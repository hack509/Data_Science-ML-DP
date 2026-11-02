"""Local progress tracking stored in ``progress/progress.json``.

Nothing is ever marked as completed automatically: every change goes through
an explicit call (command line ``scripts/progress.py`` or a button in the
local web interface).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from lab.paths import PROGRESS_FILE

SCHEMA_VERSION = 1
CATEGORIES = ("lessons", "exercises", "projects")


def empty_progress() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "learner": "",
        "current_module": "01_python_for_data_science",
        "lessons": {},
        "exercises": {},
        "projects": {},
        "quizzes": {},
        "updated_at": None,
    }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProgressTracker:
    """Read and update the learner's progress file."""

    def __init__(self, path: Path = PROGRESS_FILE):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return empty_progress()
        with self.path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        base = empty_progress()
        base.update(data)  # keeps unknown keys, fills missing ones
        if base["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported progress schema version: {base['schema_version']}")
        return base

    def save(self) -> None:
        """Write atomically so that a crash never corrupts the file."""
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, self.path)

    # -- updates (always explicit) -------------------------------------------
    def set_completed(self, category: str, item_id: str, completed: bool = True, note: str = "") -> None:
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        if completed:
            self.data[category][item_id] = {"completed": True, "completed_at": _now(), "note": note}
        else:
            self.data[category].pop(item_id, None)

    def record_quiz(self, module_id: str, score: float, max_score: float) -> None:
        if not 0 <= score <= max_score or max_score <= 0:
            raise ValueError("score must be between 0 and max_score, and max_score > 0")
        history = self.data["quizzes"].setdefault(module_id, [])
        history.append({"score": score, "max_score": max_score, "date": _now()})

    def set_current_module(self, module_id: str) -> None:
        self.data["current_module"] = module_id

    # -- queries --------------------------------------------------------------
    def is_completed(self, category: str, item_id: str) -> bool:
        return bool(self.data[category].get(item_id, {}).get("completed"))

    def best_quiz(self, module_id: str) -> dict | None:
        history = self.data["quizzes"].get(module_id, [])
        return max(history, key=lambda h: h["score"] / h["max_score"]) if history else None

    def completion_ratio(self, category: str, all_ids: list[str]) -> float:
        if not all_ids:
            return 0.0
        return sum(self.is_completed(category, i) for i in all_ids) / len(all_ids)
