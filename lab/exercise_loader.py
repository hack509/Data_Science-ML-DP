"""Load either the learner's exercise file or the reference solution.

Each exercise folder contains ``exercise.py`` (the learner's work) and
``test_exercise.py``. The tests call :func:`load_target`, which imports:

* ``exercises/<level>/<id>/exercise.py`` by default (learner mode);
* ``solutions/<level>/<id>/solution.py`` when the environment variable
  ``LAB_EXERCISE_TARGET=solution`` is set (used by the repository test suite).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from lab.paths import EXERCISES_DIR, SOLUTIONS_DIR

TARGET_ENV = "LAB_EXERCISE_TARGET"


def target_mode() -> str:
    mode = os.environ.get(TARGET_ENV, "exercise").strip().lower()
    if mode not in {"exercise", "solution"}:
        raise ValueError(f"{TARGET_ENV} must be 'exercise' or 'solution', got {mode!r}")
    return mode


def target_path(test_file: str | Path) -> Path:
    exercise_dir = Path(test_file).resolve().parent
    relative = exercise_dir.relative_to(EXERCISES_DIR)
    if target_mode() == "solution":
        return SOLUTIONS_DIR / relative / "solution.py"
    return exercise_dir / "exercise.py"


def load_target(test_file: str | Path) -> ModuleType:
    """Import the file under test for the exercise containing ``test_file``."""
    path = target_path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"File under test not found: {path}")
    module_name = f"_lab_target_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
