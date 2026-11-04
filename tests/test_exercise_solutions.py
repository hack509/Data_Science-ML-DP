"""Run each exercise's tests against its reference solution.

Learners run ``pytest exercises/...`` against their own ``exercise.py``. Here
we set LAB_EXERCISE_TARGET=solution so the same tests validate the solution.
"""

import os
import subprocess
import sys

import pytest

from lab.catalog import list_exercises
from lab.paths import ROOT

EXERCISES = list_exercises()
SLOW_IDS = {"A01", "A02", "A04", "A06"}


def _params():
    for exercise in EXERCISES:
        marks = [pytest.mark.slow] if exercise.id in SLOW_IDS else []
        yield pytest.param(exercise, id=exercise.id, marks=marks)


@pytest.mark.parametrize("exercise", list(_params()))
def test_solution_passes_exercise_tests(exercise):
    assert exercise.solution.exists(), f"missing solution file {exercise.solution}"
    env = dict(os.environ, LAB_EXERCISE_TARGET="solution", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(exercise.path), "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


@pytest.mark.parametrize("exercise", EXERCISES, ids=lambda e: e.id)
def test_exercise_has_required_files(exercise):
    for name in ("README.md", "exercise.py", "test_exercise.py"):
        assert (exercise.path / name).exists(), f"{exercise.path / name} missing"
    readme = exercise.readme.read_text(encoding="utf-8")
    for section in ("Objectifs", "Énoncé", "Données d'entrée", "Sortie attendue", "Contraintes", "Indices", "Vérification"):
        assert f"## {section}" in readme, f"{exercise.id}: section '{section}' missing in README"


@pytest.mark.parametrize("exercise", EXERCISES, ids=lambda e: e.id)
def test_exercise_file_does_not_contain_solution(exercise):
    source = (exercise.path / "exercise.py").read_text(encoding="utf-8")
    assert "NotImplementedError" in source, f"{exercise.id}: exercise.py should only contain TODO stubs"
