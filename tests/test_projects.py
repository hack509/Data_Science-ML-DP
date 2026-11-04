"""Run each project's own test suite in a separate process (isolated imports)."""

import os
import subprocess
import sys

import pytest

from lab.catalog import list_projects
from lab.paths import ROOT

PROJECTS = list_projects()
SLOW = {"05_image_classification"}
REQUIRED_SECTIONS = [
    "Énoncé du problème", "Objectifs d'apprentissage", "Données", "Architecture", "Étapes",
    "Lancer la solution de référence", "Indicateurs et résultats attendus", "Défis pour aller plus loin",
    "Solution de référence", "Pistes d'amélioration",
]


def test_six_projects_exist():
    assert len(PROJECTS) == 6


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.id)
def test_project_readme_sections(project):
    text = project.readme.read_text(encoding="utf-8")
    missing = [s for s in REQUIRED_SECTIONS if f"## {s}" not in text]
    assert not missing, f"{project.id}: missing sections {missing}"
    assert (project.path / "run.py").exists() and (project.path / "tests").is_dir()


@pytest.mark.parametrize(
    "project", [pytest.param(p, id=p.id, marks=[pytest.mark.slow] if p.id in SLOW else []) for p in PROJECTS]
)
def test_project_test_suite_passes(project):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(project.path / "tests"), "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
