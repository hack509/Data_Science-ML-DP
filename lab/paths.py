"""Central definition of the repository paths.

Every script resolves files relative to the repository root so that it works
no matter which directory it is launched from.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COURSES_DIR = ROOT / "courses"
DOCS_DIR = ROOT / "docs"
DATASETS_DIR = ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
GENERATED_DIR = DATASETS_DIR / "generated"
PROCESSED_DIR = DATASETS_DIR / "processed"
EXERCISES_DIR = ROOT / "exercises"
SOLUTIONS_DIR = ROOT / "solutions"
PROJECTS_DIR = ROOT / "projects"
NOTEBOOKS_DIR = ROOT / "notebooks"
PROGRESS_DIR = ROOT / "progress"
PROGRESS_FILE = PROGRESS_DIR / "progress.json"
OUTPUTS_DIR = ROOT / "outputs"
EXPORTS_DIR = ROOT / "exports"

LEVELS = ("beginner", "intermediate", "advanced")
