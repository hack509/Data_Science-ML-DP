"""Discover modules, lessons, exercises and projects from the file system.

The file system is the single source of truth: adding a Markdown lesson in a
module folder is enough for it to appear in the index, the web interface,
the PDF export and the progress tracker.

Naming conventions:
* modules:   ``courses/NN_name/README.md`` (module overview)
* lessons:   ``courses/NN_name/NN_lesson.md``
* answers:   ``courses/NN_name/corrections.md`` (quiz + guided exercises)
* exercises: ``exercises/<level>/<ID>_name/README.md`` (ID such as B01)
* projects:  ``projects/NN_name/README.md``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lab.markdown_lite import first_heading
from lab.paths import COURSES_DIR, EXERCISES_DIR, LEVELS, PROJECTS_DIR, ROOT, SOLUTIONS_DIR

_NUMBERED = re.compile(r"^\d{2}_")


def _title(path: Path, fallback: str) -> str:
    try:
        return first_heading(path.read_text(encoding="utf-8")) or fallback
    except FileNotFoundError:
        return fallback


@dataclass
class Lesson:
    module_id: str
    slug: str
    path: Path
    title: str

    @property
    def id(self) -> str:
        return f"{self.module_id}/{self.slug}"

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


@dataclass
class Module:
    id: str
    number: int
    path: Path
    title: str
    lessons: list[Lesson] = field(default_factory=list)

    @property
    def readme(self) -> Path:
        return self.path / "README.md"

    @property
    def corrections(self) -> Path:
        return self.path / "corrections.md"


@dataclass
class Exercise:
    id: str
    level: str
    path: Path
    title: str

    @property
    def readme(self) -> Path:
        return self.path / "README.md"

    @property
    def solution(self) -> Path:
        return SOLUTIONS_DIR / self.level / self.path.name / "solution.py"


@dataclass
class Project:
    id: str
    number: int
    path: Path
    title: str

    @property
    def readme(self) -> Path:
        return self.path / "README.md"


def list_modules(courses_dir: Path = COURSES_DIR) -> list[Module]:
    modules = []
    for path in sorted(p for p in courses_dir.iterdir() if p.is_dir() and _NUMBERED.match(p.name)):
        module = Module(path.name, int(path.name[:2]), path, _title(path / "README.md", path.name))
        for lesson_path in sorted(path.glob("[0-9][0-9]_*.md")):
            module.lessons.append(
                Lesson(path.name, lesson_path.stem, lesson_path, _title(lesson_path, lesson_path.stem))
            )
        modules.append(module)
    return modules


def list_lessons(courses_dir: Path = COURSES_DIR) -> list[Lesson]:
    return [lesson for module in list_modules(courses_dir) for lesson in module.lessons]


def get_lesson(lesson_id: str) -> Lesson | None:
    return next((lesson for lesson in list_lessons() if lesson.id == lesson_id), None)


def list_exercises(exercises_dir: Path = EXERCISES_DIR) -> list[Exercise]:
    exercises = []
    for level in LEVELS:
        level_dir = exercises_dir / level
        if not level_dir.exists():
            continue
        for path in sorted(p for p in level_dir.iterdir() if p.is_dir() and (p / "README.md").exists()):
            exercise_id = path.name.split("_", 1)[0]
            exercises.append(Exercise(exercise_id, level, path, _title(path / "README.md", path.name)))
    return exercises


def get_exercise(exercise_id: str) -> Exercise | None:
    return next((e for e in list_exercises() if e.id.upper() == exercise_id.upper()), None)


def list_projects(projects_dir: Path = PROJECTS_DIR) -> list[Project]:
    projects = []
    for path in sorted(p for p in projects_dir.iterdir() if p.is_dir() and _NUMBERED.match(p.name)):
        projects.append(Project(path.name, int(path.name[:2]), path, _title(path / "README.md", path.name)))
    return projects
