"""Run the executable code of every lesson (blocks tagged ```python)."""

import importlib.util

import pytest

from lab.catalog import list_lessons
from lab.lesson_runner import run_lesson

TORCH_MODULES = ("06_", "07_", "08_", "09_", "10_")
HAS_TORCH = importlib.util.find_spec("torch") is not None

LESSONS = list_lessons()


def _marks(lesson):
    marks = []
    if lesson.module_id.startswith(TORCH_MODULES):
        marks.append(pytest.mark.slow)
    return marks


@pytest.mark.parametrize(
    "lesson", [pytest.param(lesson, id=lesson.id, marks=_marks(lesson)) for lesson in LESSONS]
)
def test_lesson_code_runs(lesson):
    source = lesson.path.read_text(encoding="utf-8")
    if "import torch" in source and not HAS_TORCH:
        pytest.skip("PyTorch is not installed")
    result = run_lesson(lesson.path)
    assert result.ok, f"{lesson.relative_path} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"


def test_every_module_has_lessons():
    from lab.catalog import list_modules

    for module in list_modules():
        assert module.lessons, f"{module.id} has no lesson"
        assert module.readme.exists(), f"{module.id} has no README.md"
        assert module.corrections.exists(), f"{module.id} has no corrections.md"
