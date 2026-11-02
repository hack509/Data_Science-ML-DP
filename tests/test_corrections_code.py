"""Run the code of the corrections files (quiz answers and guided exercises)."""

import importlib.util

import pytest

from lab.catalog import list_modules
from lab.lesson_runner import run_lesson

HAS_TORCH = importlib.util.find_spec("torch") is not None
MODULES = [m for m in list_modules() if m.corrections.exists()]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.id)
def test_corrections_code_runs(module):
    source = module.corrections.read_text(encoding="utf-8")
    if "import torch" in source and not HAS_TORCH:
        pytest.skip("PyTorch is not installed")
    result = run_lesson(module.corrections)
    assert result.ok, f"{module.corrections} failed:\n{result.stderr[-4000:]}"
