"""Build one Jupyter notebook per lesson from the Markdown sources.

Markdown text becomes Markdown cells, every ```python block becomes a code
cell, and ```python no-run blocks stay as Markdown (illustrative code). The
first cell moves the working directory to the repository root so that the
relative paths of the lessons (datasets/..., outputs/...) work.

Usage (from the repository root):
    python scripts/build_notebooks.py
    jupyter lab notebooks/      # or open the .ipynb files in VS Code
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.catalog import Lesson, list_lessons  # noqa: E402
from lab.paths import NOTEBOOKS_DIR, ROOT  # noqa: E402

FOLDERS = {
    "01": "python_basics",
    "02": "data_analysis",
    "03": "data_analysis",
    "04": "data_analysis",
    "05": "machine_learning",
    "10": "machine_learning",
    "06": "deep_learning",
    "07": "deep_learning",
    "08": "deep_learning",
    "09": "deep_learning",
}

SETUP_CELL = """# Setup: run the lesson code from the repository root (paths such as datasets/...)
import os
import sys
from pathlib import Path

root = Path.cwd()
while not (root / "lab").is_dir() and root != root.parent:
    root = root.parent
os.chdir(root)
sys.path.insert(0, str(root))
print("working directory:", root)"""


def split_cells(text: str) -> list[tuple[str, str]]:
    """Return (kind, source) pairs: kind is 'markdown' or 'code'."""
    cells, buffer = [], []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0

    def flush():
        content = "\n".join(buffer).strip("\n")
        if content.strip():
            cells.append(("markdown", content))
        buffer.clear()

    while i < len(lines):
        if lines[i].strip() == "```python":
            flush()
            code = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                code.append(lines[i])
                i += 1
            cells.append(("code", "\n".join(code)))
            i += 1
            continue
        buffer.append(lines[i])
        i += 1
    flush()
    return cells


def lesson_notebook(lesson: Lesson):
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata["language_info"] = {"name": "python"}
    source_note = (f"*Notebook généré automatiquement à partir de `{lesson.relative_path}` "
                   "par `scripts/build_notebooks.py`. Modifiez la leçon Markdown, puis régénérez.*")
    notebook.cells.append(new_markdown_cell(source_note))
    notebook.cells.append(new_code_cell(SETUP_CELL))
    for kind, source in split_cells(lesson.path.read_text(encoding="utf-8")):
        notebook.cells.append(new_code_cell(source) if kind == "code" else new_markdown_cell(source))
    nbformat.validate(notebook)
    return notebook


def notebook_path(lesson: Lesson, output_dir: Path = NOTEBOOKS_DIR) -> Path:
    folder = FOLDERS.get(lesson.module_id[:2], "other")
    return output_dir / folder / f"{lesson.module_id[:2]}_{lesson.slug}.ipynb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=NOTEBOOKS_DIR)
    args = parser.parse_args(argv)
    try:
        import nbformat
    except ImportError:
        print("nbformat is not installed. Install it once (online) with: python -m pip install -r requirements.txt")
        return 1
    count = 0
    for lesson in list_lessons():
        path = notebook_path(lesson, args.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(lesson_notebook(lesson), str(path))
        count += 1
    print(f"[OK] {count} notebooks written under {args.output_dir.relative_to(ROOT) if args.output_dir.is_relative_to(ROOT) else args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
