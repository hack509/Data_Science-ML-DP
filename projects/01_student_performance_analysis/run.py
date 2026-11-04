"""Run project 1: python projects/01_student_performance_analysis/run.py [--output-dir DIR]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))  # repository root, for `lab`

from student_analysis.analysis import run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Student performance analysis (project 1).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    print(f"Quality: {result['quality']}")
    print(f"Group test: {result['test']}")
    print(f"Report: {result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
