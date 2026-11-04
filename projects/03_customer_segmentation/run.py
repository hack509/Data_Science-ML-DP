"""Run project 3: python projects/03_customer_segmentation/run.py [--output-dir DIR]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from segmentation.clustering import run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Customer segmentation (project 3).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    print(result["selection"].round(3).to_string())
    print(f"k={result['k']} | stability={result['stability']:.3f} | ARI vs generator={result['ari']:.3f}")
    print(result["profiles"].to_string())
    print(f"Report: {result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
