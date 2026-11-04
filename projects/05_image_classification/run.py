"""Project 5 - image classification with a CNN (CPU).

    python projects/05_image_classification/run.py [--quick]
    python projects/05_image_classification/run.py predict image1.png image2.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--quick", action="store_true", help="a few epochs only (tests)")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("predict")
    p.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        from shape_classifier.training import predict_files, run
    except ImportError as exc:
        print(f"PyTorch is required for this project ({exc}). Install it with: python -m pip install -r requirements-torch.txt")
        return 1

    if args.command == "predict":
        for row in predict_files(args.images, args.output_dir):
            print(json.dumps(row))
        return 0
    result = run(args.output_dir, quick=args.quick)
    print(json.dumps(result["metrics"], indent=2))
    print(f"outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
