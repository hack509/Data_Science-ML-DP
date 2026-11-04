"""Project 6 - end-to-end ML pipeline.

    python projects/06_end_to_end_ml_project/run.py train [--config config.json] [--output-dir DIR]
    python projects/06_end_to_end_ml_project/run.py predict new_students.csv predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from e2e_pipeline.config import load_config  # noqa: E402
from e2e_pipeline.data import DataQualityError  # noqa: E402
from e2e_pipeline.pipeline import predict, train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    p = sub.add_parser("predict")
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_csv", type=Path)
    args = parser.parse_args(argv)

    if args.command == "predict":
        result = predict(args.output_dir, pd.read_csv(args.input_csv))
        result.to_csv(args.output_csv, index=False)
        print(f"{len(result)} predictions written to {args.output_csv}")
        return 0
    try:
        result = train(load_config(args.config), args.output_dir)
    except (ValueError, DataQualityError) as exc:
        print(f"Error: {exc}")
        return 1
    print(result["cv_table"].round(4).to_string())
    m = result["metrics"]
    print(f"chosen: {result['chosen']} | test ROC-AUC={m['roc_auc']} CI95={m['ci95']['roc_auc']} | accuracy={m['accuracy']}")
    print(f"report: {result['report']}\nmodel card: {result['model_card']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
