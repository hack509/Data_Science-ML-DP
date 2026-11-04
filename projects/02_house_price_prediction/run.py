"""Project 2 - train, select and evaluate house price models.

    python projects/02_house_price_prediction/run.py [--quick] [--output-dir DIR]
    python projects/02_house_price_prediction/run.py predict new_houses.csv predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from house_prices.modeling import predict, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--quick", action="store_true", help="small grids, for tests")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("predict", help="predict prices for a CSV file")
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_csv", type=Path)
    args = parser.parse_args(argv)

    if args.command == "predict":
        result, warnings = predict(args.output_dir, pd.read_csv(args.input_csv))
        result.to_csv(args.output_csv, index=False)
        print(f"{len(result)} predictions written to {args.output_csv}")
        for message in warnings:
            print("WARNING", message)
        return 0

    result = run(args.output_dir, quick=args.quick)
    print(result["cv_table"][["cv_mae", "cv_mae_std"]].round(2).to_string())
    m = result["metrics"]
    print(f"selected: {result['selected']} | test MAE={m['test_mae']} k EUR (95% CI {m['test_mae_ci95']}) "
          f"| R2={m['test_r2']} | baseline MAE={m['baseline_mae']}")
    print(f"saved: {result['paths']['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
