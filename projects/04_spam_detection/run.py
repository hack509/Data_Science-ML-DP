"""Project 4 - spam detection.

    python projects/04_spam_detection/run.py                       # train and evaluate
    python projects/04_spam_detection/run.py classify "Vous avez gagné 500 EUR, cliquez ici"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent.parent))

from spam_filter.detector import classify, explain, run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    sub = parser.add_subparsers(dest="command")
    c = sub.add_parser("classify", help="classify one or more messages with the saved model")
    c.add_argument("messages", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "classify":
        for row in classify(args.messages, args.output_dir / "spam_model.joblib"):
            print(json.dumps(row, ensure_ascii=False))
        return 0

    result = run(args.output_dir)
    summary = {k: v for k, v in result.items() if k not in ("errors", "model")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{len(result['errors'])} test errors saved to {args.output_dir / 'errors.csv'}")
    example = "Coucou, regarde ces photos de toi http://promo-123.example"
    print("explanation for:", example, "->", [(t, round(v, 2)) for t, v in explain(result["model"], example)]
          if result["selected_model"] != "naive_bayes_words" else "(not available for naive Bayes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
