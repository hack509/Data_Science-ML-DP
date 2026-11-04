"""Verify that the lab really works without network access.

The script blocks every outgoing connection (except localhost) inside the
Python process, then runs a series of representative tasks. Any attempt to
reach the Internet makes the corresponding check fail. It also scans the
repository source code for functions that download data.

Usage (from the repository root):
    python setup/verify_offline.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from lab.offline import block_network  # noqa: E402

# Patterns that indicate code which would download something at run time.
NETWORK_PATTERNS = [
    r"urlopen\(",
    r"urlretrieve\(",
    r"requests\.(get|post|put)\(",
    r"httpx\.(get|post)\(",
    r"fetch_openml\(",
    r"sns\.load_dataset\(",
    r"download\s*=\s*True",
    r"hf_hub_download\(",
    r"torch\.hub\.load\(",
]
# tests/ is excluded: the tests only call the local web interface on 127.0.0.1.
SCAN_EXCLUDE = {".venv", "__pycache__", ".pytest_cache", "outputs", "exports", "tests"}


def scan_sources() -> list[str]:
    """Return 'file:line: code' entries matching a download pattern."""
    from lab.markdown_lite import extract_code_blocks

    pattern = re.compile("|".join(NETWORK_PATTERNS))
    findings = []
    this_file = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if any(part in SCAN_EXCLUDE for part in path.parts) or not path.is_file() or path.resolve() == this_file:
            continue
        if path.suffix == ".py":
            sources = [path.read_text(encoding="utf-8")]
        elif path.suffix == ".md":
            sources = [b.code for b in extract_code_blocks(path.read_text(encoding="utf-8"))]
        else:
            continue
        for source in sources:
            for number, line in enumerate(source.splitlines(), start=1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    findings.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    return findings


def check_imports() -> str:
    import matplotlib.pyplot  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import PIL  # noqa: F401
    import scipy.stats  # noqa: F401
    import seaborn  # noqa: F401
    import sklearn  # noqa: F401
    import sympy  # noqa: F401

    return "numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, sympy, pillow"


def check_datasets() -> str:
    from lab.data import DATASETS, load_csv, missing_datasets

    missing = missing_datasets()
    if missing:
        raise RuntimeError(f"missing datasets: {missing}. Run: python scripts/generate_datasets.py")
    rows = sum(len(load_csv(name)) for name, info in DATASETS.items() if info.relative_path.endswith(".csv"))
    return f"{len(DATASETS)} datasets present, {rows} CSV rows loaded"


def check_dataset_generation() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_datasets

    with tempfile.TemporaryDirectory() as tmp:
        created = generate_datasets.generate_all(Path(tmp), verbose=False)
    return f"{len(created)} datasets regenerated in a temporary folder"


def check_sklearn_training() -> str:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from lab.data import load_csv

    df = load_csv("binary")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns="target"), df["target"], test_size=0.25, random_state=0
    )
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    return f"LogisticRegression accuracy on synthetic test set: {model.score(X_test, y_test):.3f}"


def check_torch_training() -> str:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PyTorch not available ({exc}). Install requirements-torch.txt") from exc
    from torch import nn

    torch.manual_seed(0)
    X = torch.randn(256, 2)
    y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    accuracy = ((model(X) > 0).float() == y).float().mean().item()
    return f"tiny MLP trained on CPU, training accuracy {accuracy:.2f}"


def check_plot() -> str:
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        target = Path(tmp) / "plot.png"
        fig.savefig(target)
        plt.close(fig)
        size = target.stat().st_size
    return f"matplotlib figure saved ({size} bytes)"


def check_pdf_fonts() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_lessons_to_pdf

    fonts = export_lessons_to_pdf.find_fonts()
    return "DejaVu fonts found locally: " + ", ".join(p.name for p in fonts.values())


def check_course_files() -> str:
    from lab.catalog import list_exercises, list_modules, list_projects

    modules = list_modules()
    lessons = sum(len(m.lessons) for m in modules)
    return f"{len(modules)} modules, {lessons} lessons, {len(list_exercises())} exercises, {len(list_projects())} projects"


CHECKS = [
    ("Core imports", check_imports),
    ("Course files", check_course_files),
    ("Datasets on disk", check_datasets),
    ("Dataset regeneration", check_dataset_generation),
    ("Matplotlib rendering", check_plot),
    ("scikit-learn training", check_sklearn_training),
    ("PyTorch training (CPU)", check_torch_training),
    ("PDF fonts", check_pdf_fonts),
]


def main() -> int:
    print("Offline verification - all non-local network connections are BLOCKED\n")
    failures = 0
    with block_network() as attempts:
        for name, check in CHECKS:
            start = time.perf_counter()
            try:
                detail = check()
                status = "PASS"
            except Exception as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"
                if os.environ.get("LAB_DEBUG"):
                    traceback.print_exc()
                status = "FAIL"
                failures += 1
            print(f"[{status}] {name:<26} ({time.perf_counter() - start:4.1f}s) {detail}")
    if attempts:
        print(f"\nBlocked network attempts: {sorted(set(attempts))}")
        failures += 1

    findings = scan_sources()
    if findings:
        print("\n[WARN] Source lines that may download data at run time:")
        for line in findings:
            print("   " + line)
        failures += 1
    else:
        print("[PASS] Source scan                 no download call found in .py files or executable lesson code")

    print("\nResult: " + ("OFFLINE READY" if failures == 0 else f"{failures} problem(s) found"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
