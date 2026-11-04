"""Check that the Python environment contains everything the lab needs.

This script never accesses the Internet and never installs anything. When a
package is missing it prints the exact command to run during the (online)
installation phase.

Usage (from the repository root):
    python setup/check_environment.py
    python setup/check_environment.py --quick   # skip the import test
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_PYTHON = (3, 11)


@dataclass(frozen=True)
class Requirement:
    distribution: str  # name used by pip
    module: str  # name used by import
    minimum: str
    group: str
    purpose: str


REQUIREMENTS = [
    Requirement("numpy", "numpy", "2.0", "core", "tableaux numeriques (modules 1 a 10)"),
    Requirement("pandas", "pandas", "2.2", "core", "tableaux de donnees, CSV/JSON"),
    Requirement("matplotlib", "matplotlib", "3.9", "core", "graphiques"),
    Requirement("seaborn", "seaborn", "0.13.2", "core", "graphiques statistiques (module 4)"),
    Requirement("scipy", "scipy", "1.13", "core", "statistiques et tests (module 2)"),
    Requirement("scikit-learn", "sklearn", "1.7", "core", "Machine Learning (modules 5, 8, 10)"),
    Requirement("sympy", "sympy", "1.13", "core", "mathematiques symboliques (module 2)"),
    Requirement("pillow", "PIL", "10.4", "core", "images (module 9)"),
    Requirement("torch", "torch", "2.5", "deep-learning", "Deep Learning (modules 6 a 9)"),
    Requirement("jupyterlab", "jupyterlab", "4.2", "notebooks", "notebooks locaux"),
    Requirement("ipykernel", "ipykernel", "6.29", "notebooks", "noyau Python pour Jupyter"),
    Requirement("nbformat", "nbformat", "5.10", "notebooks", "generation des notebooks"),
    Requirement("reportlab", "reportlab", "4.2", "pdf", "export PDF des lecons"),
    Requirement("pytest", "pytest", "8.0", "dev", "tests automatiques"),
]

INSTALL_HINTS = {
    "core": "python -m pip install -r requirements.txt",
    "notebooks": "python -m pip install -r requirements.txt",
    "pdf": "python -m pip install -r requirements.txt",
    "deep-learning": "python -m pip install -r requirements-torch.txt",
    "dev": "python -m pip install -r requirements-dev.txt",
}


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.replace("+", ".").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_requirement(req: Requirement, try_import: bool = True) -> tuple[str, str, str]:
    """Return (status, installed_version, detail). Status: OK, OLD, MISSING, BROKEN."""
    try:
        version = importlib.metadata.version(req.distribution)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING", "-", f"install with: {INSTALL_HINTS[req.group]}"
    if _version_tuple(version) < _version_tuple(req.minimum):
        return "OLD", version, f"version >= {req.minimum} required"
    if try_import:
        try:
            importlib.import_module(req.module)
        except Exception as exc:  # noqa: BLE001 - we want to report any import failure
            message = str(exc).splitlines()[0][:110]
            return "BROKEN", version, f"import failed: {message}"
    return "OK", version, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="only read package metadata, do not import")
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Offline AI Learning Lab - environment check (no network access)")
    print("=" * 78)
    in_venv = sys.prefix != sys.base_prefix
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    print(f"System      : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Virtual env : {'yes' if in_venv else 'NO - activate .venv first (see docs/installation_guide.md)'}")
    print(f"CPU cores   : {os.cpu_count()}")
    python_ok = sys.version_info[:2] >= MIN_PYTHON
    if not python_ok:
        print(f"[ERROR] Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} is required.")
    print()
    print(f"{'status':<8} {'package':<14} {'version':<14} {'group':<14} detail / purpose")
    print("-" * 78)

    problems: dict[str, list[str]] = {}
    for req in REQUIREMENTS:
        status, version, detail = check_requirement(req, try_import=not args.quick)
        print(f"{status:<8} {req.distribution:<14} {version:<14} {req.group:<14} {detail or req.purpose}")
        if status != "OK":
            problems.setdefault(req.group, []).append(req.distribution)

    print()
    if "torch" not in problems.get("deep-learning", []) and not args.quick:
        try:
            import torch

            gpu = torch.cuda.is_available()
            print(f"PyTorch device: {'GPU (CUDA) available' if gpu else 'CPU only (expected, every lesson runs on CPU)'}")
            print(f"PyTorch threads: {torch.get_num_threads()}")
        except Exception:  # noqa: BLE001
            pass

    import subprocess

    probe = subprocess.run([sys.executable, "-c", "import lab"], cwd=Path.home(), capture_output=True)
    if probe.returncode == 0:
        print("Repository  : `import lab` works from any folder")
    else:
        print("[WARN] `import lab` only works from the repository root.")
        print("       Fix (offline): python setup/setup_environment.py")

    from lab.data import missing_datasets

    missing = missing_datasets()
    if missing:
        print(f"[WARN] {len(missing)} dataset(s) missing: {', '.join(missing)}")
        print("       Generate them offline with: python scripts/generate_datasets.py")
    else:
        print("Datasets      : all registered datasets are present")

    if problems:
        print("\nMissing or broken packages. During the ONLINE installation phase run:")
        for group, names in problems.items():
            print(f"  - {', '.join(names)}: {INSTALL_HINTS[group]}")
        if "torch" in problems.get("deep-learning", []):
            print("  - If torch is installed but BROKEN on Windows, install the Microsoft Visual C++")
            print("    Redistributable x64 (see docs/troubleshooting.md).")
        core_missing = any(g in problems for g in ("core",))
        print("\nResult: INCOMPLETE" + (" (core packages missing)" if core_missing else " (optional parts missing)"))
        return 1 if core_missing or not python_ok else 0
    print("\nResult: environment ready. The lab can now be used offline.")
    return 0 if python_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
