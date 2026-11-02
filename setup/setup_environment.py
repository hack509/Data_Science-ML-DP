"""Create the local virtual environment and install the dependencies.

THIS IS THE ONLY STEP THAT NEEDS THE INTERNET. It is never run implicitly:
without ``--install`` the script only creates the virtual environment and
prints the commands it would run.

Usage (from the repository root, with the system Python):
    python setup/setup_environment.py                 # create .venv, show next steps
    python setup/setup_environment.py --install       # create .venv and install (online)
    python setup/setup_environment.py --install --no-torch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"


def venv_python(venv_dir: Path = VENV_DIR) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def activation_command(venv_dir: Path = VENV_DIR) -> str:
    if os.name == "nt":
        return rf"{venv_dir.name}\Scripts\Activate.ps1   (PowerShell)   or   {venv_dir.name}\Scripts\activate.bat   (cmd)"
    return f"source {venv_dir.name}/bin/activate"


def install_commands(python: Path, with_torch: bool, with_dev: bool) -> list[list[str]]:
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "-r", "requirements.txt"],
    ]
    if with_torch:
        commands.append([str(python), "-m", "pip", "install", "-r", "requirements-torch.txt"])
    if with_dev:
        commands.append([str(python), "-m", "pip", "install", "-r", "requirements-dev.txt"])
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", action="store_true", help="download and install packages (needs Internet)")
    parser.add_argument("--no-torch", action="store_true", help="skip PyTorch (modules 6-9 will be limited)")
    parser.add_argument("--no-dev", action="store_true", help="skip pytest")
    parser.add_argument("--venv-dir", type=Path, default=VENV_DIR)
    args = parser.parse_args(argv)

    if sys.version_info < (3, 11):
        print("Python 3.11 or newer is required.")
        return 1

    python = venv_python(args.venv_dir)
    if python.exists():
        print(f"[OK] Virtual environment already exists: {args.venv_dir}")
    else:
        print(f"Creating virtual environment in {args.venv_dir} ...")
        venv.EnvBuilder(with_pip=True).create(args.venv_dir)
        print("[OK] Virtual environment created (offline operation).")

    commands = install_commands(python, with_torch=not args.no_torch, with_dev=not args.no_dev)
    if not args.install:
        print("\nNo package was installed (installation needs Internet and is explicit).")
        print("Run this once while connected:")
        print("    python setup/setup_environment.py --install")
        print("or manually:")
        for command in commands:
            print("    " + " ".join(Path(c).name if i == 0 else c for i, c in enumerate(command)))
    else:
        print("\nONLINE STEP: downloading packages from PyPI / download.pytorch.org")
        for command in commands:
            print("> " + " ".join(command))
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode != 0:
                print("[ERROR] Installation failed. See docs/troubleshooting.md")
                return result.returncode
        print("[OK] Packages installed. From now on, the lab works offline.")
        print("Generating datasets locally ...")
        subprocess.run([str(python), "scripts/generate_datasets.py"], cwd=ROOT, check=False)
        subprocess.run([str(python), "setup/check_environment.py"], cwd=ROOT, check=False)

    print(f"\nActivate the environment with:\n    {activation_command(args.venv_dir)}")
    print("Then check it with:\n    python setup/check_environment.py\n    python setup/verify_offline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
