"""Execute the Python code blocks of a Markdown lesson.

All fenced blocks tagged exactly ``python`` are concatenated in order and run
in a fresh Python process from the repository root, like a learner who
copies the examples one after another into a single script. Blocks tagged
``python no-run`` are illustrative and skipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lab.markdown_lite import extract_code_blocks
from lab.paths import ROOT


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    n_blocks: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def lesson_source(path: Path) -> tuple[str, int]:
    blocks = extract_code_blocks(Path(path).read_text(encoding="utf-8"))
    parts = [f"# --- block {i} ---\n{block.code}\n" for i, block in enumerate(blocks, start=1)]
    return "\n".join(parts), len(blocks)


def run_lesson(path: Path, timeout: int = 300) -> RunResult:
    source, n_blocks = lesson_source(path)
    if n_blocks == 0:
        return RunResult(0, "", "", 0)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(source)
        script = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, script], cwd=ROOT, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    finally:
        os.unlink(script)
    return RunResult(completed.returncode, completed.stdout, completed.stderr, n_blocks)


_SENTINEL = "@@LAB_BLOCK_END@@"
_OUTPUT_FENCE = "```text output"
MAX_OUTPUT_LINES = 45


def capture_block_outputs(path: Path, timeout: int = 300) -> list[str]:
    """Run the lesson and return the standard output of each executable block."""
    blocks = extract_code_blocks(Path(path).read_text(encoding="utf-8"))
    parts = [f"{block.code}\nprint({_SENTINEL!r}, flush=True)\n" for block in blocks]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(parts))
        script = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, script], cwd=ROOT, env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    finally:
        os.unlink(script)
    if completed.returncode != 0:
        raise RuntimeError(f"Lesson code failed:\n{completed.stderr[-3000:]}")
    outputs = completed.stdout.split(_SENTINEL + "\n")
    return [o.rstrip() for o in outputs[: len(blocks)]]


def update_expected_outputs(path: Path) -> int:
    """Fill every ```text output block with the real output of the preceding code block.

    Returns the number of updated output blocks.
    """
    path = Path(path)
    outputs = capture_block_outputs(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    result: list[str] = []
    executed = 0
    updated = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") and not stripped.startswith(_OUTPUT_FENCE) and stripped != "```":
            # Any other fenced block: copy it unchanged, counting executable ones.
            if stripped == "```python":
                executed += 1
            result.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                result.append(lines[i])
                i += 1
            if i < len(lines):
                result.append(lines[i])
            i += 1
            continue
        if stripped == _OUTPUT_FENCE:
            result.append(line)
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                i += 1  # drop the old content
            output = outputs[executed - 1] if executed else ""
            out_lines = output.split("\n") if output else ["(aucune sortie affichée)"]
            if len(out_lines) > MAX_OUTPUT_LINES:
                out_lines = out_lines[:MAX_OUTPUT_LINES] + ["... (sortie tronquée)"]
            result.extend(out_lines)
            result.append("```")
            updated += 1
            i += 1
            continue
        result.append(line)
        i += 1
    path.write_text("\n".join(result), encoding="utf-8", newline="\n")
    return updated


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run every executable code block of a lesson.")
    parser.add_argument("lesson", type=Path, help="path to a lesson Markdown file")
    parser.add_argument("--show-source", action="store_true")
    parser.add_argument(
        "--update-outputs", action="store_true",
        help="maintainer mode: rewrite the ```text output blocks with the real outputs",
    )
    args = parser.parse_args(argv)
    if args.update_outputs:
        print(f"{update_expected_outputs(args.lesson)} output block(s) updated in {args.lesson}")
        return 0
    if args.show_source:
        print(lesson_source(args.lesson)[0])
        return 0
    result = run_lesson(args.lesson)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(f"\n[{'OK' if result.ok else 'FAILED'}] {result.n_blocks} code block(s) executed", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
