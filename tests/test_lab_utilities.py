"""Markdown parser, catalog, progress tracking, offline guard, exercise loader, lesson runner."""

import json
import socket
import subprocess
import sys

import pytest

from lab import markdown_lite as md
from lab.catalog import get_exercise, get_lesson, list_exercises, list_lessons, list_modules, list_projects
from lab.exercise_loader import TARGET_ENV, load_target, target_path
from lab.lesson_runner import capture_block_outputs, run_lesson, update_expected_outputs
from lab.offline import NetworkAccessError, block_network
from lab.paths import EXERCISES_DIR, ROOT, SOLUTIONS_DIR
from lab.progress import ProgressTracker, empty_progress

SAMPLE = """# Titre

Un **gras**, un *italique*, du `code` et un [lien](autre.md).

- point 1
  - sous-point
- point 2

1. premier
2. second

| a | b |
|---|---|
| 1 | 2 |

> citation

---

```python
print("run")
```

```python no-run
print("skip")
```

```text output
ancien
```
"""


def test_markdown_blocks():
    kinds = [type(b).__name__ for b in md.parse(SAMPLE)]
    assert kinds == ["Heading", "Paragraph", "ListBlock", "ListBlock", "Table", "Quote", "Rule", "CodeBlock", "CodeBlock", "CodeBlock"]
    bullets = md.parse(SAMPLE)[2]
    assert bullets.items[0].children[0].text == "sous-point"


def test_markdown_html_escapes_and_links():
    rendered = md.to_html("Texte <script> & `a < b`\n\n[x](y.md)", link_resolver=lambda href: "/view/" + href)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered and "a &lt; b" in rendered
    assert 'href="/view/y.md"' in rendered


def test_snake_case_is_not_italic():
    assert "<em>" not in md.to_html("la variable study_hours_week et 2 * 3 * 4")


def test_only_plain_python_blocks_are_executable():
    assert [b.code for b in md.extract_code_blocks(SAMPLE)] == ['print("run")']


def test_slugify_handles_accents():
    assert md.slugify("Leçon 1.1 — Été") == "lecon-1-1-ete"


def test_lesson_runner_and_output_update(tmp_path):
    lesson = tmp_path / "lesson.md"
    lesson.write_text('```python\nprint(40 + 2)\n```\n\n```text output\nold\n```\n', encoding="utf-8")
    assert capture_block_outputs(lesson) == ["42"]
    assert update_expected_outputs(lesson) == 1
    assert "42" in lesson.read_text(encoding="utf-8") and "old" not in lesson.read_text(encoding="utf-8")
    failing = tmp_path / "bad.md"
    failing.write_text("```python\nraise ValueError('boom')\n```\n", encoding="utf-8")
    assert not run_lesson(failing).ok


def test_catalog_counts():
    modules = list_modules()
    assert len(modules) == 10
    assert len(list_lessons()) == 39
    assert len(list_exercises()) == 21
    assert len(list_projects()) == 6
    assert get_lesson("01_python_for_data_science/01_bases_python").title.startswith("Leçon 1.1")
    assert get_exercise("b01").level == "beginner"


def test_progress_is_explicit_and_persistent(tmp_path):
    path = tmp_path / "sub" / "progress.json"
    tracker = ProgressTracker(path)
    assert tracker.data["lessons"] == {} and not path.exists(), "nothing is written without an explicit action"
    tracker.set_completed("lessons", "m/l1")
    tracker.record_quiz("m", 4, 6)
    tracker.record_quiz("m", 5, 6)
    tracker.set_current_module("m")
    tracker.save()
    again = ProgressTracker(path)
    assert again.is_completed("lessons", "m/l1") and not again.is_completed("lessons", "m/l2")
    assert again.best_quiz("m")["score"] == 5
    assert again.completion_ratio("lessons", ["m/l1", "m/l2"]) == 0.5
    again.set_completed("lessons", "m/l1", completed=False)
    assert not again.is_completed("lessons", "m/l1")


def test_progress_validation(tmp_path):
    tracker = ProgressTracker(tmp_path / "p.json")
    with pytest.raises(ValueError):
        tracker.set_completed("unknown", "x")
    with pytest.raises(ValueError):
        tracker.record_quiz("m", 8, 6)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError):
        ProgressTracker(bad)


def test_repository_progress_file_starts_empty():
    data = json.loads((ROOT / "progress" / "progress.json").read_text(encoding="utf-8"))
    assert set(empty_progress()) <= set(data)


def test_progress_cli(tmp_path):
    path = tmp_path / "p.json"
    script = str(ROOT / "scripts" / "progress.py")

    def run(*args):
        return subprocess.run([sys.executable, script, "--file", str(path), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert run("exercise", "B01").returncode == 0
    assert run("exercise", "ZZZ").returncode == 1
    assert run("quiz", "01_python_for_data_science", "5", "6").returncode == 0
    shown = run("show")
    assert shown.returncode == 0 and "best quiz 5/6" in shown.stdout
    assert json.loads(path.read_text(encoding="utf-8"))["exercises"]["B01"]["completed"] is True


def test_block_network_blocks_and_restores():
    original = socket.create_connection
    with block_network() as attempts:
        with pytest.raises(NetworkAccessError):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(NetworkAccessError):
            socket.getaddrinfo("example.com", 80)
    assert "example.com" in attempts
    assert socket.create_connection is original


def test_block_network_allows_localhost():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    with block_network():
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    server.close()


def test_exercise_loader_targets(monkeypatch):
    test_file = EXERCISES_DIR / "beginner" / "B01_student_averages" / "test_exercise.py"
    monkeypatch.delenv(TARGET_ENV, raising=False)
    assert target_path(test_file).name == "exercise.py"
    monkeypatch.setenv(TARGET_ENV, "solution")
    assert target_path(test_file) == SOLUTIONS_DIR / "beginner" / "B01_student_averages" / "solution.py"
    assert load_target(test_file).average([1, 3]) == 2
    monkeypatch.setenv(TARGET_ENV, "nonsense")
    with pytest.raises(ValueError):
        target_path(test_file)
