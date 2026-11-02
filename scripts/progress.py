"""Display and update your learning progress (stored in progress/progress.json).

Nothing is marked as completed unless you run one of these commands yourself.

Examples (from the repository root):
    python scripts/progress.py show
    python scripts/progress.py list                       # all valid identifiers
    python scripts/progress.py lesson 01_python_for_data_science/01_bases_python
    python scripts/progress.py lesson 01_python_for_data_science/01_bases_python --undo
    python scripts/progress.py exercise B01
    python scripts/progress.py project 01_student_performance_analysis
    python scripts/progress.py quiz 01_python_for_data_science 8 10
    python scripts/progress.py module 02_mathematics_and_statistics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.catalog import list_exercises, list_modules, list_projects  # noqa: E402
from lab.paths import PROGRESS_FILE  # noqa: E402
from lab.progress import ProgressTracker  # noqa: E402


def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {ratio:5.0%}"


def show(tracker: ProgressTracker) -> None:
    modules = list_modules()
    exercises = list_exercises()
    projects = list_projects()
    print(f"Progress file : {tracker.path}")
    print(f"Current module: {tracker.data['current_module']}\n")
    print("Lessons by module")
    for module in modules:
        ids = [lesson.id for lesson in module.lessons]
        ratio = tracker.completion_ratio("lessons", ids)
        quiz = tracker.best_quiz(module.id)
        quiz_text = f"  best quiz {quiz['score']:g}/{quiz['max_score']:g}" if quiz else ""
        print(f"  {module.id:<36} {_bar(ratio)}{quiz_text}")
    print()
    for level in ("beginner", "intermediate", "advanced"):
        ids = [e.id for e in exercises if e.level == level]
        print(f"  exercises {level:<26} {_bar(tracker.completion_ratio('exercises', ids))}")
    ids = [p.id for p in projects]
    print(f"  {'projects':<36} {_bar(tracker.completion_ratio('projects', ids))}")


def list_ids() -> None:
    print("Lessons:")
    for module in list_modules():
        for lesson in module.lessons:
            print(f"  {lesson.id}")
    print("Exercises:")
    for exercise in list_exercises():
        print(f"  {exercise.id:<5} {exercise.level:<13} {exercise.title}")
    print("Projects:")
    for project in list_projects():
        print(f"  {project.id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=PROGRESS_FILE, help="progress file to use")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="display a summary")
    sub.add_parser("list", help="list valid identifiers")
    for name in ("lesson", "exercise", "project"):
        p = sub.add_parser(name, help=f"mark a {name} as completed")
        p.add_argument("item_id")
        p.add_argument("--undo", action="store_true", help="remove the completion mark")
        p.add_argument("--note", default="", help="optional personal note")
    q = sub.add_parser("quiz", help="record a quiz score")
    q.add_argument("module_id")
    q.add_argument("score", type=float)
    q.add_argument("max_score", type=float)
    m = sub.add_parser("module", help="set the current module")
    m.add_argument("module_id")
    args = parser.parse_args(argv)

    tracker = ProgressTracker(args.file)
    module_ids = {module.id for module in list_modules()}
    valid = {
        "lessons": {lesson.id for module in list_modules() for lesson in module.lessons},
        "exercises": {e.id for e in list_exercises()},
        "projects": {p.id for p in list_projects()},
    }

    if args.command == "show":
        show(tracker)
        return 0
    if args.command == "list":
        list_ids()
        return 0
    if args.command in ("lesson", "exercise", "project"):
        category = args.command + "s"
        item_id = args.item_id.upper() if category == "exercises" else args.item_id
        if item_id not in valid[category]:
            print(f"Unknown {args.command} id: {args.item_id!r}. Run 'python scripts/progress.py list'.")
            return 1
        tracker.set_completed(category, item_id, completed=not args.undo, note=args.note)
        tracker.save()
        print(f"{'Removed' if args.undo else 'Marked as completed'}: {item_id}")
        return 0
    if args.command == "quiz":
        if args.module_id not in module_ids:
            print(f"Unknown module id: {args.module_id!r}")
            return 1
        tracker.record_quiz(args.module_id, args.score, args.max_score)
        tracker.save()
        print(f"Quiz recorded for {args.module_id}: {args.score:g}/{args.max_score:g}")
        return 0
    if args.command == "module":
        if args.module_id not in module_ids:
            print(f"Unknown module id: {args.module_id!r}")
            return 1
        tracker.set_current_module(args.module_id)
        tracker.save()
        print(f"Current module: {args.module_id}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
