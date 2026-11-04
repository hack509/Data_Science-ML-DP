"""Generate docs/course_index.md, the Markdown table of contents of the whole lab.

The index is built from the file system (lab.catalog), so it is always in
sync with the lessons, exercises and projects actually present.

Usage (from the repository root):
    python scripts/generate_course_index.py
    python scripts/generate_course_index.py --check   # exit 1 if the index is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.catalog import list_exercises, list_modules, list_projects  # noqa: E402
from lab.paths import DOCS_DIR, ROOT  # noqa: E402

LEVEL_NAMES = {"beginner": "Débutant", "intermediate": "Intermédiaire", "advanced": "Avancé"}


def _link(path: Path) -> str:
    """Relative link from docs/ to a repository file."""
    return "../" + path.relative_to(ROOT).as_posix()


def build_index() -> str:
    modules, exercises, projects = list_modules(), list_exercises(), list_projects()
    n_lessons = sum(len(m.lessons) for m in modules)
    lines = [
        "# Index du cours",
        "",
        "Ce fichier est généré automatiquement par `python scripts/generate_course_index.py` : ne le modifiez pas à la main.",
        "",
        f"**{len(modules)} modules, {n_lessons} leçons, {len(exercises)} exercices, {len(projects)} projets.**",
        "",
        "Parcours conseillé : voir [learning_roadmap.md](learning_roadmap.md).",
        "",
        "## Modules et leçons",
        "",
    ]
    for module in modules:
        lines.append(f"### [{module.title}]({_link(module.readme)})")
        lines.append("")
        for lesson in module.lessons:
            lines.append(f"- [{lesson.title}]({_link(lesson.path)})")
        if module.corrections.exists():
            lines.append(f"- Corrigés des quiz et exercices guidés : [corrections.md]({_link(module.corrections)})")
        lines.append("")
    lines += ["## Exercices", ""]
    for level, title in LEVEL_NAMES.items():
        lines += [f"### {title}", "", "| Identifiant | Exercice | Solution |", "|---|---|---|"]
        for exercise in [e for e in exercises if e.level == level]:
            lines.append(f"| {exercise.id} | [{exercise.title}]({_link(exercise.readme)}) | [solution.py]({_link(exercise.solution)}) |")
        lines.append("")
    lines += ["## Projets", ""]
    for project in projects:
        lines.append(f"{project.number}. [{project.title}]({_link(project.readme)})")
    lines += ["", "## Documentation", ""]
    for name, title in [
        ("installation_guide.md", "Guide d'installation"),
        ("offline_usage.md", "Utilisation hors ligne"),
        ("learning_roadmap.md", "Parcours d'apprentissage"),
        ("glossary.md", "Glossaire anglais-français"),
        ("troubleshooting.md", "Dépannage"),
        ("lesson_template.md", "Gabarit des leçons"),
    ]:
        lines.append(f"- [{title}]({name})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DOCS_DIR / "course_index.md")
    parser.add_argument("--check", action="store_true", help="only check that the index is up to date")
    args = parser.parse_args(argv)
    content = build_index()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != content:
            print(f"{args.output} is out of date. Run: python scripts/generate_course_index.py")
            return 1
        print("course index is up to date")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
