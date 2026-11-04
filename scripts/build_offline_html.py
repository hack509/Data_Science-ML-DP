"""Build offline_ai_learning_lab.html: the whole repository in ONE self-contained file.

The file embeds the rendered lessons, corrections, exercises (solutions hidden
behind a click), projects, documentation, dataset previews, figures and source
code. It loads nothing from the network: a Content-Security-Policy forbids it.
Download the file, open it in any browser, and read the course offline.

Figures come from outputs/ and projects/*/outputs/: run the lessons or the test
suite first, otherwise the gallery is smaller (the build still works).

Usage (from the repository root):
    python scripts/build_offline_html.py
    python scripts/build_offline_html.py --output /path/to/file.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import posixpath
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import markdown_lite as md  # noqa: E402
from lab.catalog import list_exercises, list_modules, list_projects  # noqa: E402
from lab.data import DATASETS  # noqa: E402
from lab.paths import DATASETS_DIR, OUTPUTS_DIR, PROJECTS_DIR, ROOT  # noqa: E402

SITE_DIR = Path(__file__).resolve().parent / "offline_site"
DEFAULT_OUTPUT = ROOT / "offline_ai_learning_lab.html"
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".claude", ".vscode", "outputs", "exports", "models"}
TEXT_SUFFIXES = {".py", ".json", ".ini", ".txt", ".css", ".js", ".gitignore", ".gitattributes"}
CSV_PREVIEW_LINES = 40
MAX_TEXT_BYTES = 200_000
MAX_FIGURE_WIDTH = 820
LEVELS = [("beginner", "Débutant"), ("intermediate", "Intermédiaire"), ("advanced", "Avancé")]
KINDS = {"lesson": "leçon", "module": "module", "corrections": "corrigés", "exercise": "exercice", "project": "projet", "doc": "documentation"}
DOCS = [
    ("docs/learning_roadmap.md", "Parcours d'apprentissage"),
    ("docs/installation_guide.md", "Guide d'installation"),
    ("docs/offline_usage.md", "Utilisation hors ligne"),
    ("docs/glossary.md", "Glossaire"),
    ("docs/troubleshooting.md", "Dépannage"),
    ("docs/course_index.md", "Index du cours"),
    ("docs/lesson_template.md", "Gabarit des leçons"),
    ("docs/implementation_report.md", "Rapport d'implémentation"),
    ("exercises/README.md", "Mode d'emploi des exercices"),
    ("solutions/README.md", "Corrigé des exercices"),
    ("notebooks/README.md", "Notebooks"),
    ("progress/learning_log.md", "Journal d'apprentissage"),
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def repository_files() -> list[Path]:
    """Files of the repository, without environments and generated artefacts."""
    files = []
    for path in sorted(ROOT.rglob("*")):
        parts = path.relative_to(ROOT).parts
        if path.is_file() and not (set(parts) & EXCLUDED_DIRS) and path.resolve() != DEFAULT_OUTPUT:
            if not (parts[0] == "datasets" and parts[1:2] == ("processed",) and path.name != "README.md"):
                files.append(path)
    return files


def embedded_file(path: Path) -> dict | None:
    """Text content of a file to embed, or None for binary files."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            lines = [next(handle, "") for _ in range(CSV_PREVIEW_LINES)]
        total = sum(1 for _ in path.open(encoding="utf-8"))
        note = f"{CSV_PREVIEW_LINES} premières lignes sur {total}" if total > CSV_PREVIEW_LINES else ""
        return {"text": "".join(lines), "truncated": note}
    if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            return {"text": text[:MAX_TEXT_BYTES], "truncated": f"{MAX_TEXT_BYTES // 1000} premiers ko"}
        return {"text": text, "truncated": ""}
    return None


def image_data_uri(path: Path) -> str:
    """Shrink a PNG (width and palette) and return it as a data URI."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.width > MAX_FIGURE_WIDTH:
        image = image.resize((MAX_FIGURE_WIDTH, round(image.height * MAX_FIGURE_WIDTH / image.width)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.quantize(colors=128).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def caption(path: Path) -> str:
    return re.sub(r"[_-]+", " ", re.sub(r"^\d+_", "", path.stem)).capitalize()


def collect_figures() -> tuple[dict[str, str], list[dict]]:
    images, figures = {}, []
    sources = []
    if OUTPUTS_DIR.exists():
        for folder in sorted(p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.startswith("module")):
            sources += [(f"Module {int(folder.name[-2:])}", p) for p in sorted(folder.glob("*.png"))]
    for project in sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir()):
        sources += [(f"Projet {int(project.name[:2])}", p) for p in sorted((project / "outputs").glob("*.png"))]
    for group, path in sources:
        key = rel(path)
        images[key] = image_data_uri(path)
        figures.append({"key": key, "caption": caption(path), "group": group})
    return images, figures


def lesson_figures(text: str, module_number: int, known: set[str]) -> list[str]:
    """Figures named in a lesson: explicit output paths and save_figure(fig, 'name.png') calls."""
    names = re.findall(r"outputs/module\d\d/[\w.-]+\.png", text)
    names += [f"outputs/module{module_number:02d}/{n}" for n in re.findall(r'save_figure\(fig, "([\w.-]+\.png)"\)', text)]
    return [n for n in dict.fromkeys(names) if n in known]


def link_resolver(current: Path, pages: set[str], files: set[str]):
    """Turn relative Markdown links into in-page routes."""
    base = rel(current.parent)

    def resolve(href: str) -> str:
        if href.startswith(("http://", "https://", "mailto:")):
            return href
        target, _, anchor = href.partition("#")
        if not target:
            return f"#{anchor}"
        resolved = posixpath.normpath(posixpath.join(base, target))
        if f"{resolved}/README.md" in pages:
            resolved = f"{resolved}/README.md"
        if resolved in pages:
            return "#/p/" + resolved
        return "#/f/" + resolved if resolved in files else "#/tree"

    return resolve


def dataset_previews() -> tuple[str, dict[str, str]]:
    """HTML summary of each CSV dataset, plus sample images of the shapes."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    parts = ["<h2>Aperçu des données</h2><p>Premières lignes de chaque fichier CSV, générées localement.</p>"]
    for info in DATASETS.values():
        if not info.relative_path.endswith(".csv") or not info.path.exists():
            continue
        frame = pd.read_csv(info.path)
        head = frame.head(5).to_html(index=False, border=0, na_rep="", escape=True)
        types = ", ".join(f"{c} : {t}" for c, t in frame.dtypes.astype(str).items())
        parts.append(f"<h3><code>{html.escape(info.relative_path)}</code></h3><p>{html.escape(info.description)} — "
                     f"{len(frame)} lignes × {frame.shape[1]} colonnes.</p>{head}<p class=\"muted\">Types : {html.escape(types)}</p>")
    images = {}
    npz = DATASETS["shapes_npz"].path
    if npz.exists():
        data = np.load(npz)
        parts.append('<h3>Échantillon d\'images (agrandies 3 fois)</h3><div class="pixels">')
        for label, name in enumerate(data["class_names"]):
            for image in data["images"][data["labels"] == label][:6]:
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG", optimize=True)
                uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                parts.append(f'<img src="{uri}" width="84" height="84" alt="{html.escape(str(name))}">')
        parts.append("</div>")
    return "".join(parts), images


def build_data() -> dict:
    modules, exercises, projects = list_modules(), list_exercises(), list_projects()
    all_files = repository_files()
    files = {rel(p): content for p in all_files if (content := embedded_file(p)) is not None}

    page_sources: dict[str, dict] = {}  # repository path -> page description

    def add_page(path: Path, kind: str, title: str, **extra) -> None:
        if path.exists():
            page_sources[rel(path)] = {"source": path, "kind": kind, "title": title, **extra}

    add_page(ROOT / "README.md", "doc", "README du dépôt")
    add_page(DATASETS_DIR / "README.md", "doc", "Jeux de données")
    for path, title in DOCS:
        add_page(ROOT / path, "doc", title)
    module_entries = []
    for module in modules:
        add_page(module.readme, "module", module.title)
        lessons = []
        for lesson in module.lessons:
            item = {"category": "lessons", "id": lesson.id}
            add_page(lesson.path, "lesson", lesson.title, item=item, module_number=module.number)
            lessons.append({"path": rel(lesson.path), "title": lesson.title, "item": item})
        add_page(module.corrections, "corrections", f"Corrigés — {module.title}")
        module_entries.append({"id": module.id, "title": module.title, "readme": rel(module.readme),
                               "corrections": rel(module.corrections), "lessons": lessons})

    exercise_entries = []
    for exercise in exercises:
        item = {"category": "exercises", "id": exercise.id}
        extra = [
            {"label": "exercise.py — fichier à compléter", "path": rel(exercise.path / "exercise.py"), "open": True},
            {"label": "test_exercise.py — tests", "path": rel(exercise.path / "test_exercise.py")},
            {"label": "Afficher la solution", "path": rel(exercise.solution), "hidden": True},
        ]
        add_page(exercise.readme, "exercise", exercise.title, item=item, extra=extra)
        exercise_entries.append({"id": exercise.id, "level": exercise.level, "title": exercise.title, "path": rel(exercise.readme), "item": item})

    project_entries = []
    for project in projects:
        item = {"category": "projects", "id": project.id}
        code = sorted(p for p in project.path.rglob("*.py") if not (set(p.relative_to(ROOT).parts) & EXCLUDED_DIRS))
        extra = [{"label": rel(p).removeprefix(rel(project.path) + "/"), "path": rel(p), "hidden": False} for p in code if p.stat().st_size > 100]
        add_page(project.readme, "project", project.title, item=item, extra=extra)
        project_entries.append({"title": project.title, "path": rel(project.readme), "item": item})

    images, figures = collect_figures()
    known = set(images)
    preview_html, _ = dataset_previews()

    pages = {}
    page_set, file_set = set(page_sources), set(files)
    for path, source in page_sources.items():
        text = source["source"].read_text(encoding="utf-8")
        page = {"title": source["title"], "kind": source["kind"], "html": md.to_html(text, link_resolver=link_resolver(source["source"], page_set, file_set))}
        for key in ("item", "extra"):
            if key in source:
                page[key] = source[key]
        if "module_number" in source:
            page["figures"] = lesson_figures(text, source["module_number"], known)
        if source["kind"] == "project":
            prefix = path.rsplit("/", 1)[0] + "/outputs/"
            page["figures"] = [key for key in known if key.startswith(prefix)]
        pages[path] = page
    pages["datasets/README.md"]["preview"] = preview_html

    ordered = [(lesson["path"], lesson["title"]) for module in module_entries for lesson in module["lessons"]]
    for i, (path, _) in enumerate(ordered):
        if i:
            pages[path]["prev"] = {"path": ordered[i - 1][0], "title": ordered[i - 1][1]}
        if i + 1 < len(ordered):
            pages[path]["next"] = {"path": ordered[i + 1][0], "title": ordered[i + 1][1]}

    return {
        "meta": {"modules": len(modules), "lessons": len(ordered), "exercises": len(exercises), "projects": len(projects), "datasets": len(DATASETS)},
        "modules": module_entries,
        "exercises": exercise_entries,
        "projects": project_entries,
        "levels": [{"id": i, "title": t} for i, t in LEVELS],
        "docs": [{"path": p, "title": t} for p, t in DOCS if p in pages],
        "kinds": KINDS,
        "pages": pages,
        "files": files,
        "tree": [rel(p) for p in all_files],
        "images": images,
        "figures": figures,
    }


def render(data: dict) -> str:
    style = (SITE_DIR / "style.css").read_text(encoding="utf-8")
    script = (SITE_DIR / "app.js").read_text(encoding="utf-8")
    # "</" and "<!--" must not appear raw inside a <script> element.
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "\\u003c!--")
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; connect-src 'none'">
<title>Offline AI Learning Lab</title>
<style>{style}</style>
</head>
<body>
<header class="topbar">
<button id="menu-button" class="icon-button" type="button" aria-label="Ouvrir le menu">Menu</button>
<a class="brand" href="#/">Offline AI <span>Learning Lab</span></a>
<form id="search-form" class="search" role="search"><input id="search-input" type="search" placeholder="Rechercher dans le cours…" aria-label="Rechercher dans le cours"></form>
<button id="theme-button" class="icon-button" type="button">Sombre</button>
</header>
<div class="layout">
<nav id="sidebar" class="sidebar" aria-label="Sommaire"></nav>
<main id="view" class="content"></main>
</div>
<noscript><p>Ce fichier a besoin de JavaScript (exécuté localement, sans accès réseau). Sinon, lisez les fichiers Markdown du dépôt.</p></noscript>
<script id="lab-data" type="application/json">{payload}</script>
<script>{script}</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    data = build_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8", newline="\n")
    size = args.output.stat().st_size / 1_000_000
    print(f"[OK] {args.output} ({size:.1f} Mo): {len(data['pages'])} pages, {len(data['files'])} files, {len(data['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
