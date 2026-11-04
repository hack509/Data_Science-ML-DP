"""Local learning interface: browse lessons, exercises and projects, search, track progress.

Pure standard library (http.server). The server only listens on 127.0.0.1,
reads Markdown files from this repository and writes progress only when you
click a button. No external resource is loaded (no CDN, no fonts, no API).

Usage (from the repository root):
    python scripts/run_learning_environment.py              # http://127.0.0.1:8765
    python scripts/run_learning_environment.py --port 9000 --open
    python scripts/run_learning_environment.py --jupyter    # also start JupyterLab locally
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import subprocess
import sys
import threading
import unicodedata
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import markdown_lite as md  # noqa: E402
from lab.catalog import list_exercises, list_modules, list_projects  # noqa: E402
from lab.paths import PROGRESS_FILE, ROOT  # noqa: E402
from lab.progress import ProgressTracker  # noqa: E402

TEXT_SUFFIXES = {".md", ".py", ".json", ".csv", ".txt", ".ini"}
LEVEL_NAMES = {"beginner": "Débutant", "intermediate": "Intermédiaire", "advanced": "Avancé"}

STYLE = """
:root { --bg:#fcfcfb; --panel:#f3f3f0; --text:#2b2b29; --muted:#6b6b66; --accent:#2a78d6; --ok:#008300; --border:#dcdcd6; --code:#f1f1ee; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --panel:#242422; --text:#f0efec; --muted:#a9a8a0; --accent:#6da7ec; --ok:#3fb950; --border:#3a3a37; --code:#2b2b29; } }
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); line-height:1.55; }
header { padding:10px 20px; border-bottom:1px solid var(--border); display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header a { color:var(--text); text-decoration:none; font-weight:600; }
header form { margin-left:auto; }
input[type=search] { padding:6px 10px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text); min-width:220px; }
.layout { display:flex; }
nav { width:290px; flex-shrink:0; padding:16px; border-right:1px solid var(--border); background:var(--panel); min-height:calc(100vh - 50px); font-size:14px; }
nav a { color:var(--text); text-decoration:none; display:block; padding:3px 0; }
nav a:hover, main a { color:var(--accent); }
main { padding:20px 32px; max-width:900px; width:100%; }
pre { background:var(--code); padding:12px; border-radius:6px; overflow-x:auto; font-size:13px; }
code { background:var(--code); padding:1px 4px; border-radius:4px; font-size:0.92em; }
pre code { padding:0; background:none; }
table { border-collapse:collapse; margin:10px 0; font-size:14px; }
th, td { border:1px solid var(--border); padding:4px 8px; text-align:left; vertical-align:top; }
blockquote { border-left:4px solid var(--accent); margin:10px 0; padding:4px 12px; background:var(--panel); }
.done { color:var(--ok); font-weight:600; }
.badge { font-size:12px; padding:1px 6px; border-radius:10px; border:1px solid var(--border); color:var(--muted); }
.bar { height:8px; background:var(--panel); border:1px solid var(--border); border-radius:4px; width:220px; display:inline-block; vertical-align:middle; }
.bar span { display:block; height:100%; background:var(--accent); border-radius:4px; }
button { padding:6px 12px; border-radius:6px; border:1px solid var(--accent); background:var(--accent); color:white; cursor:pointer; }
button.secondary { background:transparent; color:var(--accent); }
.actions { margin:12px 0; padding:10px; border:1px solid var(--border); border-radius:8px; background:var(--panel); }
.muted { color:var(--muted); }
mark { background:#eda10055; color:inherit; }
@media (max-width: 800px) { .layout { flex-direction:column; } nav { width:100%; min-height:0; } main { padding:16px; } }
"""


def fold(text: str) -> str:
    """Lower case without accents, for accent-insensitive search."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def safe_path(relative: str) -> Path | None:
    """Resolve a repository-relative path, refusing anything outside the repository."""
    if not relative:
        return None
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return None
    if any(part in {".venv", ".git"} for part in candidate.relative_to(ROOT).parts):
        return None
    return candidate if candidate.exists() else None


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def view_url(path: Path, anchor: str = "") -> str:
    return "/view?" + urlencode({"path": rel(path)}) + (f"#{anchor}" if anchor else "")


class Catalog:
    """Cached view of the course structure; refreshed on every request (cheap)."""

    def __init__(self):
        self.modules = list_modules()
        self.exercises = list_exercises()
        self.projects = list_projects()
        self.items = {}  # repository path -> (category, item_id, label)
        for module in self.modules:
            for lesson in module.lessons:
                self.items[rel(lesson.path)] = ("lessons", lesson.id, lesson.title)
        for exercise in self.exercises:
            self.items[rel(exercise.readme)] = ("exercises", exercise.id, exercise.title)
        for project in self.projects:
            self.items[rel(project.readme)] = ("projects", project.id, project.title)


class LabHandler(BaseHTTPRequestHandler):
    server_version = "OfflineAILab/1.0"
    progress_file: Path = PROGRESS_FILE

    # -- helpers --------------------------------------------------------------
    def log_message(self, format, *args):  # quieter console
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def tracker(self) -> ProgressTracker:
        return ProgressTracker(self.progress_file)

    def page(self, title: str, content: str, catalog: Catalog) -> str:
        tracker = self.tracker()
        nav = [f'<p><a href="/"><strong>Accueil</strong></a><a href="/progress">Progression</a>'
               f'<a href="/exercises">Exercices</a><a href="/projects">Projets</a></p><hr>']
        for module in catalog.modules:
            done = sum(tracker.is_completed("lessons", lesson.id) for lesson in module.lessons)
            current = " ◂" if tracker.data["current_module"] == module.id else ""
            nav.append(f'<a href="{view_url(module.readme)}">{html.escape(module.title)}</a>'
                       f'<span class="muted">&nbsp;&nbsp;{done}/{len(module.lessons)} leçons{current}</span>')
        return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} · Offline AI Lab</title>
<style>{STYLE}</style></head><body>
<header><a href="/">Offline AI Learning Lab</a><span class="badge">local · hors ligne</span>
<form action="/search"><input type="search" name="q" placeholder="Rechercher dans le cours..." aria-label="Rechercher"></form></header>
<div class="layout"><nav>{''.join(nav)}</nav><main>{content}</main></div></body></html>"""

    def link_resolver(self, current: Path):
        def resolve(href: str) -> str:
            if href.startswith(("http://", "https://", "mailto:")):
                return href
            target, _, anchor = href.partition("#")
            if not target:
                return f"#{anchor}"
            resolved = posixpath.normpath(posixpath.join(rel(current.parent) if current.parent != ROOT else "", target))
            path = safe_path(resolved)
            if path is None:
                return "#"
            if path.is_dir() and (path / "README.md").exists():
                path = path / "README.md"
            return view_url(path, anchor)
        return resolve

    # -- GET ------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        catalog = Catalog()
        routes = {"/": self.home, "/view": self.view, "/exercises": self.exercises, "/projects": self.projects,
                  "/search": self.search, "/progress": self.progress_page}
        handler = routes.get(url.path)
        if handler is None:
            self.send_html(self.page("Introuvable", "<h1>Page introuvable</h1>", catalog), 404)
            return
        handler(query, catalog)

    def home(self, query, catalog: Catalog):
        tracker = self.tracker()
        parts = ["<h1>Offline AI Learning Lab</h1>",
                 "<p>Environnement d'apprentissage local : Python, mathématiques, données, Machine Learning, Deep Learning, NLP, vision et projets. "
                 "Tout fonctionne sans Internet.</p>",
                 f'<p>Module en cours : <strong>{html.escape(tracker.data["current_module"])}</strong> · '
                 f'<a href="{view_url(ROOT / "docs" / "learning_roadmap.md")}">parcours conseillé</a> · '
                 f'<a href="{view_url(ROOT / "README.md")}">README</a></p>']
        for module in catalog.modules:
            parts.append(f'<h2><a href="{view_url(module.readme)}">{html.escape(module.title)}</a></h2><ul>')
            for lesson in module.lessons:
                status = '<span class="done">✓</span> ' if tracker.is_completed("lessons", lesson.id) else ""
                parts.append(f'<li>{status}<a href="{view_url(lesson.path)}">{html.escape(lesson.title)}</a></li>')
            parts.append("</ul>")
        self.send_html(self.page("Accueil", "".join(parts), catalog))

    def view(self, query, catalog: Catalog):
        path = safe_path(query.get("path", [""])[0])
        if path is None or path.is_dir() or path.suffix not in TEXT_SUFFIXES:
            self.send_html(self.page("Introuvable", "<h1>Fichier introuvable</h1>", catalog), 404)
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = rel(path)
        vscode = f'vscode://file/{quote(path.as_posix())}'
        header = f'<p class="muted">{html.escape(relative)} · <a href="{vscode}">ouvrir dans VS Code</a></p>'
        if path.suffix == ".md":
            body = md.to_html(text, link_resolver=self.link_resolver(path))
        else:
            body = f"<pre><code>{html.escape(text[:200_000])}</code></pre>"
        actions = ""
        if relative in catalog.items:
            category, item_id, _ = catalog.items[relative]
            done = self.tracker().is_completed(category, item_id)
            label = {"lessons": "leçon", "exercises": "exercice", "projects": "projet"}[category]
            button = (f'<span class="done">✓ {label} terminé(e)</span> <button class="secondary" name="action" value="undo">Annuler</button>'
                      if done else f'<button name="action" value="complete">Marquer ce(tte) {label} comme terminé(e)</button>')
            actions = (f'<form class="actions" method="post" action="/progress"><input type="hidden" name="category" value="{category}">'
                       f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
                       f'<input type="hidden" name="next" value="{html.escape(view_url(path))}">{button}</form>')
            if category == "lessons":
                notebook = next(ROOT.glob(f"notebooks/*/{item_id[:2]}_{path.stem}.ipynb"), None)
                if notebook:
                    actions += (f'<p class="muted">Notebook : <a href="vscode://file/{quote(notebook.as_posix())}">{html.escape(rel(notebook))}</a> · '
                                f'exécuter tout le code : <code>python -m lab.lesson_runner {html.escape(relative)}</code></p>')
            if category == "exercises":
                exercise_dir = ROOT / relative.rsplit("/", 1)[0]
                actions += (f'<p class="muted">Fichier à compléter : <a href="{view_url(exercise_dir / "exercise.py")}">exercise.py</a> · '
                            f'vérification : <code>python -m pytest {html.escape(rel(exercise_dir))} -q</code></p>')
        for module in catalog.modules:
            if path == module.readme:
                actions += (f'<form class="actions" method="post" action="/progress"><input type="hidden" name="category" value="module">'
                            f'<input type="hidden" name="item_id" value="{module.id}"><input type="hidden" name="next" value="{html.escape(view_url(path))}">'
                            f'<button class="secondary" name="action" value="set_module">Définir comme module en cours</button></form>')
        self.send_html(self.page(path.stem, header + actions + body + actions, catalog))

    def exercises(self, query, catalog: Catalog):
        tracker = self.tracker()
        parts = ["<h1>Exercices</h1><p>Chaque exercice contient un énoncé, un fichier <code>exercise.py</code> à compléter et des tests. "
                 "Les solutions sont dans <code>solutions/</code>.</p>"]
        for level, title in LEVEL_NAMES.items():
            parts.append(f"<h2>{title}</h2><ul>")
            for exercise in [e for e in catalog.exercises if e.level == level]:
                status = '<span class="done">✓</span> ' if tracker.is_completed("exercises", exercise.id) else ""
                parts.append(f'<li>{status}<a href="{view_url(exercise.readme)}">{html.escape(exercise.title)}</a></li>')
            parts.append("</ul>")
        self.send_html(self.page("Exercices", "".join(parts), catalog))

    def projects(self, query, catalog: Catalog):
        tracker = self.tracker()
        parts = ["<h1>Projets</h1><ol>"]
        for project in catalog.projects:
            status = '<span class="done">✓</span> ' if tracker.is_completed("projects", project.id) else ""
            parts.append(f'<li>{status}<a href="{view_url(project.readme)}">{html.escape(project.title)}</a> '
                         f'<span class="muted">· <code>python projects/{project.id}/run.py</code></span></li>')
        parts.append("</ol>")
        self.send_html(self.page("Projets", "".join(parts), catalog))

    def search(self, query, catalog: Catalog):
        term = query.get("q", [""])[0].strip()
        results = []
        if len(term) >= 2:
            needle = fold(term)
            files = [(m.title + " — " + lesson.title, lesson.path) for m in catalog.modules for lesson in m.lessons]
            files += [(f"Exercice {e.id} — {e.title}", e.readme) for e in catalog.exercises]
            files += [(p.title, p.readme) for p in catalog.projects]
            for title, path in files:
                text = path.read_text(encoding="utf-8")
                folded = fold(text)
                count = folded.count(needle)
                if count:
                    position = folded.index(needle)
                    start, end = max(0, position - 90), min(len(text), position + len(needle) + 90)
                    snippet = html.escape(text[start:end].replace("\n", " "))
                    snippet = re.sub(re.escape(html.escape(text[position:position + len(needle)])), lambda m: f"<mark>{m.group(0)}</mark>", snippet, count=1)
                    results.append((count, title, path, snippet))
        results.sort(key=lambda r: -r[0])
        parts = [f"<h1>Recherche : {html.escape(term)}</h1>",
                 f"<p class='muted'>{len(results)} document(s) trouvé(s) (recherche insensible aux accents et à la casse).</p>"]
        for count, title, path, snippet in results[:50]:
            parts.append(f'<p><a href="{view_url(path)}"><strong>{html.escape(title)}</strong></a> '
                         f'<span class="badge">{count} occurrence(s)</span><br><span class="muted">…{snippet}…</span></p>')
        self.send_html(self.page("Recherche", "".join(parts), catalog))

    def progress_page(self, query, catalog: Catalog):
        tracker = self.tracker()

        def bar(ratio):
            return f'<span class="bar"><span style="width:{ratio * 100:.0f}%"></span></span> {ratio:.0%}'

        parts = ["<h1>Progression</h1>", f'<p class="muted">Fichier : {html.escape(str(tracker.path))}. '
                 "Rien n'est marqué automatiquement : utilisez les boutons des pages.</p><table><tr><th>Module</th><th>Leçons</th><th>Meilleur quiz</th></tr>"]
        for module in catalog.modules:
            ratio = tracker.completion_ratio("lessons", [lesson.id for lesson in module.lessons])
            quiz = tracker.best_quiz(module.id)
            quiz_text = f"{quiz['score']:g} / {quiz['max_score']:g}" if quiz else "—"
            parts.append(f"<tr><td>{html.escape(module.title)}</td><td>{bar(ratio)}</td><td>{quiz_text}</td></tr>")
        parts.append("</table>")
        for category, items, title in [("exercises", [e.id for e in catalog.exercises], "Exercices"), ("projects", [p.id for p in catalog.projects], "Projets")]:
            parts.append(f"<p><strong>{title}</strong> : {bar(tracker.completion_ratio(category, items))}</p>")
        options = "".join(f'<option value="{m.id}">{html.escape(m.title)}</option>' for m in catalog.modules)
        parts.append(f"""<h2>Enregistrer un score de quiz</h2>
<form method="post" action="/quiz" class="actions"><select name="module_id" aria-label="Module">{options}</select>
<input name="score" type="number" step="0.5" min="0" placeholder="score" required aria-label="Score">
<input name="max_score" type="number" step="0.5" min="1" value="6" required aria-label="Maximum">
<button>Enregistrer</button></form>
<p class="muted">Journal d'apprentissage : <a href="{view_url(ROOT / 'progress' / 'learning_log.md')}">progress/learning_log.md</a></p>""")
        self.send_html(self.page("Progression", "".join(parts), catalog))

    # -- POST -----------------------------------------------------------------
    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost"}:
            self.send_error(HTTPStatus.FORBIDDEN, "cross-origin request refused")
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8")).items()}
        url = urlparse(self.path).path
        tracker = self.tracker()
        catalog = Catalog()
        try:
            if url == "/progress":
                category, item_id, action = form["category"], form["item_id"], form["action"]
                if category == "module" and action == "set_module":
                    if item_id not in {m.id for m in catalog.modules}:
                        raise ValueError("unknown module")
                    tracker.set_current_module(item_id)
                else:
                    known = {(c, i) for c, i, _ in catalog.items.values()}
                    if (category, item_id) not in known:
                        raise ValueError("unknown item")
                    tracker.set_completed(category, item_id, completed=(action == "complete"))
                tracker.save()
                next_url = form.get("next", "/")
                self.redirect(next_url if next_url.startswith("/") else "/")
            elif url == "/quiz":
                if form["module_id"] not in {m.id for m in catalog.modules}:
                    raise ValueError("unknown module")
                tracker.record_quiz(form["module_id"], float(form["score"]), float(form["max_score"]))
                tracker.save()
                self.redirect("/progress")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as exc:
            self.send_html(self.page("Erreur", f"<h1>Requête invalide</h1><p>{html.escape(str(exc))}</p>", catalog), 400)


def create_server(port: int = 8765, progress_file: Path = PROGRESS_FILE, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("ConfiguredLabHandler", (LabHandler,), {"progress_file": Path(progress_file)})
    return ThreadingHTTPServer((host, port), handler)


def start_jupyter() -> subprocess.Popen | None:
    try:
        import jupyterlab  # noqa: F401
    except ImportError:
        print("JupyterLab is not installed. Install it once (online) with: python -m pip install -r requirements.txt")
        return None
    command = [sys.executable, "-m", "jupyter", "lab", "--no-browser", "--ip=127.0.0.1", f"--notebook-dir={ROOT}"]
    print("Starting JupyterLab locally (the URL with its token is printed below)...")
    return subprocess.Popen(command, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in the default web browser")
    parser.add_argument("--jupyter", action="store_true", help="also start JupyterLab on 127.0.0.1")
    parser.add_argument("--progress-file", type=Path, default=PROGRESS_FILE)
    args = parser.parse_args(argv)
    server = create_server(args.port, args.progress_file)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Offline AI Learning Lab running at {url} (Ctrl+C to stop). Only local files are used.")
    jupyter = start_jupyter() if args.jupyter else None
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        if jupyter:
            jupyter.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
