"""PDF export, course index, notebooks, local web interface, offline HTML, setup scripts."""

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from lab.catalog import list_exercises, list_lessons, list_modules
from lab.paths import ROOT


def load_script(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def run_script(*args, timeout=300):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


# --- PDF ---------------------------------------------------------------------
def test_pdf_export_one_module(tmp_path):
    pytest.importorskip("reportlab")
    pdf = load_script("scripts/export_lessons_to_pdf.py")
    module = next(m for m in list_modules() if m.id.startswith("04"))
    path = pdf.export_module(module, tmp_path)
    content = path.read_bytes()
    assert content.startswith(b"%PDF") and len(content) > 50_000
    assert content.count(b"/Type /Page") > 5
    assert b"DejaVuSans" in content, "the Unicode font must be embedded"


def test_pdf_wraps_long_code_lines_and_keeps_indentation():
    pdf = load_script("scripts/export_lessons_to_pdf.py")
    wrapped = pdf.wrap_code("    " + "x = 1 + " * 40, 60).split("\n")
    assert len(wrapped) > 1 and all(len(line) <= 60 for line in wrapped)
    assert wrapped[1].startswith("    " + pdf.CONTINUATION)


def test_pdf_inline_markup_escapes():
    pdf = load_script("scripts/export_lessons_to_pdf.py")
    assert "&lt;" in pdf.inline_markup("a < b")
    assert "<b>gras</b>" in pdf.inline_markup("**gras**")


def test_pdf_fonts_cover_the_symbols_used_in_lessons():
    pdf = load_script("scripts/export_lessons_to_pdf.py")
    from fontTools.ttLib import TTFont

    cmap = TTFont(pdf.find_fonts()["DejaVuSans"]).getBestCmap()
    for character in "éèàçœ«»θσμΣ√≤≥≈∂∇−×·→₁²":
        assert ord(character) in cmap, character


# --- index and notebooks -------------------------------------------------------
def test_course_index_is_up_to_date():
    result = run_script("scripts/generate_course_index.py", "--check")
    assert result.returncode == 0, result.stdout


def test_internal_markdown_links_exist():
    broken = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)#\s]+)(?:#[^)]*)?\)")
    sources = [p for p in ROOT.rglob("*.md") if not ({".venv", "outputs", "exports", ".pytest_cache", ".claude"} & set(p.relative_to(ROOT).parts))]
    for path in sources:
        code_free = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in pattern.findall(code_free):
            if not target.startswith(("http://", "https://")) and not (path.parent / target).exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, broken


def test_notebooks_are_valid_and_complete(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    build = load_script("scripts/build_notebooks.py")
    assert build.main(["--output-dir", str(tmp_path)]) == 0
    notebooks = list(tmp_path.rglob("*.ipynb"))
    assert len(notebooks) == len(list_lessons())
    for path in notebooks[:5]:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert any(c.cell_type == "code" for c in notebook.cells)


def test_split_cells_keeps_no_run_blocks_as_markdown():
    build = load_script("scripts/build_notebooks.py")
    cells = build.split_cells("texte\n\n```python\nprint(1)\n```\n\n```python no-run\nprint(2)\n```\n")
    assert [kind for kind, _ in cells] == ["markdown", "code", "markdown"]
    assert "print(2)" in cells[2][1]


@pytest.mark.slow
def test_a_notebook_really_executes(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    build = load_script("scripts/build_notebooks.py")
    lesson = list_lessons()[0]
    notebook = build.lesson_notebook(lesson)
    nbclient.NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
    assert any(o for c in notebook.cells if c.cell_type == "code" for o in c.outputs)
    nbformat.validate(notebook)


# --- local web interface -------------------------------------------------------
@pytest.fixture(scope="module")
def web(tmp_path_factory):
    ui = load_script("scripts/run_learning_environment.py")
    progress = tmp_path_factory.mktemp("web") / "progress.json"
    server = ui.create_server(0, progress)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", progress
    server.shutdown()
    server.server_close()


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, ""


def post(url, **fields):
    request = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode())
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_web_pages(web):
    base, _ = web
    for page in ["/", "/exercises", "/projects", "/progress"]:
        assert fetch(base + page)[0] == 200, page
    status, body = fetch(base + "/view?path=courses/05_machine_learning/01_introduction_ml.md")
    assert status == 200 and "<h1" in body and "Marquer" in body
    assert "http://" not in re.sub(r'<pre>.*?</pre>', "", body, flags=re.S).split("<main>")[0], "no external resource in the page head"


def test_web_search_ignores_accents(web):
    base, _ = web
    _, with_accent = fetch(base + "/search?q=r%C3%A9gression")
    _, without_accent = fetch(base + "/search?q=regression")
    assert with_accent.count("occurrence(s)") == without_accent.count("occurrence(s)") > 5


def test_web_refuses_paths_outside_the_repository(web):
    base, _ = web
    assert fetch(base + "/view?path=../../Windows/win.ini")[0] == 404
    assert fetch(base + "/view?path=.venv/pyvenv.cfg")[0] == 404


def test_web_progress_only_on_explicit_post(web):
    base, progress = web
    fetch(base + "/view?path=courses/01_python_for_data_science/01_bases_python.md")
    assert not progress.exists(), "reading a lesson must not change the progress"
    item = "01_python_for_data_science/01_bases_python"
    assert post(base + "/progress", category="lessons", item_id=item, action="complete", next="/") == 200
    assert item in json.loads(progress.read_text(encoding="utf-8"))["lessons"]
    assert post(base + "/progress", category="lessons", item_id="nope", action="complete") == 400
    assert post(base + "/quiz", module_id="01_python_for_data_science", score="5", max_score="6") == 200


def test_web_server_listens_on_localhost_only():
    ui = load_script("scripts/run_learning_environment.py")
    server = ui.create_server(0)
    assert server.server_address[0] == "127.0.0.1"
    server.server_close()


# --- single-file offline HTML ----------------------------------------------------
@pytest.fixture(scope="module")
def offline_html(tmp_path_factory):
    builder = load_script("scripts/build_offline_html.py")
    path = tmp_path_factory.mktemp("html") / "lab.html"
    assert builder.main(["--output", str(path)]) == 0
    text = path.read_text(encoding="utf-8")
    payload = re.search(r'<script id="lab-data" type="application/json">(.*?)</script>', text, re.S).group(1)
    return text, json.loads(payload)


def test_offline_html_contains_everything(offline_html):
    _, data = offline_html
    pages = data["pages"]
    for lesson in list_lessons():
        assert lesson.relative_path in pages and "<h1" in pages[lesson.relative_path]["html"]
    for exercise in list_exercises():
        page = pages[exercise.readme.relative_to(ROOT).as_posix()]
        assert any(entry.get("hidden") for entry in page["extra"]), "solutions stay behind a click"
        assert exercise.solution.relative_to(ROOT).as_posix() in data["files"]
    assert "README.md" in pages and "datasets/README.md" in pages and "preview" in pages["datasets/README.md"]
    assert data["meta"]["lessons"] == 39 and data["meta"]["exercises"] == 21 and data["meta"]["projects"] == 6


def test_offline_html_loads_nothing_from_the_network(offline_html):
    text, _ = offline_html
    shell = re.sub(r'<script id="lab-data".*?</script>', "", text, flags=re.S)
    assert "default-src 'none'" in text and "connect-src 'none'" in text
    assert not re.search(r'(src|href)="https?://', shell)
    assert "@import" not in shell and "url(http" not in shell
    assert not re.search(r"\bfetch\(|XMLHttpRequest|WebSocket", shell)


def test_offline_html_internal_links_resolve(offline_html):
    _, data = offline_html
    broken = []
    for path, page in data["pages"].items():
        for kind, target in re.findall(r'href="#/([pf])/([^"]+)"', page["html"]):
            if target not in (data["pages"] if kind == "p" else data["files"]):
                broken.append(f"{path} -> {target}")
    assert not broken, broken


def test_offline_html_excludes_private_and_generated_files(offline_html):
    _, data = offline_html
    assert not [p for p in data["tree"] if p.startswith((".venv/", ".git/", "outputs/", "exports/"))]


def test_committed_offline_html_exists():
    path = ROOT / "offline_ai_learning_lab.html"
    assert path.exists() and path.stat().st_size < 20_000_000


# --- setup scripts -----------------------------------------------------------------
def test_check_environment_reports_ok():
    result = run_script("setup/check_environment.py", "--quick")
    assert result.returncode == 0, result.stdout
    assert "numpy" in result.stdout and "torch" in result.stdout


def test_check_environment_detects_missing_package():
    check = load_script("setup/check_environment.py")
    fake = check.Requirement("package-that-does-not-exist", "nope", "1.0", "core", "test")
    status, _, detail = check.check_requirement(fake)
    assert status == "MISSING" and "pip install" in detail
    old = check.Requirement("numpy", "numpy", "999.0", "core", "test")
    assert check.check_requirement(old, try_import=False)[0] == "OLD"


def test_setup_does_not_install_without_explicit_flag():
    setup = load_script("setup/setup_environment.py")
    commands = setup.install_commands(ROOT / "python", with_torch=False, with_dev=False)
    assert all("torch" not in " ".join(c) for c in commands)
    assert "--install" in (setup.__doc__ or "")


def test_source_scan_finds_no_download_call():
    verify = load_script("setup/verify_offline.py")
    assert verify.scan_sources() == []


@pytest.mark.slow
def test_verify_offline_passes():
    result = run_script("setup/verify_offline.py", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:]
    assert "OFFLINE READY" in result.stdout
