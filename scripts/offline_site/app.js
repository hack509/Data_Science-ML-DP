// Offline AI Learning Lab - single-file reader. No network access: all content is in #lab-data.
(function () {
  "use strict";

  const DATA = JSON.parse(document.getElementById("lab-data").textContent);
  const view = document.getElementById("view");
  const sidebar = document.getElementById("sidebar");
  const STORE_KEY = "offline-ai-lab-progress";
  const THEME_KEY = "offline-ai-lab-theme";

  // --- storage (a browser may refuse localStorage on file:// pages) ----------
  let storageOk = true;
  function readStore(key) {
    try { return localStorage.getItem(key); } catch (error) { storageOk = false; return null; }
  }
  function writeStore(key, value) {
    try { localStorage.setItem(key, value); } catch (error) { storageOk = false; }
  }
  function emptyProgress() {
    return { schema_version: 1, learner: "", current_module: DATA.modules[0].id, lessons: {}, exercises: {}, projects: {}, quizzes: {}, updated_at: null };
  }
  function loadProgress() {
    const base = emptyProgress();
    try { return Object.assign(base, JSON.parse(readStore(STORE_KEY) || "{}")); } catch (error) { return base; }
  }
  let progress = loadProgress();
  function saveProgress() {
    progress.updated_at = new Date().toISOString().slice(0, 19);
    writeStore(STORE_KEY, JSON.stringify(progress));
  }
  function isDone(item) { return Boolean(item && progress[item.category] && progress[item.category][item.id]); }
  function setDone(item, done) {
    if (done) progress[item.category][item.id] = { completed: true, completed_at: new Date().toISOString().slice(0, 19), note: "" };
    else delete progress[item.category][item.id];
    saveProgress();
  }

  // --- helpers ---------------------------------------------------------------
  function esc(text) { return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function fold(text) { return text.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase(); }
  function pageHref(path) { return "#/p/" + encodeURI(path); }
  function fileHref(path) { return "#/f/" + encodeURI(path); }
  function ratio(items) {
    if (!items.length) return 0;
    return items.filter(isDone).length / items.length;
  }
  function bar(value) { return `<div class="bar" role="img" aria-label="${Math.round(value * 100)} %"><span style="width:${value * 100}%"></span></div>`; }

  const PY_KEYWORDS = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" "));
  const PY_TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfRBF]{0,2}"(?:\\.|[^"\\\n])*"|[rbfRBF]{0,2}'(?:\\.|[^'\\\n])*')|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)\b(?=(\s*\())?/g;
  function highlightPython(code) {
    let out = "", last = 0, match;
    PY_TOKEN.lastIndex = 0;
    while ((match = PY_TOKEN.exec(code))) {
      out += esc(code.slice(last, match.index));
      const [token, comment, string, number, word, call] = match;
      if (comment) out += `<span class="tok-com">${esc(token)}</span>`;
      else if (string) out += `<span class="tok-str">${esc(token)}</span>`;
      else if (number) out += `<span class="tok-num">${esc(token)}</span>`;
      else if (PY_KEYWORDS.has(word)) out += `<span class="tok-kw">${esc(word)}</span>`;
      else if (call !== undefined) out += `<span class="tok-fn">${esc(word)}</span>`;
      else out += esc(word);
      last = match.index + token.length;
    }
    return out + esc(code.slice(last));
  }

  function copyText(text, button) {
    const done = () => { button.textContent = "Copié"; setTimeout(() => { button.textContent = "Copier"; }, 1200); };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
    } else fallbackCopy(text, done);
  }
  function fallbackCopy(text, done) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); done(); } catch (error) { /* copy unavailable */ }
    area.remove();
  }

  function enhance(root) {
    root.querySelectorAll("pre").forEach((pre) => {
      const code = pre.querySelector("code");
      if (!code) return;
      const raw = code.textContent;
      if (code.classList.contains("language-python")) code.innerHTML = highlightPython(raw);
      const button = document.createElement("button");
      button.className = "copy";
      button.type = "button";
      button.textContent = "Copier";
      button.addEventListener("click", () => copyText(raw, button));
      pre.appendChild(button);
    });
    root.querySelectorAll("table").forEach((table) => {
      if (table.parentElement.classList.contains("table-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
    root.querySelectorAll('a[href^="#"]').forEach((link) => {
      const target = link.getAttribute("href");
      if (target.startsWith("#/")) return;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        const anchor = root.querySelector(`[id="${CSS.escape(target.slice(1))}"]`);
        if (anchor) anchor.scrollIntoView();
      });
    });
  }

  // --- sidebar ---------------------------------------------------------------
  function navLink(href, label, item) {
    const mark = isDone(item) ? '<span class="check" aria-label="terminé">✓</span>' : "";
    return `<a href="${href}" data-href="${esc(href)}">${mark}<span>${esc(label)}</span></a>`;
  }
  function renderSidebar() {
    const openKeys = new Set([...sidebar.querySelectorAll("details[open]")].map((d) => d.dataset.key));
    const details = (key, title, body) => `<details data-key="${esc(key)}"${openKeys.has(key) ? " open" : ""}><summary>${esc(title)}</summary><div>${body}</div></details>`;
    let html = navLink("#/", "Accueil") + navLink("#/progress", "Ma progression");
    html += '<div class="nav-title">Cours</div>';
    DATA.modules.forEach((module) => {
      let body = navLink(pageHref(module.readme), "Présentation du module");
      module.lessons.forEach((lesson) => { body += navLink(pageHref(lesson.path), lesson.title.replace(/^Leçon /, ""), lesson.item); });
      body += navLink(pageHref(module.corrections), "Corrigés");
      html += details(module.id, module.title.replace(/^Module /, ""), body);
    });
    html += '<div class="nav-title">Pratique</div>';
    DATA.levels.forEach((level) => {
      const body = DATA.exercises.filter((e) => e.level === level.id).map((e) => navLink(pageHref(e.path), e.title, e.item)).join("");
      html += details("level-" + level.id, "Exercices — " + level.title, body);
    });
    html += details("projects", "Projets", DATA.projects.map((p) => navLink(pageHref(p.path), p.title.replace(/^Projet /, ""), p.item)).join(""));
    html += '<div class="nav-title">Ressources</div>';
    html += navLink(pageHref("datasets/README.md"), "Jeux de données") + navLink("#/figures", "Galerie des figures");
    DATA.docs.forEach((doc) => { html += navLink(pageHref(doc.path), doc.title); });
    html += navLink("#/tree", "Fichiers du dépôt");
    sidebar.innerHTML = html;
  }
  function markActive() {
    const hash = decodeURI(location.hash || "#/");
    sidebar.querySelectorAll("a").forEach((link) => {
      const active = decodeURI(link.dataset.href) === hash;
      link.classList.toggle("active", active);
      if (active) {
        let parent = link.parentElement;
        while (parent && parent !== sidebar) {
          if (parent.tagName === "DETAILS") parent.open = true;
          parent = parent.parentElement;
        }
      }
    });
  }

  // --- views -----------------------------------------------------------------
  function figureHtml(figure) {
    return `<figure><img src="${DATA.images[figure.key]}" alt="${esc(figure.caption)}" loading="lazy"><figcaption>${esc(figure.caption)}</figcaption></figure>`;
  }
  function viewHome() {
    const m = DATA.meta;
    const cards = DATA.modules.map((module) => {
      const value = ratio(module.lessons.map((l) => l.item));
      return `<a class="card" href="${pageHref(module.readme)}"><strong>${esc(module.title)}</strong><span class="muted">${module.lessons.length} leçons</span>${bar(value)}</a>`;
    }).join("");
    const projects = DATA.projects.map((p) => `<a class="card" href="${pageHref(p.path)}"><strong>${esc(p.title)}</strong>${isDone(p.item) ? '<span class="done-label">✓ terminé</span>' : ""}</a>`).join("");
    return `<article>
      <h1>Offline AI Learning Lab</h1>
      <p>Un cours complet, en français, de Python à l'IA : mathématiques, analyse de données, Machine Learning, Deep Learning, NLP, vision par ordinateur et projets. <strong>Ce fichier unique fonctionne sans Internet</strong> : téléchargez-le et ouvrez-le dans un navigateur.</p>
      <div class="stats">
        <div class="stat"><strong>${m.modules}</strong><span>modules</span></div>
        <div class="stat"><strong>${m.lessons}</strong><span>leçons</span></div>
        <div class="stat"><strong>${m.exercises}</strong><span>exercices</span></div>
        <div class="stat"><strong>${m.projects}</strong><span>projets</span></div>
        <div class="stat"><strong>${m.datasets}</strong><span>jeux de données</span></div>
      </div>
      <p class="actions"><a class="button" href="${pageHref("docs/learning_roadmap.md")}">Voir le parcours</a><a class="button secondary" href="${pageHref(DATA.modules[0].lessons[0].path)}">Commencer la leçon 1.1</a><a class="button secondary" href="${pageHref("README.md")}">README du dépôt</a></p>
      <div class="notice">Ce fichier sert à <strong>lire</strong> le contenu. Pour exécuter le code, les exercices et les projets, installez le dépôt localement : voir le <a href="${pageHref("docs/installation_guide.md")}">guide d'installation</a>.</div>
      <h2>Modules</h2><div class="cards">${cards}</div>
      <h2>Projets</h2><div class="cards">${projects}</div>
    </article>`;
  }

  function doneButton(item) {
    if (!item) return "";
    const label = { lessons: "cette leçon comme terminée", exercises: "cet exercice comme terminé", projects: "ce projet comme terminé" }[item.category];
    return isDone(item)
      ? `<p class="actions"><span class="done-label">✓ Terminé</span><button class="button secondary" data-action="undo" type="button">Annuler</button></p>`
      : `<p class="actions"><button class="button" data-action="done" type="button">Marquer ${label}</button></p>`;
  }
  function fileBlock(entry) {
    const file = DATA.files[entry.path];
    if (!file) return "";
    const language = entry.path.endsWith(".py") ? "language-python" : "language-text";
    const open = entry.open ? " open" : "";
    const warning = entry.hidden ? '<p class="notice">Solution : ne l\'ouvrez qu\'après avoir essayé.</p>' : "";
    return `<details${open}><summary>${esc(entry.label)}</summary><div>${warning}<pre><code class="${language}">${esc(file.text)}</code></pre></div></details>`;
  }
  function viewPage(path) {
    const page = DATA.pages[path];
    if (!page) return viewMissing();
    const figures = (page.figures || []).map((key) => DATA.figures.find((f) => f.key === key)).filter(Boolean);
    const figureSection = figures.length ? `<h2>Figures produites par cette page</h2><div class="gallery">${figures.map(figureHtml).join("")}</div>` : "";
    const tabs = page.extra && page.extra.length ? `<h2>Fichiers</h2><div class="tabs">${page.extra.map(fileBlock).join("")}</div>` : "";
    const links = [];
    if (page.prev) links.push(`<a href="${pageHref(page.prev.path)}">← ${esc(page.prev.title)}</a>`); else links.push("<span></span>");
    if (page.next) links.push(`<a href="${pageHref(page.next.path)}">${esc(page.next.title)} →</a>`);
    const notice = page.kind === "corrections" ? '<div class="notice">Corrigés : essayez de répondre avant de lire.</div>' : "";
    return `<article><p class="path">${esc(path)}</p>${notice}${doneButton(page.item)}${page.html}${page.preview || ""}${figureSection}${tabs}${doneButton(page.item)}<nav class="pager" aria-label="Pages voisines">${links.join("")}</nav></article>`;
  }
  function viewFile(path) {
    const file = DATA.files[path];
    if (!file) return viewMissing();
    const language = path.endsWith(".py") ? "language-python" : "language-text";
    const note = file.truncated ? `<p class="notice">Aperçu tronqué : ${esc(file.truncated)}.</p>` : "";
    return `<article><h1>${esc(path.split("/").pop())}</h1><p class="path">${esc(path)}</p>${note}<pre><code class="${language}">${esc(file.text)}</code></pre></article>`;
  }
  function viewMissing() { return '<article><h1>Page introuvable</h1><p><a href="#/">Retour à l\'accueil</a></p></article>'; }

  function viewFigures() {
    const groups = {};
    DATA.figures.forEach((figure) => { (groups[figure.group] = groups[figure.group] || []).push(figure); });
    const body = Object.keys(groups).sort().map((group) => `<h2>${esc(group)}</h2><div class="gallery">${groups[group].map(figureHtml).join("")}</div>`).join("");
    return `<article><h1>Galerie des figures</h1><p>Figures réellement produites par le code des leçons et des projets (${DATA.figures.length}).</p>${body || "<p>Aucune figure n'a été incluse : exécutez les leçons avant de générer ce fichier.</p>"}</article>`;
  }

  function treeHtml(node) {
    const folders = Object.keys(node.dirs).sort().map((name) => {
      const child = node.dirs[name];
      return `<details><summary>${esc(name)}/ <span class="muted">(${child.count})</span></summary>${treeHtml(child)}</details>`;
    }).join("");
    const files = node.files.map((path) => {
      const name = esc(path.split("/").pop());
      if (DATA.pages[path]) return `<a href="${pageHref(path)}">${name}</a>`;
      if (DATA.files[path]) return `<a href="${fileHref(path)}">${name}</a>`;
      return `<span class="file muted">${name}</span>`;
    }).join("");
    return folders + files;
  }
  function viewTree() {
    const root = { dirs: {}, files: [], count: 0 };
    DATA.tree.forEach((path) => {
      let node = root;
      node.count += 1;
      const parts = path.split("/");
      parts.slice(0, -1).forEach((part) => {
        node.dirs[part] = node.dirs[part] || { dirs: {}, files: [], count: 0 };
        node = node.dirs[part];
        node.count += 1;
      });
      node.files.push(path);
    });
    return `<article><h1>Fichiers du dépôt</h1><p>${DATA.tree.length} fichiers. Les fichiers grisés (images, notebooks, archives) sont listés mais non inclus ici ; ils sont dans le dépôt.</p><div class="tree">${treeHtml(root)}</div></article>`;
  }

  function viewProgress() {
    const rows = DATA.modules.map((module) => {
      const items = module.lessons.map((l) => l.item);
      const count = items.filter(isDone).length;
      return `<tr><td><a href="${pageHref(module.readme)}">${esc(module.title)}</a></td><td>${count} / ${items.length}</td></tr>`;
    }).join("");
    const exercises = DATA.exercises.map((e) => e.item);
    const projects = DATA.projects.map((p) => p.item);
    const warning = storageOk ? "" : '<p class="notice">Ce navigateur refuse le stockage local pour ce fichier : la progression sera perdue à la fermeture. Exportez-la avant de quitter.</p>';
    return `<article><h1>Ma progression</h1>${warning}
      <p>Stockée uniquement dans ce navigateur, jamais envoyée ailleurs. Rien n'est marqué automatiquement. Le fichier exporté a le même format que <code>progress/progress.json</code> du dépôt.</p>
      <div class="stats">
        <div class="stat"><strong>${exercises.filter(isDone).length} / ${exercises.length}</strong><span>exercices</span></div>
        <div class="stat"><strong>${projects.filter(isDone).length} / ${projects.length}</strong><span>projets</span></div>
      </div>
      <table><thead><tr><th>Module</th><th>Leçons terminées</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="actions"><button class="button" data-action="export" type="button">Exporter (JSON)</button>
      <label class="button secondary">Importer<input id="import-file" type="file" accept=".json" hidden></label>
      <button class="button secondary" data-action="reset" type="button">Tout réinitialiser</button></p></article>`;
  }

  // --- search ----------------------------------------------------------------
  let searchIndex = null;
  function buildIndex() {
    searchIndex = Object.keys(DATA.pages).map((path) => {
      const page = DATA.pages[path];
      const text = page.html.replace(/<[^>]+>/g, " ").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/\s+/g, " ");
      return { path, title: page.title, kind: page.kind, text, folded: fold(text), foldedTitle: fold(page.title) };
    });
  }
  function viewSearch(query) {
    if (!searchIndex) buildIndex();
    const terms = fold(query).split(/\s+/).filter((t) => t.length > 1);
    if (!terms.length) return "<article><h1>Recherche</h1><p>Saisissez au moins deux lettres.</p></article>";
    const results = searchIndex.map((entry) => {
      let score = 0;
      for (const term of terms) {
        const count = entry.folded.split(term).length - 1;
        if (!count) return null;
        score += count + (entry.foldedTitle.includes(term) ? 25 : 0);
      }
      return { entry, score };
    }).filter(Boolean).sort((a, b) => b.score - a.score).slice(0, 40);
    const body = results.map(({ entry }) => {
      const position = entry.folded.indexOf(terms[0]);
      const start = Math.max(0, position - 90);
      const before = esc(entry.text.slice(start, position));
      const hit = esc(entry.text.slice(position, position + terms[0].length));
      const after = esc(entry.text.slice(position + terms[0].length, position + 130));
      return `<div class="result"><a href="${pageHref(entry.path)}"><strong>${esc(entry.title)}</strong></a><span class="kind">${esc(DATA.kinds[entry.kind] || entry.kind)}</span><br><span class="muted">…${before}<mark>${hit}</mark>${after}…</span></div>`;
    }).join("");
    return `<article><h1>Recherche : ${esc(query)}</h1><p class="muted">${results.length} résultat(s), sans tenir compte des accents ni de la casse.</p>${body}</article>`;
  }

  // --- router ----------------------------------------------------------------
  function route() {
    const hash = location.hash || "#/";
    let html;
    if (hash.startsWith("#/p/")) html = viewPage(decodeURI(hash.slice(4)));
    else if (hash.startsWith("#/f/")) html = viewFile(decodeURI(hash.slice(4)));
    else if (hash.startsWith("#/search/")) html = viewSearch(decodeURIComponent(hash.slice(9)));
    else if (hash === "#/figures") html = viewFigures();
    else if (hash === "#/tree") html = viewTree();
    else if (hash === "#/progress") html = viewProgress();
    else html = viewHome();
    view.innerHTML = html;
    enhance(view);
    const heading = view.querySelector("h1");
    document.title = (heading ? heading.textContent + " · " : "") + "Offline AI Learning Lab";
    markActive();
    sidebar.classList.remove("open");
    window.scrollTo(0, 0);
  }
  function currentItem() {
    const hash = location.hash;
    return hash.startsWith("#/p/") ? (DATA.pages[decodeURI(hash.slice(4))] || {}).item : null;
  }
  function refresh() { renderSidebar(); route(); }

  view.addEventListener("click", (event) => {
    const action = event.target.dataset && event.target.dataset.action;
    if (!action) return;
    if (action === "done" || action === "undo") { setDone(currentItem(), action === "done"); refresh(); }
    if (action === "reset" && confirm("Effacer toute la progression de ce navigateur ?")) { progress = emptyProgress(); saveProgress(); refresh(); }
    if (action === "export") {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([JSON.stringify(progress, null, 2)], { type: "application/json" }));
      link.download = "progress.json";
      link.click();
      URL.revokeObjectURL(link.href);
    }
  });
  view.addEventListener("change", (event) => {
    if (event.target.id !== "import-file" || !event.target.files[0]) return;
    event.target.files[0].text().then((text) => {
      const parsed = JSON.parse(text);
      if (!parsed.lessons || !parsed.exercises || !parsed.projects) throw new Error("format");
      progress = Object.assign(emptyProgress(), parsed);
      saveProgress();
      refresh();
    }).catch(() => alert("Ce fichier n'est pas un fichier de progression valide."));
  });

  // --- theme, menu, search box -------------------------------------------------
  const themeButton = document.getElementById("theme-button");
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeButton.textContent = theme === "dark" ? "Clair" : "Sombre";
    themeButton.setAttribute("aria-label", theme === "dark" ? "Passer au thème clair" : "Passer au thème sombre");
  }
  const preferred = window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(readStore(THEME_KEY) || preferred);
  themeButton.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    writeStore(THEME_KEY, next);
  });
  document.getElementById("menu-button").addEventListener("click", () => sidebar.classList.toggle("open"));
  document.getElementById("search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    location.hash = "#/search/" + encodeURIComponent(document.getElementById("search-input").value.trim());
  });

  window.addEventListener("hashchange", route);
  refresh();
})();
