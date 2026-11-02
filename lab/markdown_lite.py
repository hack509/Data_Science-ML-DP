"""A small, dependency-free Markdown parser for the course files.

It supports the subset of Markdown used in this repository:

* ATX headings (``#`` to ``######``)
* paragraphs with **bold**, *italic*, ``inline code`` and [links](url)
* fenced code blocks (``` with an optional language and attributes)
* bullet and numbered lists, with one level of nesting
* GitHub-style tables
* block quotes and horizontal rules

The parser produces a list of blocks that are rendered to HTML (local web
interface) or to ReportLab flowables (PDF export). Fenced blocks whose info
string is exactly ``python`` are considered executable lesson code; use
``python no-run`` for illustrative snippets that must not be executed.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class Heading:
    level: int
    text: str

    @property
    def anchor(self) -> str:
        return slugify(self.text)


@dataclass
class Paragraph:
    text: str


@dataclass
class CodeBlock:
    language: str
    attributes: list[str]
    code: str

    @property
    def executable(self) -> bool:
        return self.language == "python" and not self.attributes


@dataclass
class ListItem:
    text: str
    children: list[ListItem] = field(default_factory=list)


@dataclass
class ListBlock:
    ordered: bool
    items: list[ListItem]


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]


@dataclass
class Quote:
    text: str


@dataclass
class Rule:
    pass


Block = Heading | Paragraph | CodeBlock | ListBlock | Table | Quote | Rule

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_RULE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    # Allow escaped pipes inside cells.
    cells = re.split(r"(?<!\\)\|", stripped)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def parse(text: str) -> list[Block]:
    """Parse Markdown text into a list of blocks."""
    lines = text.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(2)
            info = fence.group(3).split()
            language = info[0].lower() if info else ""
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker):
                body.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            blocks.append(CodeBlock(language, info[1:], "\n".join(body)))
            continue

        heading = _HEADING.match(line)
        if heading:
            blocks.append(Heading(len(heading.group(1)), heading.group(2)))
            i += 1
            continue

        if _RULE.match(line):
            blocks.append(Rule())
            i += 1
            continue

        if line.lstrip().startswith("|") and i + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[i + 1]):
            header = _split_row(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(Table(header, rows))
            continue

        if line.lstrip().startswith(">"):
            quoted = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quoted.append(lines[i].lstrip()[1:].strip())
                i += 1
            blocks.append(Quote(" ".join(q for q in quoted if q) if quoted else ""))
            continue

        if _BULLET.match(line) or _NUMBERED.match(line):
            ordered = bool(_NUMBERED.match(line)) and not _BULLET.match(line)
            items: list[ListItem] = []
            base_indent = len(line) - len(line.lstrip())
            while i < len(lines):
                current = lines[i]
                if not current.strip():
                    # A blank line ends the list unless the next line continues it.
                    nxt = lines[i + 1] if i + 1 < len(lines) else ""
                    if _BULLET.match(nxt) or _NUMBERED.match(nxt):
                        i += 1
                        continue
                    break
                match = _BULLET.match(current) or _NUMBERED.match(current)
                indent = len(current) - len(current.lstrip())
                if match and indent <= base_indent + 1:
                    items.append(ListItem(match.group(2)))
                elif match and items:
                    items[-1].children.append(ListItem(match.group(2)))
                elif items and indent > base_indent:
                    # Continuation line of the previous item.
                    target = items[-1].children[-1] if items[-1].children else items[-1]
                    target.text += " " + current.strip()
                else:
                    break
                i += 1
            blocks.append(ListBlock(ordered, items))
            continue

        paragraph = [line.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if (
                not nxt.strip()
                or _HEADING.match(nxt)
                or _FENCE.match(nxt)
                or _BULLET.match(nxt)
                or _NUMBERED.match(nxt)
                or nxt.lstrip().startswith(("|", ">"))
                or _RULE.match(nxt)
            ):
                break
            paragraph.append(nxt.strip())
            i += 1
        blocks.append(Paragraph(" ".join(paragraph)))
    return blocks


# ---------------------------------------------------------------------------
# Inline formatting
# ---------------------------------------------------------------------------
_INLINE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<link>\[(?P<label>[^\]]+)\]\((?P<url>[^)\s]+)\))"
    r"|(?P<bold>\*\*(?P<bold_text>.+?)\*\*)"
    r"|(?P<italic>(?<![\w*])\*(?P<italic_text>[^*\s][^*]*?)\*(?![\w*]))"
)


def iter_inline(text: str):
    """Yield ``(kind, value, extra)`` tokens for inline Markdown."""
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            yield "text", text[position : match.start()], None
        if match.group("code"):
            yield "code", match.group("code")[1:-1], None
        elif match.group("link"):
            yield "link", match.group("label"), match.group("url")
        elif match.group("bold"):
            yield "bold", match.group("bold_text"), None
        else:
            yield "italic", match.group("italic_text"), None
        position = match.end()
    if position < len(text):
        yield "text", text[position:], None


def inline_to_html(text: str, link_resolver=None) -> str:
    parts = []
    for kind, value, extra in iter_inline(text):
        if kind == "text":
            parts.append(html.escape(value, quote=False))
        elif kind == "code":
            parts.append(f"<code>{html.escape(value, quote=False)}</code>")
        elif kind == "link":
            url = link_resolver(extra) if link_resolver else extra
            parts.append(f'<a href="{html.escape(url)}">{inline_to_html(value)}</a>')
        elif kind == "bold":
            parts.append(f"<strong>{inline_to_html(value)}</strong>")
        else:
            parts.append(f"<em>{inline_to_html(value)}</em>")
    return "".join(parts)


def inline_to_plain(text: str) -> str:
    return "".join(value for _, value, _ in iter_inline(text))


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def _list_to_html(block: ListBlock, link_resolver) -> str:
    tag = "ol" if block.ordered else "ul"
    out = [f"<{tag}>"]
    for item in block.items:
        out.append(f"<li>{inline_to_html(item.text, link_resolver)}")
        if item.children:
            out.append(_list_to_html(ListBlock(False, item.children), link_resolver))
        out.append("</li>")
    out.append(f"</{tag}>")
    return "".join(out)


def to_html(text_or_blocks, link_resolver=None) -> str:
    """Render Markdown (text or parsed blocks) to an HTML fragment."""
    blocks = parse(text_or_blocks) if isinstance(text_or_blocks, str) else text_or_blocks
    out = []
    for block in blocks:
        if isinstance(block, Heading):
            out.append(
                f'<h{block.level} id="{block.anchor}">{inline_to_html(block.text, link_resolver)}</h{block.level}>'
            )
        elif isinstance(block, Paragraph):
            out.append(f"<p>{inline_to_html(block.text, link_resolver)}</p>")
        elif isinstance(block, CodeBlock):
            language = html.escape(block.language or "text")
            out.append(f'<pre><code class="language-{language}">{html.escape(block.code, quote=False)}</code></pre>')
        elif isinstance(block, ListBlock):
            out.append(_list_to_html(block, link_resolver))
        elif isinstance(block, Table):
            head = "".join(f"<th>{inline_to_html(c, link_resolver)}</th>" for c in block.header)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline_to_html(c, link_resolver)}</td>" for c in row) + "</tr>"
                for row in block.rows
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif isinstance(block, Quote):
            out.append(f"<blockquote>{inline_to_html(block.text, link_resolver)}</blockquote>")
        elif isinstance(block, Rule):
            out.append("<hr>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Helpers used by tests and tools
# ---------------------------------------------------------------------------
def extract_code_blocks(text: str, executable_only: bool = True) -> list[CodeBlock]:
    blocks = [b for b in parse(text) if isinstance(b, CodeBlock)]
    return [b for b in blocks if b.executable] if executable_only else blocks


def first_heading(text: str) -> str | None:
    for block in parse(text):
        if isinstance(block, Heading):
            return inline_to_plain(block.text)
    return None


def headings(text: str, max_level: int = 6) -> list[Heading]:
    return [b for b in parse(text) if isinstance(b, Heading) and b.level <= max_level]
