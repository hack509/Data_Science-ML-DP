"""Export the Markdown lessons to PDF, fully offline.

Conversion pipeline: Markdown -> lab.markdown_lite blocks -> ReportLab
flowables -> PDF. Fonts are the DejaVu TrueType fonts shipped with
matplotlib, so French accents, Greek letters and common math symbols
(e.g. "é", "θ", "Σ", "√", "≤") are rendered without downloading anything.

Features: one PDF per module, optional complete course PDF, table of
contents, page numbers, code blocks with preserved indentation and wrapped
long lines (marked with "↪"), tables, lists and quotes.

Known limitations (documented in docs/offline_usage.md):
* images referenced in Markdown are not embedded (lessons save figures to
  outputs/ and describe them in text instead);
* nested lists deeper than one level are flattened;
* no syntax highlighting inside code blocks.

Usage (from the repository root):
    python scripts/export_lessons_to_pdf.py                 # one PDF per module
    python scripts/export_lessons_to_pdf.py --module 05     # a single module
    python scripts/export_lessons_to_pdf.py --all-in-one    # + complete course
    python scripts/export_lessons_to_pdf.py --with-corrections
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import markdown_lite as md  # noqa: E402
from lab.catalog import Module, list_modules  # noqa: E402
from lab.paths import EXPORTS_DIR  # noqa: E402

FONT_FILES = {
    "DejaVuSans": "DejaVuSans.ttf",
    "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
    "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    "DejaVuSansMono": "DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold": "DejaVuSansMono-Bold.ttf",
}
CODE_FONT_SIZE = 7.6
CONTINUATION = "↪ "


def find_fonts() -> dict[str, Path]:
    """Locate the DejaVu fonts bundled with matplotlib (no download)."""
    import matplotlib

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = {name: font_dir / filename for name, filename in FONT_FILES.items()}
    missing = [str(p) for p in fonts.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"DejaVu fonts not found: {missing}. Reinstall matplotlib.")
    return fonts


def register_fonts() -> None:
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if "DejaVuSans" in pdfmetrics.getRegisteredFontNames():
        return
    for name, path in find_fonts().items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    addMapping("DejaVuSans", 0, 0, "DejaVuSans")
    addMapping("DejaVuSans", 1, 0, "DejaVuSans-Bold")
    addMapping("DejaVuSans", 0, 1, "DejaVuSans-Oblique")
    addMapping("DejaVuSans", 1, 1, "DejaVuSans-BoldOblique")
    addMapping("DejaVuSansMono", 0, 0, "DejaVuSansMono")
    addMapping("DejaVuSansMono", 1, 0, "DejaVuSansMono-Bold")
    addMapping("DejaVuSansMono", 0, 1, "DejaVuSansMono")
    addMapping("DejaVuSansMono", 1, 1, "DejaVuSansMono-Bold")


def build_styles():
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    base = ParagraphStyle("body", fontName="DejaVuSans", fontSize=9.5, leading=13.5, spaceAfter=5)
    styles = {
        "body": base,
        "h1": ParagraphStyle("h1", parent=base, fontName="DejaVuSans-Bold", fontSize=18, leading=23, spaceBefore=6, spaceAfter=10, textColor=colors.HexColor("#1f3a5f")),
        "h2": ParagraphStyle("h2", parent=base, fontName="DejaVuSans-Bold", fontSize=13.5, leading=18, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1f3a5f")),
        "h3": ParagraphStyle("h3", parent=base, fontName="DejaVuSans-Bold", fontSize=11, leading=15, spaceBefore=9, spaceAfter=4),
        "h4": ParagraphStyle("h4", parent=base, fontName="DejaVuSans-Bold", fontSize=10, leading=14, spaceBefore=6, spaceAfter=3),
        "bullet": ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=4, spaceAfter=2),
        "bullet2": ParagraphStyle("bullet2", parent=base, leftIndent=28, bulletIndent=18, spaceAfter=2),
        "quote": ParagraphStyle("quote", parent=base, leftIndent=10, borderPadding=(4, 4, 4, 6), backColor=colors.HexColor("#eef4fb"), borderColor=colors.HexColor("#7aa6d6"), borderWidth=0.5, spaceBefore=4, spaceAfter=8),
        "code": ParagraphStyle("code", fontName="DejaVuSansMono", fontSize=CODE_FONT_SIZE, leading=CODE_FONT_SIZE * 1.3, backColor=colors.HexColor("#f4f4f4"), borderColor=colors.HexColor("#d0d0d0"), borderWidth=0.4, borderPadding=4, spaceBefore=4, spaceAfter=9),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.2, leading=10.5, spaceAfter=0),
        "toc_title": ParagraphStyle("toc_title", parent=base, fontName="DejaVuSans-Bold", fontSize=16, leading=20, spaceAfter=12),
        "cover": ParagraphStyle("cover", parent=base, fontName="DejaVuSans-Bold", fontSize=24, leading=30, spaceAfter=16, textColor=colors.HexColor("#1f3a5f")),
    }
    return styles


def inline_markup(text: str) -> str:
    """Convert inline Markdown to ReportLab's mini-HTML."""
    out = []
    for kind, value, extra in md.iter_inline(text):
        if kind == "text":
            out.append(escape(value))
        elif kind == "code":
            out.append(f'<font name="DejaVuSansMono" backColor="#efefef">{escape(value)}</font>')
        elif kind == "bold":
            out.append(f"<b>{inline_markup(value)}</b>")
        elif kind == "italic":
            out.append(f"<i>{inline_markup(value)}</i>")
        else:  # link: keep the label, show external URLs in small text
            label = inline_markup(value)
            if extra.startswith(("http://", "https://")):
                out.append(f'<u>{label}</u> <font size="7">({escape(extra)})</font>')
            else:
                out.append(f"<u>{label}</u>")
    return "".join(out)


def wrap_code(code: str, max_chars: int) -> str:
    """Wrap long code lines without losing indentation."""
    wrapped = []
    for line in code.split("\n"):
        if len(line) <= max_chars:
            wrapped.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        pieces = textwrap.wrap(
            line.strip(), width=max_chars - indent - len(CONTINUATION),
            break_long_words=True, break_on_hyphens=False, drop_whitespace=False,
        )
        wrapped.append(" " * indent + pieces[0])
        wrapped.extend(" " * indent + CONTINUATION + piece for piece in pieces[1:])
    return "\n".join(wrapped)


def blocks_to_flowables(blocks, styles, frame_width: float, toc_level_offset: int = 0):
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable, Paragraph, Preformatted, Spacer, Table, TableStyle

    char_width = CODE_FONT_SIZE * 0.602  # DejaVu Sans Mono advance width ratio
    max_chars = int((frame_width - 12) / char_width)
    flowables = []
    for block in blocks:
        if isinstance(block, md.Heading):
            level = min(block.level, 4)
            paragraph = Paragraph(inline_markup(block.text), styles[f"h{level}"])
            paragraph.toc_level = block.level - 1 + toc_level_offset
            paragraph.toc_text = md.inline_to_plain(block.text)
            flowables.append(paragraph)
        elif isinstance(block, md.Paragraph):
            flowables.append(Paragraph(inline_markup(block.text), styles["body"]))
        elif isinstance(block, md.CodeBlock):
            flowables.append(Preformatted(wrap_code(block.code, max_chars), styles["code"]))
        elif isinstance(block, md.ListBlock):
            for number, item in enumerate(block.items, start=1):
                bullet = f"{number}." if block.ordered else "•"
                flowables.append(Paragraph(inline_markup(item.text), styles["bullet"], bulletText=bullet))
                for child in item.children:
                    flowables.append(Paragraph(inline_markup(child.text), styles["bullet2"], bulletText="–"))
            flowables.append(Spacer(1, 4))
        elif isinstance(block, md.Table):
            columns = max(len(block.header), *(len(r) for r in block.rows)) if block.rows else len(block.header)
            data = [[Paragraph(f"<b>{inline_markup(c)}</b>", styles["cell"]) for c in block.header]]
            for row in block.rows:
                row = row + [""] * (columns - len(row))
                data.append([Paragraph(inline_markup(c), styles["cell"]) for c in row])
            table = Table(data, colWidths=[frame_width / columns] * columns, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ebf5")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            flowables.extend([table, Spacer(1, 8)])
        elif isinstance(block, md.Quote):
            flowables.append(Paragraph(inline_markup(block.text), styles["quote"]))
        elif isinstance(block, md.Rule):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999"), spaceBefore=4, spaceAfter=6))
    return flowables


def _make_doc_template(output: Path, title: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    class LessonDocTemplate(BaseDocTemplate):
        def afterFlowable(self, flowable):
            level = getattr(flowable, "toc_level", None)
            if level is not None and level <= 2:
                key = f"h{id(flowable)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(flowable.toc_text[:90], key, level=level, closed=level > 0)
                self.notify("TOCEntry", (level, flowable.toc_text, self.page, key))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVuSans", 7.5)
        canvas.drawString(2 * cm, 1.2 * cm, title[:90])
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = LessonDocTemplate(
        str(output), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
        bottomMargin=2 * cm, title=title, author="Offline AI Learning Lab",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=footer)])
    return doc


def _toc(styles):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus.tableofcontents import TableOfContents

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="DejaVuSans-Bold", fontSize=10.5, leading=14, leftIndent=0),
        ParagraphStyle("toc1", fontName="DejaVuSans", fontSize=9.5, leading=12.5, leftIndent=14),
        ParagraphStyle("toc2", fontName="DejaVuSans", fontSize=8.5, leading=11, leftIndent=28),
    ]
    toc.dotsMinLevel = 0
    return toc


def module_story(module: Module, styles, frame_width: float, with_corrections: bool, toc_level_offset: int = 0):
    from reportlab.platypus import PageBreak

    story = []
    sources = [module.readme] + [lesson.path for lesson in module.lessons]
    if with_corrections and module.corrections.exists():
        sources.append(module.corrections)
    for index, path in enumerate(sources):
        if index > 0:
            story.append(PageBreak())
        blocks = md.parse(path.read_text(encoding="utf-8"))
        # Lessons start with a level-1 heading; nest them under the module in the TOC.
        offset = toc_level_offset + (0 if index == 0 else 1)
        story.extend(blocks_to_flowables(blocks, styles, frame_width, offset))
    return story


def export_module(module: Module, output_dir: Path, with_corrections: bool = False) -> Path:
    from reportlab.platypus import PageBreak, Paragraph

    register_fonts()
    styles = build_styles()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{module.id}.pdf"
    doc = _make_doc_template(output, module.title)
    story = [Paragraph("Sommaire", styles["toc_title"]), _toc(styles), PageBreak()]
    story += module_story(module, styles, doc.width, with_corrections)
    doc.multiBuild(story)
    return output


def export_course(modules: list[Module], output_dir: Path, with_corrections: bool = False) -> Path:
    from reportlab.platypus import PageBreak, Paragraph, Spacer

    register_fonts()
    styles = build_styles()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "cours_complet.pdf"
    doc = _make_doc_template(output, "Offline AI Learning Lab - cours complet")
    story = [
        Spacer(1, 120),
        Paragraph("Offline AI Learning Lab", styles["cover"]),
        Paragraph("Cours complet : Python, mathématiques, données, Machine Learning, Deep Learning, NLP, vision et projets.", styles["body"]),
        PageBreak(),
        Paragraph("Sommaire", styles["toc_title"]),
        _toc(styles),
    ]
    for module in modules:
        story.append(PageBreak())
        story += module_story(module, styles, doc.width, with_corrections)
    doc.multiBuild(story)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", help="module number or id prefix, e.g. 05")
    parser.add_argument("--all-in-one", action="store_true", help="also build the complete course PDF")
    parser.add_argument("--with-corrections", action="store_true", help="append quiz answers and corrections")
    parser.add_argument("--output-dir", type=Path, default=EXPORTS_DIR / "pdf")
    args = parser.parse_args(argv)

    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("reportlab is not installed. During the online installation phase run:")
        print("    python -m pip install -r requirements.txt")
        return 1

    modules = list_modules()
    if args.module:
        modules = [m for m in modules if m.id.startswith(args.module)]
        if not modules:
            print(f"No module matches {args.module!r}")
            return 1
    for module in modules:
        path = export_module(module, args.output_dir, args.with_corrections)
        print(f"[OK] {path}")
    if args.all_in_one:
        path = export_course(modules, args.output_dir, args.with_corrections)
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
