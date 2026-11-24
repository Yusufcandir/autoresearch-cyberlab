"""Word (.docx) export of a manuscript in the same two-column journal layout as the PDF page
and the LaTeX export (spec Step 21: Word/PDF export where practical).

Title, authors and abstract across the page; body in two columns with numbered sections
(I., II., ... and A., B., ...); each table and figure right after the paragraph that cites it,
'TABLE III' captions above tables and 'Fig. 1.' captions below figures; wide tables and figures
span both columns; citations numbered in order of first use with an IEEE-style reference list.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

from cyberlab_manuscript.layout import (
    WIDE_FIGURE,
    citation_numbers,
    ieee_reference,
    journal_refs,
    place_floats,
    references,
    roman,
    split_sections,
)

_IMG = re.compile(r"^!\[[^\]]*\]\(data:image/png;base64,(?P<b64>[A-Za-z0-9+/=]+)\)\s*$")
_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
FONT = "Times New Roman"
COLUMN_IN = 3.45  # one column of an A4 two-column page
FULL_IN = 7.1


def _runs(par: Any, text: str, size: float | None = None, bold: bool = False) -> None:
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1])
            r.italic = True
        else:
            r = par.add_run(part)
        if bold:
            r.bold = True
        if size:
            r.font.size = Pt(size)


def _columns(section: Any, n: int) -> None:
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(0.22 * 1440)))


def _new_section(doc: Any, n_cols: int) -> None:
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    _columns(sec, n_cols)


def _border(cell: Any, edge: str, size: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:color"), "222222")
    borders.append(el)


def _para(
    doc: Any,
    text: str = "",
    *,
    size: float = 10,
    align: Any = None,
    indent: bool = False,
    space_after: float = 0,
) -> Any:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if indent:
        pf.first_line_indent = Pt(10)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align is None else align
    if text:
        _runs(p, text, size)
    return p


def _rows(block: str) -> list[list[str]]:
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in block.splitlines()
        if ln.strip().startswith("|")
    ]
    return [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]


def _table(doc: Any, number: str | None, caption: str, rows: list[list[str]]) -> None:
    if number:
        cap = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
        cap.paragraph_format.space_before = Pt(6)
        cap.paragraph_format.keep_with_next = True
        r = cap.add_run(f"TABLE {roman(int(number))}\n")
        r.font.size = Pt(8)
        r2 = cap.add_run(caption)
        r2.font.size = Pt(8)
        r2.font.small_caps = True
    ncol = max(len(x) for x in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, cells in enumerate(rows):
        for ci in range(ncol):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            _runs(p, cells[ci] if ci < len(cells) else "", 7.5, bold=ri == 0)
            if ri == 0:
                _border(cell, "top", 10)
                _border(cell, "bottom", 4)
            if ri == len(rows) - 1:
                _border(cell, "bottom", 10)
    _para(doc, space_after=4)


def _figure(doc: Any, b64: str, number: str, caption: str, wide: bool) -> None:
    data = base64.b64decode(b64)
    width = FULL_IN if wide else COLUMN_IN
    try:  # never enlarge a figure beyond its drawn size (300 dpi)
        from PIL import Image

        px = Image.open(io.BytesIO(data)).size[0]
        width = min(width, px / 300)
    except Exception:  # noqa: BLE001, S110 - size probing is optional
        pass
    p = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(io.BytesIO(data), width=Inches(width))
    cap = _para(doc, space_after=6)
    r = cap.add_run(f"Fig. {number}. ")
    r.bold = True
    r.font.size = Pt(8)
    _runs(cap, caption, 8)


def markdown_to_docx(markdown: str) -> bytes:
    from cyberlab_manuscript.latex import _keywords

    markdown = place_floats(markdown)
    title, sections, order = split_sections(markdown)
    refs = references(markdown)
    body_names = [n for n in order if n not in ("Abstract", "References", "Evidence")]
    body_md = "\n".join(f"## {n}\n{sections[n]}" for n in body_names)
    numbers = citation_numbers(body_md, refs)

    def cites(text: str) -> str:
        def one(m: re.Match[str]) -> str:
            keys = [k for k in re.findall(r"@([A-Za-z0-9_]+)", m.group(0)) if k in numbers]
            return "[" + ", ".join(str(numbers[k]) for k in keys) + "]" if keys else ""

        return journal_refs(re.sub(r"\[@[A-Za-z0-9_]+(?:;\s*@[A-Za-z0-9_]+)*\]", one, text))

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(1.9)
    sec.left_margin = sec.right_margin = Cm(1.6)

    # ---- front matter, one column
    if title:
        t = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
        r = t.add_run(title)
        r.font.size = Pt(22)
        _para(doc, "Author Names", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
        aff = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
        ra = aff.add_run("Affiliation, City, Country · corresponding author e-mail")
        ra.italic = True
        ra.font.size = Pt(9)
    if sections.get("Abstract", "").strip():
        ab = _para(doc, space_after=3)
        lead = ab.add_run("Abstract—")
        lead.bold = lead.italic = True
        lead.font.size = Pt(9)
        _runs(ab, " ".join(cites(sections["Abstract"]).split()), 9, bold=True)
        kw = _para(doc, space_after=8)
        k = kw.add_run("Index Terms—")
        k.bold = k.italic = True
        k.font.size = Pt(9)
        _runs(kw, ", ".join(_keywords(title, markdown)) + ".", 9, bold=True)

    # ---- body, two columns
    _new_section(doc, 2)
    cols = 2

    def span(n: int) -> None:
        nonlocal cols
        if cols != n:
            _new_section(doc, n)
            cols = n

    for n_sec, name in enumerate(body_names, 1):
        n_sub = 0
        h = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.keep_with_next = True
        hr = h.add_run(f"{roman(n_sec)}. {name}")
        hr.font.small_caps = True
        blocks = [b for b in re.split(r"\n[ \t]*\n", sections[name].strip("\n")) if b.strip()]
        first = True
        i = 0
        while i < len(blocks):
            b = blocks[i].strip()
            nxt = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
            if b.startswith("### "):
                n_sub += 1
                sh = _para(doc, space_after=2)
                sh.paragraph_format.space_before = Pt(5)
                sh.paragraph_format.keep_with_next = True
                sr = sh.add_run(f"{chr(64 + n_sub)}. {b[4:].strip()}")
                sr.italic = True
                first = True
                i += 1
                continue
            tm = re.match(r"^\*Table (\d+)\.\s*(.*)\*$", b)
            if tm and nxt.startswith("|"):
                rows = _rows(nxt)
                wide = len(rows[0]) > 4
                if wide:
                    span(1)
                _table(doc, tm.group(1), tm.group(2), rows)
                span(2)
                i, first = i + 2, True
                continue
            im = _IMG.match(b)
            fm = re.match(r"^\*Figure (\d+)\.\s*(.*)\*$", nxt)
            if im and fm:
                wide = bool(WIDE_FIGURE.search(fm.group(2)))
                if wide:
                    span(1)
                _figure(doc, im.group("b64"), fm.group(1), fm.group(2), wide)
                span(2)
                i, first = i + 2, True
                continue
            if b.startswith("|"):  # a table without a caption
                if rows := _rows(b):
                    _table(doc, None, "", rows)
                i += 1
                continue
            for ln in b.splitlines():
                item = re.match(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$", ln)
                if item:
                    p = _para(doc, align=WD_ALIGN_PARAGRAPH.LEFT)
                    p.paragraph_format.left_indent = Pt(12)
                    p.paragraph_format.first_line_indent = Pt(-8)
                    _runs(p, "• " + cites(item.group(1)), 10)
                elif not ln.startswith(">"):  # '>' is the dashboard's draft banner
                    _para(doc, cites(ln.strip()), indent=not first)
                    first = False
            i += 1

    # ---- references
    if numbers:
        h = _para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
        h.paragraph_format.space_before = Pt(8)
        h.add_run("References").font.small_caps = True
    for key, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        before, venue, after = ieee_reference(refs[key])
        p = _para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=1)
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.first_line_indent = Pt(-18)
        for txt, italic in ((f"[{n}]\t" + before, False), (venue, True), (after, False)):
            run = p.add_run(txt)
            run.italic = italic
            run.font.size = Pt(8)
    shas = re.findall(r"SHA-256: `([0-9a-f]+)`", sections.get("Evidence", ""))
    if shas:
        note = _para(doc)
        note.paragraph_format.space_before = Pt(6)
        nb = note.add_run("Reproducibility. ")
        nb.bold = True
        nb.font.size = Pt(8)
        _runs(
            note,
            "All numbers, tables and figures were generated from stored experiment records "
            f"(analysis artifact SHA-256 {shas[0][:16]}…).",
            8,
        )
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
