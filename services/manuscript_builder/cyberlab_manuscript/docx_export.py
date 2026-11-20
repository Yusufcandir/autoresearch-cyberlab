"""Word (.docx) export of a manuscript (spec Step 21: Word/PDF export where practical).

Converts the platform's Markdown subset: headings, paragraphs with **bold**/*italic*, bullet
and numbered lists, pipe tables, block quotes and embedded PNG figures (data URIs).
"""

from __future__ import annotations

import base64
import io
import re

from docx import Document
from docx.shared import Inches, Pt

_IMG = re.compile(r"^!\[[^\]]*\]\(data:image/png;base64,(?P<b64>[A-Za-z0-9+/=]+)\)\s*$")
_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def _runs(par: object, text: str) -> None:
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True  # type: ignore[attr-defined]
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])  # type: ignore[attr-defined]
            r.font.name = "Consolas"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True  # type: ignore[attr-defined]
        else:
            par.add_run(part)  # type: ignore[attr-defined]


def markdown_to_docx(markdown: str) -> bytes:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            doc.add_heading(m.group(2).strip(), level=0 if level == 1 else level - 1)
            i += 1
            continue
        img = _IMG.match(line.strip())
        if img:
            doc.add_picture(io.BytesIO(base64.b64decode(img.group("b64"))), width=Inches(6))
            i += 1
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=ncol)
                table.style = "Table Grid"
                for r, cells in enumerate(rows):
                    for c, text in enumerate(cells):
                        cell = table.cell(r, c)
                        cell.text = ""
                        _runs(cell.paragraphs[0], text)
                        if r == 0:
                            for run in cell.paragraphs[0].runs:
                                run.bold = True
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        number = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if bullet or number:
            par = doc.add_paragraph(style="List Bullet" if bullet else "List Number")
            _runs(par, (bullet or number).group(1))  # type: ignore[union-attr]
            i += 1
            continue
        if line.startswith(">"):
            par = doc.add_paragraph()
            run = par.add_run(line.lstrip("> ").strip())
            run.italic = True
            i += 1
            continue
        # a paragraph: join following plain lines
        chunk = [line.strip()]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#|\||>|\s*[-*+]\s|\s*\d+\.\s|!\[)", lines[i])
        ):
            chunk.append(lines[i].strip())
            i += 1
        _runs(doc.add_paragraph(), " ".join(chunk))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
