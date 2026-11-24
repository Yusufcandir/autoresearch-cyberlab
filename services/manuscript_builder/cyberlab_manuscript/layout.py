"""Reading layout of a manuscript (used by every rendering: dashboard view, print/PDF page,
LaTeX and Word export). The stored Markdown keeps tables and figures in fixed sub-sections so
the evidence checks can find them; a reader expects each table and figure right after the
paragraph that first cites it, as in a journal article."""

from __future__ import annotations

import re

_TABLE_CAP = re.compile(r"^\*Table (\d+)\.")
_FIG_IMG = re.compile(r"^!\[Figure (\d+)\]")
_REF = re.compile(r"\b(Table|Figure) (\d+)\b")


def _blocks(body: str) -> list[str]:
    return [b for b in re.split(r"\n[ \t]*\n", body.strip("\n")) if b.strip()]


def _section_floats(body: str) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    """Items ('h' heading | 'p' text) and the floats keyed by ('Table'|'Figure', number)."""
    blocks = _blocks(body)
    items: list[tuple[str, str]] = []
    floats: dict[tuple[str, str], str] = {}
    i = 0
    while i < len(blocks):
        b = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else ""
        if (m := _TABLE_CAP.match(b)) and nxt.startswith("|"):
            floats[("Table", m.group(1))] = b + "\n\n" + nxt
            i += 2
            continue
        if (m := _FIG_IMG.match(b)) and nxt.startswith(f"*Figure {m.group(1)}."):
            floats[("Figure", m.group(1))] = b + "\n\n" + nxt
            i += 2
            continue
        items.append(("h" if b.startswith("#") else "p", b))
        i += 1
    return items, floats


def place_floats(markdown: str) -> str:
    """Move each table and figure to just after the paragraph that first cites it ('Table 3',
    'Figure 2'); sub-section headings left with nothing under them are dropped; a float that
    is never cited stays at the end of its section."""
    parts = re.split(r"(?m)^(## .+)$", markdown)
    for i in range(1, len(parts) - 1, 2):
        items, floats = _section_floats(parts[i + 1])
        if not floats:
            continue
        # a heading directly followed by another heading (or nothing) lost its content
        kept = [
            (k, t)
            for j, (k, t) in enumerate(items)
            if not (
                k == "h" and t.startswith("###") and (j + 1 == len(items) or items[j + 1][0] == "h")
            )
        ]
        out: list[str] = []
        placed: set[tuple[str, str]] = set()
        for kind, text in kept:
            out.append(text)
            if kind != "p":
                continue
            for m in _REF.finditer(text):
                key = (m.group(1), m.group(2))
                if key in floats and key not in placed:
                    out.append(floats[key])
                    placed.add(key)
        out += [f for k, f in floats.items() if k not in placed]
        parts[i + 1] = "\n\n" + "\n\n".join(out) + "\n\n"
    return "".join(parts)


# ----------------------------------------------------------------- shared journal conventions

ROMAN = [
    "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII",
    "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
]  # fmt: skip
WIDE_FIGURE = re.compile(
    r"per model and dataset|per class|confusion matrix|trained on the row dataset", re.I
)
_REF_LINE = re.compile(r"^- \[([^\]]+)\] (.*?) \((\d{4})\)\. (.*?)\. (.*?)\. doi:(\S+)\s*$")


def roman(n: int) -> str:
    return ROMAN[n] if 0 <= n < len(ROMAN) else str(n)


def journal_refs(text: str) -> str:
    """'Table 3' -> 'Table III' and 'Figure 2' -> 'Fig. 2' (IEEE style)."""
    text = re.sub(r"\bTable (\d+)\b", lambda m: f"Table {roman(int(m.group(1)))}", text)
    return re.sub(r"\bFigure (\d+)\b", r"Fig. \1", text)


def ieee_authors(authors: str) -> str:
    names = [a.strip() for a in authors.split(",") if a.strip()]

    def short(name: str) -> str:
        parts = name.split()
        if len(parts) < 2:
            return name
        return " ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1]

    names = [short(n) for n in names]
    if not names:
        return ""
    if len(names) > 3:
        return names[0] + " et al."
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]


def split_sections(markdown: str) -> tuple[str, dict[str, str], list[str]]:
    title = next((ln[2:].strip() for ln in markdown.splitlines() if ln.startswith("# ")), "")
    parts = re.split(r"(?m)^## (.+)$", markdown)
    sections: dict[str, str] = {}
    order: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip()] = parts[i + 1]
        order.append(parts[i].strip())
    return title, sections, order


def references(markdown: str) -> dict[str, tuple[str, ...]]:
    """key -> (authors, year, title, venue, doi) from the References list."""
    _, sections, _ = split_sections(markdown)
    out: dict[str, tuple[str, ...]] = {}
    for line in sections.get("References", "").splitlines():
        if m := _REF_LINE.match(line.strip()):
            out[m.group(1)] = m.groups()[1:]
    return out


def citation_numbers(body: str, known: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """Reference numbers in order of first citation (IEEE)."""
    numbers: dict[str, int] = {}
    for key in re.findall(r"@([A-Za-z0-9_]+)", " ".join(re.findall(r"\[@[^\]]+\]", body))):
        if key in known and key not in numbers:
            numbers[key] = len(numbers) + 1
    return numbers


def ieee_reference(ref: tuple[str, ...]) -> tuple[str, str, str]:
    """(before the venue, venue, after the venue) so the venue can be set in italics."""
    authors, year, title, venue, doi = ref
    return f"{ieee_authors(authors)}, “{title},” ", venue, f", {year}, doi: {doi}."


def png_width_inches(data_uri: str, dpi: int = 300) -> float | None:
    """Printed width of an embedded figure (the figures are drawn at 300 dpi), so a page shows
    it at its real size instead of one screen pixel per image pixel."""
    import base64
    import struct

    b64 = data_uri.split("base64,", 1)[-1]
    try:
        head = base64.b64decode(b64[:64] + "=" * (-len(b64[:64]) % 4))
        width = struct.unpack(">I", head[16:20])[0]
    except Exception:  # noqa: BLE001 - not a PNG: leave the size to the page
        return None
    return round(width / dpi, 2) if head[:8] == b"\x89PNG\r\n\x1a\n" else None
