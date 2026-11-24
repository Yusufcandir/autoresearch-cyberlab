"""Markdown manuscript -> LaTeX (FR-009) in a two-column journal layout (IEEEtran, the usual
class for network-security venues; it compiles on Overleaf and TeX Live without extra files).

Covers exactly what the builder emits: headings, paragraphs, bullet and numbered lists, pipe
tables with their 'Table n.' captions, figures with their 'Figure n.' captions, **bold**,
`code`, block quotes and [@key] citations (rendered as \\cite{key} against the exported BibTeX
file). 'Table 3' / 'Figure 2' in the text become \\ref links to the right float."""

from __future__ import annotations

import re

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_UNICODE = {
    "±": r"$\pm$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "—": "---",
    "–": "--",
    "“": "``",
    "”": "''",
    "’": "'",
    "‘": "`",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
    "≈": r"$\approx$",
}
# sections that are not part of a journal article body
_SKIP_SECTIONS = {"References", "Evidence"}
_WIDE_FIGURE = re.compile(
    r"per model and dataset|per class|confusion matrix|trained on the row dataset", re.I
)


def _escape(text: str) -> str:
    out = "".join(_SPECIAL.get(ch, ch) for ch in text)
    for k, v in _UNICODE.items():
        out = out.replace(k, v)
    return out


def _inline(text: str) -> str:
    """Escape text but keep citations, bold, code and table/figure references."""
    parts = re.split(r"(\[@[^\]]+\]|\*\*[^*]+\*\*|`[^`]+`|\b(?:Table|Figure|Fig\.) \d+\b)", text)
    out = []
    for p in parts:
        if p.startswith("[@"):
            keys = [k.strip().lstrip("@") for k in p[1:-1].split(";")]
            out.append(r"\cite{" + ",".join(keys) + "}")
        elif p.startswith("**") and p.endswith("**") and len(p) > 4:
            out.append(r"\textbf{" + _inline(p[2:-2]) + "}")
        elif p.startswith("`") and p.endswith("`") and len(p) > 2:
            out.append(r"\texttt{" + _escape(p[1:-1]) + "}")
        elif m := re.fullmatch(r"(Table|Figure|Fig\.) (\d+)", p):
            kind = "tab" if m.group(1) == "Table" else "fig"
            word = "Table" if kind == "tab" else "Fig."
            out.append(f"{word}~\\ref{{{kind}:{m.group(2)}}}")
        else:
            out.append(_escape(p))
    return "".join(out)


def _keywords(title: str, body: str) -> list[str]:
    """Index terms from the paper's own words (no model call)."""
    text = f"{title} {body}".lower()
    pool = [
        ("intrusion detection", "Network intrusion detection"),
        ("netflow", "NetFlow"),
        ("tabular", "Tabular deep learning"),
        ("gradient boost", "Gradient boosting"),
        ("xgboost", "Gradient boosting"),
        ("random forest", "Random forest"),
        ("transformer", "Transformers"),
        ("benchmark", "Benchmarking"),
        ("friedman", "Statistical comparison of classifiers"),
        ("imbalance", "Class imbalance"),
        ("leakage", "Data leakage"),
        ("duplicate", "Data leakage"),
    ]
    out: list[str] = []
    for needle, kw in pool:
        if needle in text and kw not in out:
            out.append(kw)
    return out[:6] or ["Machine learning", "Cybersecurity"]


def _table(rows: list[list[str]], caption: str, number: str | None) -> list[str]:
    ncol = len(rows[0])
    wide = ncol > 4 or sum(len(c) for c in rows[0]) > 60
    env = "table*" if wide else "table"
    numeric = [
        all(re.match(r"^[\s*<>=+\-–\[(]*\d", r[j]) or r[j] in ("-", "–", "n/a") for r in rows[1:])
        and len(rows) > 1
        for j in range(ncol)
    ]
    spec = "".join("c" if numeric[j] else "l" for j in range(ncol))
    out = [rf"\begin{{{env}}}[!t]", r"\centering", r"\footnotesize"]
    if caption:
        out.append(r"\caption{" + _inline(caption) + "}")
    if number:
        out.append(rf"\label{{tab:{number}}}")
    # shrink a table that is too wide, never enlarge a small one
    out.append(r"\begin{adjustbox}{max width=" + (r"\textwidth" if wide else r"\columnwidth") + "}")
    out.append(r"\begin{tabular}{" + spec + "}")
    out.append(r"\toprule")
    out.append(" & ".join(r"\textbf{" + _inline(c) + "}" for c in rows[0]) + r" \\")
    out.append(r"\midrule")
    for r in rows[1:]:
        out.append(" & ".join(_inline(c) for c in r[:ncol]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", rf"\end{{{env}}}"]
    return out


def markdown_to_latex(markdown: str, bib_name: str = "references") -> str:
    lines = markdown.split("\n")
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "Manuscript")
    body: list[str] = []
    abstract: list[str] = []
    section = ""
    i = 0
    in_list: str | None = None
    pending_caption: tuple[str, str] | None = None  # ('3', 'Main results ...')

    def emit(text: str) -> None:
        (abstract if section == "Abstract" else body).append(text)

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            emit(r"\end{" + in_list + "}")
            in_list = None

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            i += 1
            continue
        if ln.startswith("## "):
            close_list()
            section = ln[3:].strip()
            if section in _SKIP_SECTIONS:
                # skip to the next top-level section (the bibliography comes from BibTeX)
                i += 1
                while i < len(lines) and not lines[i].startswith("## "):
                    i += 1
                continue
            if section != "Abstract":
                body.append(r"\section{" + _inline(section) + "}")
        elif ln.startswith("### "):
            close_list()
            sub = ln[4:].strip()
            # a subsection holding only tables or figures would be an empty heading in print
            # (floats move); the captions carry the information
            j = i + 1
            only_floats = True
            while j < len(lines) and not lines[j].startswith("#"):
                t = lines[j].strip()
                if t and not t.startswith(("|", "![", "*Table", "*Figure")):
                    only_floats = False
                    break
                j += 1
            if not only_floats:
                body.append(r"\subsection{" + _inline(sub) + "}")
        elif (cap := re.match(r"^\*Table (\d+)\.\s*(.*)\*\s*$", ln.strip())) is not None:
            pending_caption = (cap.group(1), cap.group(2))
        elif (img := re.match(r"^!\[[^\]]*\]\(([^)\s]+)\)", ln)) is not None:
            close_list()
            path = img.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            caption, number = "", None
            cap = (
                re.match(r"^\*Figure (\d+)\.\s*(.*)\*$", lines[j].strip())
                if j < len(lines)
                else None
            )
            if cap:
                number, caption, i = cap.group(1), cap.group(2), j
            if path.startswith("data:"):
                body.append("% figure omitted: export the LaTeX bundle to get the image files")
            else:
                wide = bool(_WIDE_FIGURE.search(caption))
                env = "figure*" if wide else "figure"
                body += [
                    rf"\begin{{{env}}}[!t]",
                    r"\centering",
                    # figures are drawn at their printed size: never scale them up
                    r"\includegraphics[max width="
                    + (r"\textwidth" if wide else r"\columnwidth")
                    + "]{"
                    + path
                    + "}",
                    *([r"\caption{" + _inline(caption) + "}"] if caption else []),
                    *([rf"\label{{fig:{number}}}"] if number else []),
                    rf"\end{{{env}}}",
                ]
        elif ln.startswith("|"):
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                cap_text, cap_num = (
                    (pending_caption[1], pending_caption[0]) if pending_caption else ("", None)
                )
                body += _table(rows, cap_text, cap_num)
            pending_caption = None
            continue
        elif re.match(r"^\s*[-*+]\s", ln):
            if in_list != "itemize":
                close_list()
                emit(r"\begin{itemize}")
                in_list = "itemize"
            emit(r"\item " + _inline(re.sub(r"^\s*[-*+]\s+", "", ln)))
        elif re.match(r"^\s*\d+\.\s", ln):
            if in_list != "enumerate":
                close_list()
                emit(r"\begin{enumerate}")
                in_list = "enumerate"
            emit(r"\item " + _inline(re.sub(r"^\s*\d+\.\s+", "", ln)))
        elif ln.startswith(">"):
            close_list()
            if not ln.lstrip("> ").startswith("DRAFT"):  # the dashboard draft banner
                emit(r"\begin{quote}\itshape " + _inline(ln.lstrip("> ")) + r"\end{quote}")
        elif not ln.strip():
            close_list()
            emit("")
        else:
            close_list()
            emit(_inline(ln.rstrip()))
        i += 1
    close_list()
    keywords = _keywords(title, "\n".join(lines))
    return "\n".join(
        [
            r"\documentclass[journal]{IEEEtran}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{graphicx}",
            r"\usepackage[export]{adjustbox}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\usepackage{cite}",
            r"\usepackage{url}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\title{" + _escape(title) + "}",
            r"\author{Author~Names% fill in the authors and affiliations",
            r"\thanks{Affiliation and correspondence.}}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            *[a for a in abstract if a.strip()],
            r"\end{abstract}",
            r"\begin{IEEEkeywords}",
            ", ".join(keywords) + ".",
            r"\end{IEEEkeywords}",
            *body,
            r"\bibliographystyle{IEEEtran}",
            r"\bibliography{" + bib_name + "}",
            r"\end{document}",
            "",
        ]
    )
