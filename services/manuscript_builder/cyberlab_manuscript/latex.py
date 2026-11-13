"""Markdown manuscript -> LaTeX (FR-009). Covers exactly what the builder emits:
headings, paragraphs, bullet and numbered lists, pipe tables, **bold**, `code`, block quotes
and [@key] citations (rendered as \\cite{key} against the exported BibTeX file)."""

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
}


def _escape(text: str) -> str:
    out = "".join(_SPECIAL.get(ch, ch) for ch in text)
    for k, v in _UNICODE.items():
        out = out.replace(k, v)
    return out


def _inline(text: str) -> str:
    """Escape text but keep citations, bold and code."""
    parts = re.split(r"(\[@[^\]]+\]|\*\*[^*]+\*\*|`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("[@"):
            keys = [k.strip().lstrip("@") for k in p[1:-1].split(";")]
            out.append(r"\cite{" + ",".join(keys) + "}")
        elif p.startswith("**") and p.endswith("**") and len(p) > 4:
            out.append(r"\textbf{" + _escape(p[2:-2]) + "}")
        elif p.startswith("`") and p.endswith("`") and len(p) > 2:
            out.append(r"\texttt{" + _escape(p[1:-1]) + "}")
        else:
            out.append(_escape(p))
    return "".join(out)


def markdown_to_latex(markdown: str, bib_name: str = "references") -> str:
    lines = markdown.split("\n")
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "Manuscript")
    body: list[str] = []
    i = 0
    in_list: str | None = None

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            body.append(r"\end{" + in_list + "}")
            in_list = None

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            i += 1
            continue
        if ln.startswith("## References"):
            break  # the bibliography comes from BibTeX
        if ln.startswith("## "):
            close_list()
            body.append(r"\section{" + _inline(ln[3:].strip()) + "}")
        elif ln.startswith("### "):
            close_list()
            body.append(r"\subsection{" + _inline(ln[4:].strip()) + "}")
        elif ln.startswith("|"):
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = len(rows[0])
                body.append(r"\begin{table}[h]\centering\small")
                body.append(r"\begin{tabular}{" + "l" * ncol + "}")
                body.append(r"\hline")
                body.append(
                    " & ".join(r"\textbf{" + _inline(c) + "}" for c in rows[0]) + r" \\ \hline"
                )
                for r in rows[1:]:
                    body.append(" & ".join(_inline(c) for c in r[:ncol]) + r" \\")
                body.append(r"\hline\end{tabular}\end{table}")
            continue
        elif re.match(r"^\s*[-*+]\s", ln):
            if in_list != "itemize":
                close_list()
                body.append(r"\begin{itemize}")
                in_list = "itemize"
            body.append(r"\item " + _inline(re.sub(r"^\s*[-*+]\s+", "", ln)))
        elif re.match(r"^\s*\d+\.\s", ln):
            if in_list != "enumerate":
                close_list()
                body.append(r"\begin{enumerate}")
                in_list = "enumerate"
            body.append(r"\item " + _inline(re.sub(r"^\s*\d+\.\s+", "", ln)))
        elif ln.startswith(">"):
            close_list()
            body.append(r"\begin{quote}\itshape " + _inline(ln.lstrip("> ")) + r"\end{quote}")
        elif not ln.strip():
            close_list()
            body.append("")
        else:
            close_list()
            body.append(_inline(ln.rstrip()))
        i += 1
    close_list()
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[margin=2.5cm]{geometry}",
            r"\usepackage{hyperref}",
            r"\title{" + _escape(title) + "}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            *body,
            r"\bibliographystyle{plain}",
            r"\bibliography{" + bib_name + "}",
            r"\end{document}",
            "",
        ]
    )
