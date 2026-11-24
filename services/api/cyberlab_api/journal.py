"""Print/PDF page of a manuscript in a two-column journal layout (FR-009: the browser's 'Save
as PDF' produces the PDF). Title block, abstract and index terms across the page; numbered
sections; tables and figures next to the paragraph that cites them, with 'TABLE III' / 'Fig. 1'
captions; citations numbered in order of first use and an IEEE-style reference list."""

from __future__ import annotations

import html as _html
import re

from cyberlab_api.progress import _md
from cyberlab_manuscript.latex import _keywords
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


def journalize_html(out: str) -> str:
    """Captioned journal floats ('TABLE III' above tables, 'Fig. 1.' below figures, wide ones
    marked) and journal-style in-text references, on HTML rendered from the manuscript."""
    out = re.sub(
        r"<p><em>Table (\d+)\. (.*?)</em></p>\s*<table>(.*?)</table>",
        lambda m: (
            f'<figure class="tbl{" wide" if m.group(3).count("<th") > 4 else ""}">'
            f'<figcaption><span class="lbl">TABLE {roman(int(m.group(1)))}</span>'
            f"<br>{m.group(2)}</figcaption><table>{m.group(3)}</table></figure>"
        ),
        out,
        flags=re.S,
    )
    out = re.sub(
        r'<p><img src="([^"]+)" alt="Figure (\d+)"\s*/?></p>\s*<p><em>Figure \d+\. (.*?)</em></p>',
        lambda m: (
            f'<figure class="fig{" wide" if WIDE_FIGURE.search(m.group(3)) else ""}">'
            f'<img src="{m.group(1)}" alt="Figure {m.group(2)}">'
            f"<figcaption><b>Fig. {m.group(2)}.</b> {m.group(3)}</figcaption></figure>"
        ),
        out,
        flags=re.S,
    )
    return journal_refs(out)


def journal_page(markdown: str, version: int | str) -> str:
    markdown = place_floats(markdown)
    title, sections, order = split_sections(markdown)
    refs = references(markdown)
    body_names = [n for n in order if n not in ("Abstract", "References", "Evidence")]
    body_md = "\n".join(f"## {n}\n{sections[n]}" for n in body_names)
    numbers = citation_numbers(body_md, refs)

    def render(md: str) -> str:
        out = _md.render(md)

        def cite(m: re.Match[str]) -> str:
            keys = [k for k in re.findall(r"@([A-Za-z0-9_]+)", m.group(0)) if k in numbers]
            if not keys:
                return ""
            return "[" + ", ".join(f'<a href="#ref-{k}">{numbers[k]}</a>' for k in keys) + "]"

        out = re.sub(r"\[@[A-Za-z0-9_]+(?:;\s*@[A-Za-z0-9_]+)*\]", cite, out)
        return journalize_html(out)

    abstract = render(sections.get("Abstract", "").strip())
    abstract = re.sub(r"^<p>", "<p><b><i>Abstract</i></b>—", abstract, count=1)
    keywords = ", ".join(_keywords(title, markdown))
    ref_items = []
    for key, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        before, venue, after = ieee_reference(refs[key])
        ref_items.append(
            f'<li id="ref-{key}"><span class="n">[{n}]</span> {_html.escape(before)}'
            f"<i>{_html.escape(venue)}</i>{_html.escape(after)}</li>"
        )
    evidence = sections.get("Evidence", "")
    shas = re.findall(r"SHA-256: `([0-9a-f]+)`", evidence)
    repro = (
        '<p class="repro"><b>Reproducibility.</b> All numbers, tables and figures were generated '
        "from stored experiment records"
        + (f" (analysis artifact SHA-256 {shas[0][:16]}…)" if shas else "")
        + ".</p>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{_html.escape(title)} (v{version})</title>
<style>{CSS}</style></head><body>
<div class="bar"><button onclick="window.print()">Save as PDF</button>
<a href="javascript:history.back()">Back</a>
<span>Draft v{version}: prose is AI-written and must be reviewed by the authors.</span></div>
<article class="paper">
<header>
<h1>{_html.escape(title)}</h1>
<p class="authors">Author Names</p>
<p class="affil">Affiliation, City, Country · corresponding author e-mail</p>
</header>
<section class="abstract">{abstract}
<p><b><i>Index Terms</i></b>—{_html.escape(keywords)}.</p></section>
<div class="body">
{render(body_md)}
<h2 class="nonum">References</h2>
<ol class="refs">{"".join(ref_items)}</ol>
{repro}
</div>
</article>
</body></html>"""


CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
:root { --ink: #111; --rule: #222; }
* { box-sizing: border-box; }
body { margin: 0; background: #e9e9e9; color: var(--ink);
  font-family: "Times New Roman", Times, "Nimbus Roman", "STIX Two Text", serif; }
.bar { font: 13px system-ui, sans-serif; padding: 10px 16px; display: flex; gap: 14px;
  align-items: center; background: #fff; border-bottom: 1px solid #ccc; position: sticky; top: 0; }
.bar span { color: #666; }
.paper { background: #fff; max-width: 210mm; margin: 18px auto; padding: 17mm 15mm;
  box-shadow: 0 1px 6px rgba(0,0,0,.15); font-size: 10pt; line-height: 1.32; }
header { text-align: center; margin-bottom: 10pt; }
h1 { font-size: 22pt; font-weight: normal; line-height: 1.15; margin: 0 0 10pt; }
.authors { font-size: 11pt; margin: 0; }
.affil { font-size: 9pt; font-style: italic; margin: 2pt 0 0; color: #333; }
.abstract { font-size: 9pt; font-weight: bold; text-align: justify; margin: 0 0 12pt;
  hyphens: auto; }
.abstract p { margin: 0 0 4pt; }
.body { column-count: 2; column-gap: 5.5mm; text-align: justify; hyphens: auto;
  counter-reset: sec; }
.body p { margin: 0; text-indent: 1em; }
.body h2 + p, .body h3 + p, .body figure + p, .body ul + p, .body ol + p { text-indent: 0; }
.body h2 { counter-increment: sec; counter-reset: sub; font-size: 10pt; font-weight: normal;
  font-variant: small-caps; text-align: center; letter-spacing: .02em; margin: 10pt 0 5pt;
  break-after: avoid; }
.body h2::before { content: counter(sec, upper-roman) ". "; }
.body h2.nonum { counter-increment: none; }
.body h2.nonum::before { content: none; }
.body h3 { counter-increment: sub; font-size: 10pt; font-weight: normal; font-style: italic;
  margin: 7pt 0 3pt; break-after: avoid; }
.body h3::before { content: counter(sub, upper-alpha) ". "; font-style: normal; }
.body ul, .body ol { margin: 3pt 0 3pt 1.2em; padding: 0; }
figure { margin: 8pt 0; break-inside: avoid; text-align: center; }
figure.wide { column-span: all; }
figure img { max-width: 100%; height: auto; }
figure.fig figcaption { font-size: 8pt; text-align: justify; margin-top: 3pt; }
figure.tbl figcaption { font-size: 8pt; font-variant: small-caps; margin-bottom: 4pt;
  text-align: center; }
figure.tbl .lbl { font-variant: normal; letter-spacing: .04em; }
table { border-collapse: collapse; margin: 0 auto; font-size: 7.5pt; line-height: 1.25;
  text-align: left; }
thead th { border-top: 1.2px solid var(--rule); border-bottom: .6px solid var(--rule);
  padding: 2pt 5pt; font-weight: bold; }
tbody tr:last-child td { border-bottom: 1.2px solid var(--rule); }
td { padding: 1.5pt 5pt; }
td:not(:first-child), th:not(:first-child) { text-align: center; }
a { color: inherit; text-decoration: none; }
ol.refs { list-style: none; margin: 0; padding: 0; font-size: 8pt; line-height: 1.25; }
ol.refs li { padding-left: 2.2em; text-indent: -2.2em; margin-bottom: 2pt; text-align: left; }
ol.refs .n { display: inline-block; width: 2.2em; text-indent: 0; }
.repro { font-size: 8pt; margin-top: 6pt !important; text-indent: 0 !important; color: #333; }
code { font-size: 8.5pt; }
@media print {
  body { background: #fff; }
  .bar { display: none; }
  .paper { margin: 0; padding: 0; max-width: none; box-shadow: none; }
}
@media screen and (max-width: 800px) { .body { column-count: 1; } }
"""
