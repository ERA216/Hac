"""Реалистичные форматы: Word-документ, оформленный стилями, и скан без текстового слоя."""

import re
from pathlib import Path

import docx
import pymupdf
import pytest

from revizor.document import load_document
from revizor.rules import run_rules
from revizor.segmentation import rule_segments
from revizor.verify import QuoteLocator

P2P = Path(__file__).resolve().parent.parent / "eval" / "cases" / "p2p" / "requirements.md"


def markdown_to_docx(md_path: Path, out: Path) -> Path:
    """Как оформил бы ТЗ аналитик в Word: стили заголовков, списки, жирные номера, таблицы."""
    d = docx.Document()
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|\s*:?-", lines[i]):
                    rows.append([c.strip().strip("`*") for c in lines[i].strip().strip("|").split("|")])
                i += 1
            table = d.add_table(rows=len(rows), cols=len(rows[0]))
            for r, row in enumerate(rows):
                for c, text in enumerate(row):
                    table.cell(r, c).text = text
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            d.add_heading(m.group(2), level=len(m.group(1)))
        elif line.startswith("* "):
            p = d.add_paragraph(style="List Bullet")
            body = line[2:]
            idm = re.match(r"\*\*(.+?)\*\*\s*(.*)", body)
            if idm:
                p.add_run(idm.group(1)).bold = True
                p.add_run(" " + idm.group(2))
            else:
                p.add_run(body)
        elif line.strip() and line.strip() != "---":
            d.add_paragraph(re.sub(r"\*\*|`", "", line))
        i += 1
    d.save(out)
    return out


def _summary(doc):
    rules = {h.rule for h in run_rules(doc, QuoteLocator(doc))}
    ids = {r.source_id for r in rule_segments(doc) if re.match(r"^[А-ЯA-Z]{1,6}-\d", r.source_id)}
    return rules, ids


def test_word_document_gives_same_findings_as_markdown(tmp_path):
    md_rules, md_ids = _summary(load_document(P2P))
    word = load_document(markdown_to_docx(P2P, tmp_path / "p2p.docx"))
    word_rules, word_ids = _summary(word)
    assert word.title == "Переводы по номеру телефона"
    assert word_ids == md_ids == {"Ф-1", "Ф-2", "Ф-3", "Ф-5", "Ф-6"}
    # ключевые находки правил те же, что у исходного Markdown
    core = {"R-HTTP-GET", "R-API-TYPE", "R-API-QUERY-PII", "R-ID-GAP", "R-ID-REF", "R-RANGE", "R-TEMPLATE", "R-IDEMPOTENCY"}
    assert core <= md_rules and core <= word_rules


def test_scanned_pdf_without_text_layer_is_explained(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 400, 300), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))  # «картинка» без текста
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    with pytest.raises(ValueError, match="скан без текстового слоя"):
        load_document(path)
