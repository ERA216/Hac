"""PDF-версия отчёта: HTML для печати и печать через Chrome/Edge."""

import sys

import pytest

from revizor.report.pdf import export_pdf, find_browser, markdown_to_print_html


def test_print_html_keeps_tables_and_line_breaks_but_no_markup_from_document():
    md = "# Отчёт\n\n| A | B |\n| :--- | :--- |\n| строка 1<br>строка 2 | &lt;script&gt;alert(1)&lt;/script&gt; |\n"
    page = markdown_to_print_html(md, "Отчёт")
    assert "<table>" in page and "строка 1<br>строка 2" in page
    assert "<script>" not in page  # текст документа в отчёте уже экранирован и остаётся текстом
    assert "@page" in page and 'lang="ru"' in page


def test_missing_browser_gives_clear_error(tmp_path, monkeypatch):
    (tmp_path / "r.md").write_text("# Отчёт\n", encoding="utf-8")
    monkeypatch.setattr("revizor.report.pdf.find_browser", lambda: None)
    with pytest.raises(RuntimeError, match="Chrome, Edge или Chromium"):
        export_pdf(tmp_path / "r.md")


@pytest.mark.skipif(sys.platform != "win32" or not find_browser(), reason="печать проверяется на Windows с Chrome/Edge")
def test_pdf_is_printed(tmp_path):
    (tmp_path / "r.md").write_text("# Отчёт\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n", encoding="utf-8")
    pdf = export_pdf(tmp_path / "r.md")
    assert pdf.read_bytes()[:4] == b"%PDF"
