"""PDF-версия отчёта: Markdown-отчёт → HTML для печати (A4) → PDF через Chrome, Edge или Chromium без окна.

    python -m revizor.report reports/report.json examples/requirements.md --pdf

Браузер ищется автоматически; путь можно задать переменной окружения CHROME_PATH.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

_WINDOWS_BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
_MAC_BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
_NAMES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge"]

_CSS = """
@page { size: A4; margin: 14mm 12mm 16mm; }
body { font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #1b1f24; }
h1 { font-size: 17pt; margin: 0 0 8pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #c9d1d9; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 16pt; }
blockquote { margin: 6pt 0; padding: 4pt 8pt; border-left: 3px solid #4a72d9; background: #f2f6fd; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 8.5pt; background: #f2f4f7; padding: 0 2pt; }
pre { white-space: pre-wrap; font-size: 8.5pt; background: #f6f8fa; padding: 6pt; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0; font-size: 8pt; }
th, td { border: 1px solid #d0d7de; padding: 2.5pt 4pt; vertical-align: top; overflow-wrap: break-word; hyphens: none; }
th { background: #eef1f5; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 10pt 0; }
"""


def find_browser() -> str | None:
    """Chrome, Edge или Chromium, установленный в системе (или CHROME_PATH)."""
    if os.getenv("CHROME_PATH"):
        return os.environ["CHROME_PATH"]
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _WINDOWS_BROWSERS if sys.platform == "win32" else _MAC_BROWSERS:
        if Path(path).exists():
            return path
    return None


def markdown_to_print_html(md_text: str, title: str) -> str:
    """HTML для печати. Текст документа в отчёте уже экранирован; разрешён только перенос строки <br> в ячейках."""
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(md_text)
    return (
        f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


def export_pdf(md_path: Path, pdf_path: Path | None = None, browser: str | None = None) -> Path:
    """Сохранить Markdown-отчёт в PDF. RuntimeError — если браузер не найден или печать не удалась."""
    browser = browser or find_browser()
    if not browser:
        raise RuntimeError("Для PDF нужен Chrome, Edge или Chromium (путь можно задать в CHROME_PATH).")
    md_path = Path(md_path)
    pdf_path = Path(pdf_path or md_path.with_suffix(".pdf")).resolve()
    text = md_path.read_text(encoding="utf-8")
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), md_path.stem)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(markdown_to_print_html(text, title), encoding="utf-8")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={Path(tmp) / 'profile'}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            page.as_uri(),
        ]
        if hasattr(os, "geteuid") and os.geteuid() == 0:  # Chrome не запускается от root без этого флага (контейнеры)
            cmd.insert(1, "--no-sandbox")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Браузер не сохранил PDF: {(result.stderr or result.stdout)[-300:]}")
    return pdf_path
