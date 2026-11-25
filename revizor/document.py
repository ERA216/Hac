"""Загрузка документа требований и его представление с номерами строк.

Поддерживаемые форматы: PDF, DOCX, Markdown, TXT. Любой формат приводится к
Markdown-подобному тексту; у каждой строки есть номер, страница (для PDF) и путь
разделов — это позволяет ссылаться на конкретное место документа.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# идентификатор требования посреди строки (PDF-парсер иногда склеивает пункты): «… **Ф-4.** …»
_MID_ID = re.compile(r"(?<=[.!?;])\s+(?=\*\*[A-Za-zА-Яа-яЁё]{1,6}[-‑–]?\d+(?:\.\d+)*\.?\*\*)")


def _split_glued(text: str) -> list[str]:
    if text.lstrip().startswith("|"):
        return [text]
    parts = _MID_ID.split(text)
    if len(parts) == 1:
        return [text]
    bullet = re.match(r"^\s*[-*]\s+", text)
    return [parts[0]] + [(bullet.group(0) if bullet else "") + p.strip() for p in parts[1:]]


@dataclass
class Line:
    no: int
    text: str
    page: int | None = None
    section: str = ""


@dataclass
class Document:
    path: Path
    fmt: str
    sha256: str
    title: str
    lines: list[Line] = field(default_factory=list)
    pages: int | None = None

    def numbered(self) -> str:
        """Текст для LLM: «L012| текст», пустые строки пропущены."""
        return "\n".join(f"L{ln.no:03d}| {ln.text}" for ln in self.lines if ln.text.strip())

    def line(self, no: int) -> Line | None:
        if 1 <= no <= len(self.lines):
            return self.lines[no - 1]
        return None

    def location(self, start: int | None, end: int | None = None) -> str:
        """Человекочитаемое место: раздел, строки, страница."""
        if not start:
            return "место не определено"
        ln = self.line(start)
        end = end or start
        rng = f"строка {start}" if end == start else f"строки {start}–{end}"
        parts = []
        if ln and ln.section:
            parts.append(ln.section)
        parts.append(rng)
        if ln and ln.page:
            parts.append(f"стр. {ln.page}")
        return ", ".join(parts)

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)


def _clean_heading(text: str) -> str:
    return re.sub(r"[*_`]", "", text).strip()


def _build(path: Path, fmt: str, raw: bytes, chunks: list[tuple[int | None, str]]) -> Document:
    lines: list[Line] = []
    stack: list[tuple[int, str]] = []
    title = ""
    for page, text in chunks:
        split = []
        for raw_line in text.splitlines():
            t = re.sub(r"<br\s*/?>", " ", raw_line.rstrip()).replace(" ", " ").replace(" ", " ")
            split.extend(_split_glued(t))
        for t in split:
            m = _HEADING.match(t.strip())
            if m:
                level, name = len(m.group(1)), _clean_heading(m.group(2))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, name))
                if not title and level == 1:
                    title = name
            # H1 обычно совпадает с названием документа — в путь раздела не включаем
            section = " › ".join(name for lvl, name in stack if not (lvl == 1 and name == title))
            lines.append(Line(no=len(lines) + 1, text=t, page=page, section=section))
    if not title:
        first = next((ln.text for ln in lines if ln.text.strip()), path.stem)
        title = _clean_heading(first.lstrip("# "))[:120]
    pages = max((ln.page or 0 for ln in lines), default=0) or None
    # для текстовых форматов хэш не зависит от переводов строк (CRLF в Windows, LF в git) —
    # паспорт отчёта совпадает на любой ОС и с оригиналом в репозитории
    digest_src = raw.replace(b"\r\n", b"\n") if fmt in ("md", "markdown", "txt") else raw
    return Document(
        path=path, fmt=fmt, sha256=hashlib.sha256(digest_src).hexdigest(), title=title, lines=lines, pages=pages
    )


def _read_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_pdf(path: Path) -> list[tuple[int | None, str]]:
    import contextlib
    import io

    # парсер печатает служебные сообщения (например, про OCR) — в отчёт и консоль они не нужны
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pymupdf4llm

        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    chunks: list[tuple[int | None, str]] = []
    for i, page in enumerate(pages, start=1):
        meta = page.get("metadata", {}) if isinstance(page, dict) else {}
        num = meta.get("page_number") or meta.get("page") or i
        text = page.get("text", "") if isinstance(page, dict) else str(page)
        chunks.append((int(num), text))
    return chunks


def _docx_table(table) -> list[str]:
    rows = []
    for r_i, row in enumerate(table.rows):
        cells = [" ".join(c.text.split()).replace("|", "\\|") for c in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
        if r_i == 0:
            rows.append("|" + "---|" * len(cells))
    return rows


def _load_docx(path: Path) -> list[tuple[int | None, str]]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    out: list[str] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            style = (p.style.name or "").lower() if p.style is not None else ""
            m = re.search(r"(heading|заголовок)\s*(\d)", style)
            if m and text:
                out.append("#" * int(m.group(2)) + " " + text)
            elif style == "title" and text:
                out.append("# " + text)
            elif "list" in style and text:
                out.append("* " + text)
            else:
                out.append(text)
        elif tag == "tbl":
            out.append("")
            out.extend(_docx_table(Table(child, d)))
            out.append("")
    return [(None, "\n".join(out))]


def load_document(path: str | Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Формат {ext} не поддерживается. Поддерживаются: {', '.join(sorted(SUPPORTED))}")
    raw = path.read_bytes()
    if ext == ".pdf":
        chunks = _load_pdf(path)
    elif ext == ".docx":
        chunks = _load_docx(path)
    else:
        chunks = [(None, _read_text(raw))]
    doc = _build(path, ext.lstrip("."), raw, chunks)
    if not any(ln.text.strip() for ln in doc.lines):
        raise ValueError(
            "Из документа не удалось извлечь текст (возможно, это скан без текстового слоя)."
        )
    return doc
