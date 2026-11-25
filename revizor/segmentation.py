"""Нарезка документа без LLM и деление длинных документов на части.

- rule_segments(): детерминированная нарезка по структуре (пункты, нумерованные шаги,
  строки таблиц, абзацы). Используется, если LLM-нарезчик недоступен: прогон не обрывается.
- chunk_ranges(): деление длинного документа на части по заголовкам, чтобы ответ
  нарезчика не упирался в лимит выходных токенов модели.
"""

from __future__ import annotations

import re

from .document import Document
from .schemas import RequirementOut
from .verify import clean_line

# разделы без требований: метаданные, ревью, согласования, история
SKIP_SECTION = re.compile(
    r"общие сведения|общие параметры|ревью требований|блок согласующ|лист согласовани|согласующ|история изменени|"
    r"глоссари|термины и определени|содержание|оглавление",
    re.I,
)
_ID = re.compile(r"^\s*(?:[-*]\s+)?\**\s*([A-Za-zА-Яа-яЁё]{1,6}[-‑–]\d+(?:\.\d+)*)\.?\**")
_NUM_ITEM = re.compile(r"^\s*(\d{1,2})[.)]\s+")
_HEADING = re.compile(r"^#{1,6}\s")
_STATUS = re.compile(r"^\s*\*\*([1-5]\d\d)\.?\*\*")
_PLACEHOLDER = re.compile(r"заполня\w+ (позже|тестировщик\w*|аналитик\w*)|\bTBD\b|\bTODO\b|не заполнен", re.I)

KIND_BY_SECTION = [
    (re.compile(r"нефункциональн|производительн|надёжн|надежн|доступност|SLA", re.I), "nonfunctional"),
    (re.compile(r"функциональн", re.I), "functional"),
    (re.compile(r"критери\w* при[её]мк|acceptance", re.I), "acceptance"),
    (re.compile(r"данн|модель данных|хранени", re.I), "data"),
    (re.compile(r"интеграц", re.I), "integration"),
    (re.compile(r"UI|интерфейс|дизайн|экран", re.I), "interface"),
    (re.compile(r"API|параметр|ответ|статус|логика работы|особые правила|запрос", re.I), "api"),
    (re.compile(r"BPMN|sequence|процесс|архитектур|схема", re.I), "process"),
]


def _kind(section: str) -> str:
    for rx, kind in KIND_BY_SECTION:
        if rx.search(section):
            return kind
    return "constraint"


def _last(section: str) -> str:
    return section.split(" › ")[-1] if section else ""


def rule_segments(doc: Document, line_range: tuple[int, int] | None = None) -> list[RequirementOut]:
    a, b = line_range or (1, len(doc.lines))
    out: list[RequirementOut] = []
    in_code = False
    table_row = 0
    prev_table = False
    for no in range(a, b + 1):
        ln = doc.line(no)
        if ln is None:
            continue
        t = ln.text.strip()
        if t.startswith("```"):
            in_code = not in_code
            continue
        is_table = t.startswith("|")
        if not is_table:
            prev_table, table_row = False, 0
        if in_code or not t or t == "---" or ln.heading or _HEADING.match(t) or SKIP_SECTION.search(ln.section):
            continue
        if is_table:
            if re.match(r"^\|?\s*:?-{2,}", t):
                continue
            if not prev_table:  # первая строка таблицы — заголовок
                prev_table = True
                continue
            table_row += 1
            cells = [c for c in clean_line(t).split(" | ") if c]
            if not cells or all(re.fullmatch(r"[—–\-]|Да\s*/\s*Нет.*", c) for c in cells):
                continue
        quote = clean_line(t)
        # вводный текст до первого раздела, обрывки («или», «404») и заглушки шаблона — не требования
        if not ln.section or len(quote) < 12 or _PLACEHOLDER.search(quote):
            continue
        m_id = _ID.match(t)
        m_num = _NUM_ITEM.match(t)
        m_status = _STATUS.match(t)
        if m_status:
            source_id = f"Статус {m_status.group(1)}"
        elif m_id:
            source_id = m_id.group(1)
        elif m_num:
            source_id = f"{_last(ln.section)}, п. {m_num.group(1)}"
        elif is_table:
            source_id = f"{_last(ln.section)}, строка {table_row}"
        else:
            source_id = ""
        if m_id and not m_status:
            body = quote[len(source_id):].lstrip(" .")
        elif m_num or m_status:
            body = re.sub(r"^\d{1,3}[.)]\s*", "", quote)
        else:
            body = quote
        words = body.split()
        summary = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
        out.append(
            RequirementOut(
                source_id=source_id, section=ln.section, kind=_kind(ln.section),
                summary=summary, quote=quote, line_start=no, line_end=no,
            )
        )
    return out


def chunk_ranges(doc: Document, max_chars: int = 24000) -> list[tuple[int, int]]:
    """Диапазоны строк, каждый не длиннее max_chars нумерованного текста, с разрезами по заголовкам."""
    sizes = [len(ln.text) + 7 for ln in doc.lines]  # «L012| » + перевод строки
    if sum(sizes) <= max_chars:
        return [(1, len(doc.lines))]
    cuts = [ln.no for ln in doc.lines if re.match(r"^#{1,3}\s", ln.text)] or [1]
    if cuts[0] != 1:
        cuts.insert(0, 1)
    blocks = [(s, (cuts[i + 1] - 1) if i + 1 < len(cuts) else len(doc.lines)) for i, s in enumerate(cuts)]
    ranges: list[tuple[int, int]] = []
    cur_start, cur_size = None, 0
    for s, e in blocks:
        size = sum(sizes[s - 1 : e])
        if size > max_chars:  # раздел сам по себе велик — режем построчно
            if cur_start is not None:
                ranges.append((cur_start, s - 1))
                cur_start, cur_size = None, 0
            start, acc = s, 0
            for no in range(s, e + 1):
                acc += sizes[no - 1]
                if acc > max_chars and no > start:
                    ranges.append((start, no - 1))
                    start, acc = no, sizes[no - 1]
            ranges.append((start, e))
            continue
        if cur_start is None:
            cur_start, cur_size = s, size
        elif cur_size + size > max_chars:
            ranges.append((cur_start, s - 1))
            cur_start, cur_size = s, size
        else:
            cur_size += size
    if cur_start is not None:
        ranges.append((cur_start, len(doc.lines)))
    return ranges


def numbered_range(doc: Document, a: int, b: int) -> str:
    return "\n".join(f"L{ln.no:03d}| {ln.text}" for ln in doc.lines[a - 1 : b] if ln.text.strip())
