"""Проверка цитат: каждая цитата от LLM должна реально существовать в документе.

Это главный барьер против галлюцинаций: находка без подтверждённой цитаты
отбрасывается. Сравнение устойчиво к регистру, кавычкам, тире, неразрывным
пробелам, markdown-разметке и пропускам через «...».
"""

from __future__ import annotations

import bisect
import re

from rapidfuzz import fuzz

from .document import Document
from .schemas import Evidence

THRESHOLD = 88.0

_TRANS = str.maketrans(
    {
        "ё": "е", "Ё": "е", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "’": "'", "‘": "'",
        "—": "-", "–": "-", "−": "-", " ": " ", " ": " ", " ": " ",
    }
)


def normalize(s: str) -> str:
    s = s.translate(_TRANS).lower()
    s = re.sub(r"[*`|#>]", " ", s)
    s = re.sub(r"\\", "", s)
    return re.sub(r"\s+", " ", s).strip()


class QuoteLocator:
    def __init__(self, doc: Document):
        self.doc = doc
        parts, starts = [], []
        pos = 0
        for ln in doc.lines:
            norm = normalize(ln.text)
            starts.append(pos)
            parts.append(norm)
            pos += len(norm) + 1
        self.text = " ".join(parts)
        self.starts = starts

    def _line_at(self, offset: int) -> int:
        return max(1, bisect.bisect_right(self.starts, offset))

    def _find(self, piece: str) -> tuple[int, int, float] | None:
        q = normalize(piece)
        if len(q) < 3:
            return None
        idx = self.text.find(q)
        if idx != -1:
            return idx, idx + len(q), 100.0
        al = fuzz.partial_ratio_alignment(q, self.text, score_cutoff=THRESHOLD)
        if al is None:
            return None
        return al.dest_start, al.dest_end, al.score

    def locate(self, quote: str, hint_line: int | None = None) -> Evidence:
        pieces = [p for p in re.split(r"\.\.\.|…", quote) if normalize(p)]
        found = [self._find(p) for p in pieces] if pieces else []
        ev = Evidence(quote=quote.strip())
        if not found or any(f is None for f in found):
            if hint_line and self.doc.line(hint_line):
                ev.line_start = ev.line_end = hint_line
                ev.location = self.doc.location(hint_line)
            return ev
        start = min(f[0] for f in found)
        end = max(f[1] for f in found)
        ev.line_start = self._line_at(start)
        ev.line_end = max(ev.line_start, self._line_at(max(start, end - 1)))
        ev.score = round(min(f[2] for f in found), 1)
        ev.verified = True
        ev.quote = clean_line(quote) or quote.strip()  # для отчёта — без markdown-разметки
        ev.location = self.doc.location(ev.line_start, ev.line_end)
        return ev

    def line_evidence(self, line_no: int, quote: str | None = None) -> Evidence:
        """Доказательство, найденное правилом по номеру строки (заведомо верное)."""
        ln = self.doc.line(line_no)
        text = quote if quote is not None else clean_line(ln.text if ln else "")
        return Evidence(
            quote=text, line_start=line_no, line_end=line_no, location=self.doc.location(line_no),
            verified=True, score=100.0,
        )


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    """Убирает повторы: одинаковые цитаты и цитаты, вложенные в другую цитату той же строки."""
    out: list[Evidence] = []
    for e in sorted(items, key=lambda x: -len(x.quote)):
        ne = normalize(e.quote)
        dup = any(
            o.line_start == e.line_start and (ne in normalize(o.quote) or normalize(o.quote) in ne)
            for o in out
        )
        if not dup:
            out.append(e)
    order = {id(e): i for i, e in enumerate(items)}
    return sorted(out, key=lambda e: order[id(e)])


def clean_line(text: str) -> str:
    """Строка документа без markdown-разметки — для цитирования."""
    t = re.sub(r"\*\*|`", "", text)
    t = re.sub(r"^\s*([*\-]|#{1,6})\s+", "", t)
    if t.strip().startswith("|"):
        cells = [c.strip() for c in t.strip().strip("|").split("|")]
        t = " | ".join(c for c in cells if c)
    return re.sub(r"\s+", " ", t).strip()
