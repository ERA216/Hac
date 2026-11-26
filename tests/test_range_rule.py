"""Правило «разные границы одной величины» (R-RANGE)."""

from pathlib import Path

from revizor.document import load_document
from revizor.rules import check_range_consistency
from revizor.verify import QuoteLocator

from .test_pipeline import DOC

P2P = Path(__file__).resolve().parent.parent / "eval" / "cases" / "p2p" / "requirements.md"
CLEAN = Path(__file__).resolve().parent / "fixtures" / "clean.md"


def _hits(path_or_text, tmp_path=None):
    if tmp_path is not None:
        p = tmp_path / "tz.md"
        p.write_text(path_or_text, encoding="utf-8")
        path_or_text = p
    doc = load_document(path_or_text)
    return check_range_consistency(doc, QuoteLocator(doc))


def test_conflicting_amount_ranges_between_parts():
    hits = _hits(P2P)
    assert len(hits) == 1 and "сумма" in hits[0].title
    assert [e.line_start for e in hits[0].evidence] == [35, 102]  # Ф-2 и параметр API amount
    assert hits[0].severity == "critical"


def test_consistent_ranges_are_quiet():
    assert _hits(DOC) == []  # Ф-1 и API: 1 000–1 000 000 в обоих местах
    assert _hits(CLEAN) == []


def test_different_quantities_do_not_conflict(tmp_path):
    text = (
        "# ТЗ\n\n## Функциональные требования\n"
        "* **Ф-1.** Сумма перевода — от 100 до 500 000 ₸.\n"
        "* **Ф-2.** Комиссия 1% от суммы, от 50 до 1 000 ₸.\n"
        "* **Ф-3.** Лимит в сутки — от 10 000 до 2 000 000 ₸.\n"
    )
    assert _hits(text, tmp_path) == []


def test_non_money_ranges_are_ignored(tmp_path):
    text = (
        "# ТЗ\n\n## API\n"
        "| Parameter | Format | Size |\n| --- | --- | --- |\n| `cardId` | string | 1–64 |\n"
        "Время работы 09:00–18:00, сутки 00:00:00–23:59:59.\n"
    )
    assert _hits(text, tmp_path) == []
