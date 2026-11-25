"""Правила чек-листа ревью финтех-ТЗ и проверка окружения."""

from pathlib import Path

from rich.console import Console

from revizor.check import run_check
from revizor.document import load_document
from revizor.rules import run_rules
from revizor.verify import QuoteLocator

from .test_pipeline import DOC

P2P = Path(__file__).resolve().parent.parent / "eval" / "cases" / "p2p" / "requirements.md"


def _rules(path_or_text, tmp_path=None):
    if tmp_path is not None:
        p = tmp_path / "tz.md"
        p.write_text(path_or_text, encoding="utf-8")
        path_or_text = p
    doc = load_document(path_or_text)
    return {h.rule: h for h in run_rules(doc, QuoteLocator(doc))}


def test_fintech_rules_on_given_document():
    hits = _rules(DOC)
    assert hits["R-NFR-METRIC"].evidence[0].line_start == 46  # «в течение минуты» без числа
    assert hits["R-IDEMPOTENCY"].evidence[0].line_start == 52  # повторная доставка без ключа
    assert "проведённые покупки" in hits["R-TXN-STAGE"].title
    assert "R-TIMEZONE" not in hits  # часовой пояс задан: «в Алматы»


def test_fintech_rules_on_unseen_document():
    hits = _rules(P2P)
    assert {"R-NFR-METRIC", "R-TIMEZONE", "R-IDEMPOTENCY"} <= set(hits)


def test_idempotency_rule_quiet_when_mechanism_defined(tmp_path):
    text = (
        "# ТЗ\n\n## Функциональные требования\n"
        "* **Ф-1.** Повторное нажатие «Отправить» не приводит к двойному списанию.\n\n"
        "## API\nЗаголовок Idempotency-Key обязателен, хранится 24 часа.\n"
    )
    assert "R-IDEMPOTENCY" not in _rules(text, tmp_path)


def test_timezone_rule_fires_without_timezone(tmp_path):
    text = "# ТЗ\n\n## Функциональные требования\n* **Ф-1.** Лимит обнуляется в начале месяца.\n"
    hits = _rules(text, tmp_path)
    assert "R-TIMEZONE" in hits
    text_tz = text + "* **Ф-2.** Время — UTC+5.\n"
    assert "R-TIMEZONE" not in _rules(text_tz, tmp_path)


def test_check_reports_missing_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    console = Console(record=True, width=120)
    assert run_check(console) == 1
    out = console.export_text()
    assert "GEMINI_API_KEY" in out and "не задан" in out
