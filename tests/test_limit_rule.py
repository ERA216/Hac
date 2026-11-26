"""R-LIMIT-OVERRIDE: ограничение числа действий и то же действие «независимо от…» без приоритета."""

from revizor.document import load_document
from revizor.rules import check_limit_override
from revizor.verify import QuoteLocator

from .test_pipeline import DOC


def _hits(tmp_path, text):
    p = tmp_path / "tz.md"
    p.write_text("# ТЗ\n\n## Функциональные требования\n" + text, encoding="utf-8")
    doc = load_document(p)
    return check_limit_override(doc, QuoteLocator(doc))


def test_given_document_limit_and_override_found():
    doc = load_document(DOC)
    hits = check_limit_override(doc, QuoteLocator(doc))
    assert len(hits) == 1
    assert [e.line_start for e in hits[0].evidence] == [39, 40]
    assert hits[0].severity == "critical" and hits[0].criterion == "consistent"


def test_other_wording_and_word_forms(tmp_path):
    hits = _hits(
        tmp_path,
        "* **Ф-1.** Клиенту отправляется не более одного уведомления в сутки.\n"
        "* **Ф-2.** При входе с нового устройства уведомление отправляется всегда.\n",
    )
    assert len(hits) == 1 and "уведомлени" in hits[0].title


def test_explicit_precedence_is_not_a_defect(tmp_path):
    hits = _hits(
        tmp_path,
        "* **Ф-1.** Клиенту отправляется не более одного уведомления в сутки.\n"
        "* **Ф-2.** Уведомление о блокировке отправляется всегда, сверх лимита из Ф-1.\n",
    )
    assert hits == []


def test_unrelated_override_and_frequency_limits_are_ignored(tmp_path):
    hits = _hits(
        tmp_path,
        "* **Ф-1.** Пароль можно сменить не чаще одного раза в сутки.\n"
        "* **Ф-2.** Выписка формируется всегда в PDF.\n"
        "* **Ф-3.** На карту приходится максимум один push за месяц.\n",
    )
    assert hits == []
