"""Качество на разных видах документов, точность правил и поведение при ошибках настройки."""

import socket
from pathlib import Path

import docx
import pytest

from revizor.document import load_document
from revizor.llm import FatalLLMError, LLMClient, LLMError
from revizor.orchestrator import Pipeline
from revizor.rules import run_rules
from revizor.schemas import ClarifyOut, RecheckOut
from revizor.segmentation import rule_segments
from revizor.verify import QuoteLocator

from .fake_llm import FakeLLM
from .test_pipeline import DOC

FIXTURES = Path(__file__).resolve().parent / "fixtures"

PLAIN = """ТЕХНИЧЕСКОЕ ЗАДАНИЕ: Кэшбэк за покупки

1. Общие сведения
Продукт: мобильный банк.

2. Функциональные требования
2.1 Начисление кэшбэка
ФТ-1. Клиент получает кэшбэк 1% за покупки в категории «Продукты».
ФТ-2. Кэшбэк начисляется быстро после покупки.
ФТ-4. Максимальный кэшбэк в месяц — 10 000 тенге.

3. Нефункциональные требования
Сервис должен быть надёжным и выдерживать высокую нагрузку.

4. Логика
1. Проверить карту
2. Начислить кэшбэк
"""


def test_plain_text_numbered_headings(tmp_path):
    p = tmp_path / "tz.txt"
    p.write_text(PLAIN, encoding="utf-8")
    doc = load_document(p)
    ft1 = next(ln for ln in doc.lines if ln.text.startswith("ФТ-1"))
    assert ft1.section == "2. Функциональные требования › 2.1 Начисление кэшбэка"
    # пункты нумерованного списка не превращаются в заголовки
    step = next(ln for ln in doc.lines if ln.text.startswith("1. Проверить карту"))
    assert not step.heading and step.section == "4. Логика"
    ids = [r.source_id for r in rule_segments(doc)]
    assert {"ФТ-1", "ФТ-2", "ФТ-4"} <= set(ids)
    assert not any("Функциональные требования" == r.quote for r in rule_segments(doc))
    rules = {h.rule for h in run_rules(doc, QuoteLocator(doc))}
    assert {"R-LANG", "R-ID-GAP"} <= rules


def test_docx_bold_paragraphs_become_headings(tmp_path):
    d = docx.Document()
    for text, bold in [
        ("Функциональные требования", True),
        ("ФТ-1. Клиент задаёт лимит.", False),
        ("Нефункциональные требования", True),
        ("Ответ за 300 мс.", False),
    ]:
        d.add_paragraph().add_run(text).bold = bold
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Parameter", "Format"
    t.cell(1, 0).text, t.cell(1, 1).text = "limit", "integer"
    d.add_paragraph("После таблицы")
    path = tmp_path / "bold.docx"
    d.save(path)
    doc = load_document(path)
    ft1 = next(ln for ln in doc.lines if ln.text.startswith("ФТ-1"))
    assert ft1.section == "Функциональные требования"
    texts = [ln.text for ln in doc.lines]
    i = texts.index("После таблицы")
    assert texts[i - 1] == ""  # после таблицы пустая строка, текст не склеивается с таблицей


def test_rules_quiet_on_clean_document():
    doc = load_document(FIXTURES / "clean.md")
    assert run_rules(doc, QuoteLocator(doc)) == []


def test_sensitive_parameter_in_url_example(tmp_path):
    p = tmp_path / "api.md"
    p.write_text("# API\n\n## Запрос\nПример: `GET /v1/clients?phone=77011234567&limit=5`\n", encoding="utf-8")
    doc = load_document(p)
    titles = [h.title for h in run_rules(doc, QuoteLocator(doc))]
    assert any("phone" in t for t in titles)


class _ContradictingRecheck(FakeLLM):
    def generate(self, agent, system, user, schema):
        if agent == "recheck":
            self.calls.append(self.calls[-1])
            return RecheckOut(
                statuses=[
                    {"finding_id": "D-01", "status": "resolved", "comment": "Ответ не уточняет поведение при смене порога."},
                ],
                new_findings=[],
            )
        return super().generate(agent, system, user, schema)


def test_resolved_status_contradicting_comment_is_reopened():
    pipe = Pipeline(_ContradictingRecheck())
    state = pipe.run(load_document(DOC))
    note = pipe.apply_answers(state, {"Q-01": "1"})
    d01 = next(f for f in state.findings if f.fid == "D-01")
    assert d01.status == "open" and d01.status_comment.startswith("Ответ не снимает дефект")
    assert "возвращено в открытые" in note


class _InvalidKeyBackend:
    model = "gemini-test"

    def complete(self, system, user, schema):
        raise RuntimeError("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key. API_KEY_INVALID")

    @staticmethod
    def retryable(err):
        return False


def test_invalid_key_stops_run_with_clear_message():
    llm = LLMClient("gemini", "gemini-test", cache_dir=None, rpm=0)
    llm._backend = _InvalidKeyBackend()
    with pytest.raises(FatalLLMError, match="ключ API"):
        Pipeline(llm).run(load_document(DOC))


def test_ollama_not_running_fails_fast():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # порт заведомо закрыт
    llm = LLMClient("ollama", "qwen3:4b", base_url=f"http://127.0.0.1:{port}", cache_dir=None, rpm=0)
    with pytest.raises(FatalLLMError, match="не отвечает"):
        llm.generate("clarifier", "s", "u", ClarifyOut)
    assert issubclass(FatalLLMError, LLMError)


def test_budget_grows_for_long_documents():
    llm = FakeLLM()
    llm.max_calls = 9
    Pipeline(llm, segment_chunk_chars=3000).run(load_document(DOC))
    assert llm.max_calls >= 9 + 3
