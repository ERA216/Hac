"""Устойчивость к незнакомым документам и недоступности модели."""

from pathlib import Path

from revizor.document import load_document
from revizor.llm import LLMError
from revizor.orchestrator import Pipeline
from revizor.rules import run_rules
from revizor.segmentation import chunk_ranges, rule_segments
from revizor.verify import QuoteLocator

from .fake_llm import FakeLLM
from .test_pipeline import DOC

P2P = Path(__file__).resolve().parent.parent / "eval" / "cases" / "p2p" / "requirements.md"


class DeadLLM(FakeLLM):
    """Модель недоступна для всех агентов."""

    def generate(self, agent, system, user, schema):
        raise LLMError("модель недоступна (тест)")


class ChunkLLM(FakeLLM):
    """Запоминает, сколько раз вызывался нарезчик (для длинных документов)."""

    def __init__(self):
        super().__init__()
        self.segment_calls = 0

    def generate(self, agent, system, user, schema):
        if agent == "segmenter":
            self.segment_calls += 1
            assert "часть" in user  # каждой части сообщается, какая это часть
        return super().generate(agent, system, user, schema)


def test_rule_segments_finds_requirements_without_llm():
    doc = load_document(DOC)
    reqs = rule_segments(doc)
    ids = [r.source_id for r in reqs]
    assert {"Ф-1", "Ф-2", "Ф-3", "Ф-4"} <= set(ids)
    assert "Статус 400" in ids and "Логика работы метода, п. 1" in ids
    assert not any("заполняются" in r.quote for r in reqs)  # заглушки шаблона не считаются требованиями
    assert not any(r.section.endswith("Общие сведения") for r in reqs)  # метаданные пропущены


def test_chunk_ranges_cover_document_without_overlap():
    doc = load_document(DOC)
    assert chunk_ranges(doc) == [(1, len(doc.lines))]
    parts = chunk_ranges(doc, max_chars=3000)
    assert len(parts) > 2 and parts[0][0] == 1 and parts[-1][1] == len(doc.lines)
    assert all(b + 1 == c for (_, b), (c, _) in zip(parts, parts[1:], strict=False))


def test_pipeline_survives_dead_model():
    state = Pipeline(DeadLLM()).run(load_document(DOC))
    assert len(state.requirements) > 30  # нарезка по структуре
    assert state.findings  # дефекты правил дошли до отчёта
    assert any("Нарезка требований" in e for e in state.errors)
    assert not state.ready_for_development


def test_long_document_is_segmented_in_parts():
    llm = ChunkLLM()
    Pipeline(llm, segment_chunk_chars=3000).run(load_document(DOC))
    assert llm.segment_calls > 2


def test_generic_rules_on_unseen_document():
    doc = load_document(P2P)
    hits = {h.rule: h for h in run_rules(doc, QuoteLocator(doc))}
    assert "требования к данным" in hits["R-TEMPLATE"].title
    assert "Ф-4" in hits["R-ID-GAP"].title
    assert "Ф-7" in hits["R-ID-REF"].title
    assert {"R-HTTP-GET", "R-API-TYPE", "R-API-QUERY-PII", "R-HTTP-CODE"} <= set(hits)


def test_template_rules_quiet_on_complete_document():
    for path in (DOC, DOC.with_suffix(".pdf")):
        doc = load_document(path)
        rules = {h.rule for h in run_rules(doc, QuoteLocator(doc))}
        assert not rules & {"R-TEMPLATE", "R-ID-GAP", "R-ID-DUP", "R-ID-REF"}
