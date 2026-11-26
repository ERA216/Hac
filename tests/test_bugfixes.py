"""Регрессионные тесты на ошибки, найденные проверкой проекта 26.11."""

import csv
import json

from revizor.document import load_document
from revizor.llm import LLMClient
from revizor.orchestrator import Pipeline
from revizor.report import _csv_safe, render_html_report, render_markdown, write_reports
from revizor.schemas import ClarifyOut, SegmentationOut
from revizor.segmentation import rule_segments

from .fake_llm import FakeLLM
from .test_pipeline import DOC

GOOD = (
    '{"questions":[{"addressee":"Аналитик","question":"?","why":"w","options":["a","b"],'
    '"finding_ids":[],"requirement_ids":[],"blocking":false}]}'
)


class Backend:
    model = "m"

    def complete(self, system, user, schema):
        return GOOD, {"model_version": "m"}

    @staticmethod
    def retryable(err):
        return False


def _client(cache_dir, **kw):
    c = LLMClient("gemini", "m", cache_dir=cache_dir, rpm=0, **kw)
    c._backend = Backend()
    return c


def test_corrupted_cache_file_is_requested_again(tmp_path):
    c = _client(tmp_path)
    (tmp_path / f"{c._cache_key('s', 'u', ClarifyOut)}.json").write_text("{не json", encoding="utf-8")
    assert c.generate("a", "s", "u", ClarifyOut).questions[0].question == "?"
    assert c.real_calls == 1


def test_unwritable_cache_does_not_lose_paid_answer(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    msgs = []
    c = _client(blocker / "sub", on_wait=msgs.append)  # на месте каталога кэша — файл
    assert c.generate("a", "s", "u", ClarifyOut).questions[0].question == "?"
    assert any("не удалось сохранить ответ в кэш" in m for m in msgs)


class EmptySegmenter(FakeLLM):
    def generate(self, agent, system, user, schema):
        if agent == "segmenter":
            return SegmentationOut(document_title="x", requirements=[])
        return super().generate(agent, system, user, schema)


def test_empty_segmentation_falls_back_to_structure():
    state = Pipeline(EmptySegmenter()).run(load_document(DOC))
    assert len(state.requirements) > 30
    assert any("не выделила ни одного требования" in e for e in state.errors)


class BrokenJudge(FakeLLM):
    def generate(self, agent, system, user, schema):
        if agent == "judge":
            raise KeyError("неожиданный сбой")
        return super().generate(agent, system, user, schema)


def test_unexpected_error_in_one_agent_keeps_the_report():
    llm = BrokenJudge()
    state = Pipeline(llm).run(load_document(DOC))
    judge = next(s for s in state.steps if s.title == "Судья-верификатор")
    assert not judge.ok and "внутренняя ошибка KeyError" in judge.note
    assert state.findings and state.questions  # кандидаты дошли до отчёта, уточнения работали
    # противоречия из сверок ведут на итоговые дефекты и без судьи
    assert state.pairs[0].finding_id.startswith("D-") and state.acceptance[1].finding_id.startswith("D-")
    assert "Прогон выполнен не полностью" in render_markdown(state, llm)


class CountingRecheck(FakeLLM):
    def __init__(self):
        super().__init__()
        self.recheck_calls = 0

    def generate(self, agent, system, user, schema):
        if agent == "recheck":
            self.recheck_calls += 1
        return super().generate(agent, system, user, schema)


def test_answers_without_linked_defects_cost_no_model_call():
    llm = CountingRecheck()
    pipe = Pipeline(llm)
    state = pipe.run(load_document(DOC))
    state.questions[0].finding_ids = []
    note = pipe.apply_answers(state, {"Q-01": "1"})
    assert llm.recheck_calls == 0 and "перепроверка не нужна" in note


def test_report_path_with_other_extension_keeps_markdown(tmp_path):
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    for name in ("report.html", "report"):
        paths = {p.name for p in write_reports(state, llm, tmp_path / name)}
        assert {"report.md", "report.html", "report.json"} <= paths
        assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# Отчёт")
        assert (tmp_path / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_document_html_is_not_executable_in_reports(tmp_path):
    p = tmp_path / "tz.md"
    p.write_text(
        "# ТЗ </script><script>alert(1)</script>\n\n## Требования\n* **Ф-1.** Поле <img src=x onerror=alert(2)>.\n",
        encoding="utf-8",
    )
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(p))
    html = render_html_report(state, llm)
    assert "<script>alert" not in html and "<img src=x" not in html
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<img src=x" in json.dumps(json.loads(payload), ensure_ascii=False)  # данные целы, просто экранированы
    md = render_markdown(state, llm)
    assert "<script>" not in md and "&lt;" in md


def test_csv_cells_cannot_become_excel_formulas(tmp_path):
    assert _csv_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _csv_safe("+7 777") == "'+7 777" and _csv_safe("@SUM") == "'@SUM" and _csv_safe("-1") == "'-1"
    assert _csv_safe("обычный текст") == "обычный текст"
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    state.findings[0].title = '=HYPERLINK("http://evil")'
    write_reports(state, llm, tmp_path / "r.md")
    with (tmp_path / "r.defects.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    assert rows[1][6].startswith("'=")


def test_rule_segmentation_of_document_without_headings(tmp_path):
    p = tmp_path / "one.txt"
    p.write_text("Система должна отправлять уведомление в течение 60 секунд.\n", encoding="utf-8")
    reqs = rule_segments(load_document(p))
    assert len(reqs) == 1 and "60 секунд" in reqs[0].quote
