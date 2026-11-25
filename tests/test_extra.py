import json

import docx

from revizor.document import load_document
from revizor.orchestrator import Pipeline
from revizor.report import render_html_report, to_json

from .fake_llm import FakeLLM
from .test_pipeline import DOC


def test_docx_loader_headings_and_tables(tmp_path):
    d = docx.Document()
    d.add_heading("Тестовое ТЗ", level=1)
    d.add_heading("Функциональные требования", level=2)
    d.add_paragraph("Ф-1. Система отправляет push быстро.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Parameter", "Format"
    t.cell(1, 0).text, t.cell(1, 1).text = "cardId", "string"
    path = tmp_path / "tz.docx"
    d.save(path)

    doc = load_document(path)
    assert doc.title == "Тестовое ТЗ"
    texts = [ln.text for ln in doc.lines]
    assert "## Функциональные требования" in texts
    f1 = next(ln for ln in doc.lines if ln.text.startswith("Ф-1"))
    assert f1.section == "Функциональные требования"
    assert any(t.startswith("| cardId | string |") for t in texts)


def test_apply_answers_resolves_and_html_renders():
    llm = FakeLLM()
    pipe = Pipeline(llm)
    state = pipe.run(load_document(DOC))
    assert state.questions and state.findings[0].status == "open"
    note = pipe.apply_answers(state, {"Q-01": "1"})  # номер варианта превращается в текст варианта
    assert state.questions[0].answer == "Один"
    assert state.findings[0].status == "resolved"
    assert "закрыто" in note
    html = render_html_report(state, llm)
    assert html.startswith("<!doctype html>")
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert data["counts"]["resolved"] == 1 and len(data["doc_lines"]) == len(state.doc.lines)


def test_fix_ids_maps_candidates_to_defects():
    m = {"C-003": "D-01", "C-004": "D-01", "C-007": "D-05"}
    assert Pipeline._fix_ids("Разделы не заполнены (C-003, C-004, C-051).", m) == "Разделы не заполнены (D-01)."
    assert Pipeline._fix_ids("Пусто (C-099)", m) == "Пусто"


def test_parallel_and_sequential_are_identical():
    a, b = FakeLLM(), FakeLLM()
    s1 = Pipeline(a, parallel=4).run(load_document(DOC))
    s2 = Pipeline(b, parallel=1).run(load_document(DOC))
    strip = lambda s: [(f.fid, f.title, f.candidate_ids) for f in s.findings]  # noqa: E731
    assert strip(s1) == strip(s2)
    assert a.prompts["judge"] == b.prompts["judge"]
    assert to_json(s1, a)["questions"] == to_json(s2, b)["questions"]
