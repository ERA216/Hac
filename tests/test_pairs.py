"""Попарная сверка требований агентом согласованности."""

import json

from revizor.document import load_document
from revizor.orchestrator import Pipeline
from revizor.report import render_html_report, render_markdown, to_json

from .fake_llm import FakeLLM
from .test_pipeline import DOC


def test_contradicting_pair_becomes_critical_finding():
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    assert [p.verdict for p in state.pairs] == ["противоречат", "совместимы"]
    bad = state.pairs[0]
    assert bad.first.verified and bad.second.verified  # обе цитаты найдены в документе
    assert (bad.first.line_start, bad.second.line_start) == (39, 40)
    assert bad.finding_id.startswith("D-")  # ссылка ведёт на итоговый номер дефекта
    finding = next(f for f in state.findings if f.fid == bad.finding_id)
    assert {e.line_start for e in finding.evidence} == {39, 40}
    candidate = next(c for c in state.candidates if c.fid in finding.candidate_ids)
    assert candidate.severity == "critical" and candidate.criterion == "consistent"
    assert "противоречат друг другу" in candidate.title and "Сценарий:" in candidate.problem
    assert state.pairs[1].finding_id == ""
    step = next(s for s in state.steps if s.title == "Агент согласованности набора")
    assert "пар сверено 2 (противоречий 1)" in step.note


def test_pair_checks_in_all_report_formats():
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    md = render_markdown(state, llm)
    assert "### 4.5. Попарная сверка требований" in md and "❌ противоречат" in md and "✅ совместимы" in md
    assert len(to_json(state, llm)["pair_checks"]) == 2
    html = render_html_report(state, llm)
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload)["pair_checks"][0]["subject"] == "отправка push"


def test_consistency_prompt_asks_for_scenario_before_verdict():
    llm = FakeLLM()
    Pipeline(llm).run(load_document(DOC))
    prompt = llm.prompts["consistency"]
    assert "pair_checks" in prompt and "сценарий" in prompt
    assert prompt.index("pair_checks") < prompt.index("Что ещё проверить")
