"""Пересборка отчёта из report.json: те же данные и паспорт исходного прогона, новое оформление."""

import json

from revizor.document import load_document
from revizor.orchestrator import Pipeline
from revizor.report import render_markdown, to_json
from revizor.report.html import main as rebuild_main
from revizor.report.rebuild import state_from_report

from .fake_llm import FakeLLM
from .test_pipeline import DOC


def test_markdown_rebuilt_from_json_is_identical():
    llm = FakeLLM()
    pipe = Pipeline(llm)
    state = pipe.run(load_document(DOC))
    pipe.apply_answers(state, {"Q-01": "1"})  # с ответами и перепроверкой — самый полный отчёт
    data = json.loads(json.dumps(to_json(state, llm), ensure_ascii=False, default=str))
    state2, run2 = state_from_report(data, load_document(DOC))
    assert render_markdown(state2, run2, version="v") == render_markdown(state, llm, version="v")


def test_rebuild_command_keeps_original_passport(tmp_path):
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    data = to_json(state, llm)
    data["passport"]["agent_version"] = "revizor 0.1.0 (git 1234567)"  # как будто прогон был старой версией
    src = tmp_path / "report.json"
    src.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    assert rebuild_main([str(src), str(DOC), str(tmp_path / "out" / "report.html")]) == 0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| Версия агента | revizor 0.1.0 (git 1234567) |" in md
    assert "| Оформление отчёта | пересобрано версией" in md and "без обращения к модели" in md
    assert (tmp_path / "out" / "report.html").exists()
