"""CLI целиком: прогон, ответы из файла, ошибки ввода, сборка HTML из JSON."""

import json

import pytest

import revizor.__main__ as cli
from revizor.config import llm_from_env
from revizor.report import html as html_mod

from .fake_llm import FakeLLM
from .test_pipeline import DOC


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(cli, "llm_from_env", lambda **kw: llm)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    return llm


def test_cli_full_run_writes_all_outputs(fake_llm, tmp_path):
    out = tmp_path / "report.md"
    assert cli.main([str(DOC), str(out)]) == 0
    for suffix in (".md", ".html", ".json"):
        assert out.with_suffix(suffix).exists()
    assert (tmp_path / "report.defects.csv").exists() and (tmp_path / "report.questions.csv").exists()
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["requirements"] and data["candidates"] and data["questions"]


def test_cli_answers_file_closes_defect(fake_llm, tmp_path):
    answers = tmp_path / "answers.txt"
    answers.write_text("Q-01: 1\n", encoding="utf-8")  # построчный формат и номер варианта
    out = tmp_path / "report.md"
    assert cli.main([str(DOC), str(out), "--answers", str(answers)]) == 0
    saved = json.loads(out.with_suffix(".answers.json").read_text(encoding="utf-8"))
    assert saved == {"Q-01": "Один"}
    assert "Закрыто уточнением" in out.read_text(encoding="utf-8")


def test_cli_input_errors(fake_llm, tmp_path):
    assert cli.main([]) == 2  # нет аргументов
    assert cli.main([str(tmp_path / "нет.md"), str(tmp_path / "r.md")]) == 2
    bad = tmp_path / "tz.xlsx"
    bad.write_bytes(b"x")
    assert cli.main([str(bad), str(tmp_path / "r.md")]) == 2


def test_cli_requires_key_unless_replay(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    monkeypatch.setattr(cli, "load_env", lambda: None)
    assert cli.main([str(DOC), str(tmp_path / "r.md")]) == 2


def test_cli_fatal_error_stops_with_code_3(monkeypatch, tmp_path):
    from revizor.llm import FatalLLMError

    class Broken(FakeLLM):
        def generate(self, agent, system, user, schema):
            raise FatalLLMError("ключ API недействителен")

    monkeypatch.setattr(cli, "llm_from_env", lambda **kw: Broken())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert cli.main([str(DOC), str(tmp_path / "r.md")]) == 3


def test_html_rebuilt_from_saved_json(fake_llm, tmp_path):
    out = tmp_path / "report.md"
    cli.main([str(DOC), str(out)])
    target = tmp_path / "again.html"
    assert html_mod.main([str(out.with_suffix(".json")), str(DOC), str(target)]) == 0
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert html_mod.main([]) == 2


def test_config_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    assert llm_from_env().provider == "gemini"  # пустая переменная — провайдер по умолчанию
    local = llm_from_env(provider="ollama", model="qwen3:4b")
    assert local.provider == "ollama" and local.min_interval == 0  # локальной модели лимит частоты не нужен
    assert llm_from_env(provider="openai", model="gpt-x").model == "gpt-x"
    assert llm_from_env(replay=True).extra_cache_dirs  # replay видит кэш финального прогона
