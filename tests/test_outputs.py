"""Выходные файлы (CSV, «было → стало») и провайдер локальной модели Ollama."""

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from revizor.document import load_document
from revizor.llm import LLMClient
from revizor.orchestrator import Pipeline
from revizor.report import render_markdown, write_reports
from revizor.schemas import ClarifyOut

from .fake_llm import FakeLLM
from .test_pipeline import DOC


def test_csv_exports_and_rewrite_appendix(tmp_path):
    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    state.findings[0].rewrite = "Новая формулировка требования"
    paths = write_reports(state, llm, tmp_path / "report.md")
    names = {p.name for p in paths}
    assert {"report.md", "report.html", "report.json", "report.defects.csv", "report.questions.csv"} <= names

    with (tmp_path / "report.defects.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    assert rows[0][:3] == ["ID", "Критичность", "Приоритет Jira"]
    assert len(rows) == len(state.findings) + 1

    with (tmp_path / "report.questions.csv").open(encoding="utf-8-sig") as fh:
        q = list(csv.reader(fh, delimiter=";"))
    assert q[1][0] == "Q-01" and "1. Один" in q[1][4]

    md = render_markdown(state, llm)
    assert "Исправления текста ТЗ: было → стало" in md and "Новая формулировка требования" in md


class _FakeOllama(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.received = body
        answer = {
            "questions": [
                {
                    "addressee": "Аналитик",
                    "question": "Что считать минутой?",
                    "why": "спор",
                    "options": ["от события", "от покупки"],
                    "finding_ids": [],
                    "requirement_ids": [],
                    "blocking": False,
                }
            ]
        }
        payload = json.dumps(
            {
                "model": body["model"],
                "message": {"content": json.dumps(answer, ensure_ascii=False)},
                "prompt_eval_count": 42,
                "eval_count": 7,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_ollama_provider_native_api():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = LLMClient(
            "ollama", "qwen3:4b", base_url=f"http://127.0.0.1:{server.server_port}/v1", cache_dir=None, rpm=0, num_ctx=16384
        )
        out = llm.generate("clarifier", "system", "user", ClarifyOut)
    finally:
        server.shutdown()
    assert out.questions[0].question == "Что считать минутой?"
    sent = _FakeOllama.received
    assert sent["options"]["num_ctx"] == 16384 and sent["think"] is False and sent["stream"] is False
    assert sent["format"]["type"] == "object"  # JSON-схема ответа передаётся модели
    assert llm.calls[0].prompt_tokens == 42 and llm.calls[0].model_version == "qwen3:4b"


def test_question_coverage_of_serious_defects_is_reported():
    from revizor.document import load_document
    from revizor.orchestrator import Pipeline
    from revizor.report import render_markdown, to_json

    from .fake_llm import FakeLLM
    from .test_pipeline import DOC

    llm = FakeLLM()
    state = Pipeline(llm).run(load_document(DOC))
    asked, total = state.asked("critical")
    assert total == state.count("critical") and 0 <= asked <= total
    state.questions[0].finding_ids = [f.fid for f in state.open_findings if f.severity == "critical"]
    assert state.asked("critical") == (total, total)
    assert f"Вопросы покрывают {total} из {total} критичных" in render_markdown(state, llm)
    assert to_json(state, llm)["question_coverage"]["critical"] == [total, total]
