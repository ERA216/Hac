"""LLM-клиент: кэш, replay, бюджет, починка JSON, повторы, резервные модели, фатальные ошибки."""

import pytest

from revizor import llm as llm_mod
from revizor.llm import BudgetExceeded, CacheMiss, FatalLLMError, LLMClient, LLMError, QuotaExhausted
from revizor.schemas import ClarifyOut

GOOD = (
    '{"questions":[{"addressee":"Аналитик","question":"?","why":"w","options":["a","b"],'
    '"finding_ids":[],"requirement_ids":[],"blocking":false}]}'
)


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(f"{code} {message}")
        self.code = code


class ScriptedBackend:
    """Отвечает по сценарию: элемент — текст ответа или исключение."""

    def __init__(self, script, model="m1"):
        self.script = list(script)
        self.model = model
        self.models_used = []
        self.prompts = []

    def complete(self, system, user, schema):
        self.models_used.append(self.model)
        self.prompts.append(user)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        meta = {"model_version": self.model, "prompt_tokens": 10, "output_tokens": 5}
        if isinstance(item, tuple):  # (текст, доп. метаданные), например признак обрезки
            item, extra = item
            meta.update(extra)
        return item, meta

    @staticmethod
    def retryable(err):
        return getattr(err, "code", None) in (429, 500, 503)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def client(script, tmp_path=None, **kw):
    c = LLMClient("gemini", "m1", cache_dir=tmp_path, rpm=0, **kw)
    c._backend = ScriptedBackend(script)
    return c


def test_cache_hit_and_replay(tmp_path):
    c = client([GOOD], tmp_path)
    c.generate("clarifier", "s", "u", ClarifyOut)
    assert c.real_calls == 1
    c2 = client([], tmp_path, replay=True)  # без сети: только кэш
    assert c2.generate("clarifier", "s", "u", ClarifyOut).questions[0].addressee == "Аналитик"
    assert c2.calls[0].cached and c2.real_calls == 0
    with pytest.raises(CacheMiss):
        c2.generate("clarifier", "s", "другой промпт", ClarifyOut)


def test_budget_exceeded():
    c = client([GOOD, GOOD], max_calls=1)
    c.generate("a", "s", "u1", ClarifyOut)
    with pytest.raises(BudgetExceeded):
        c.generate("a", "s", "u2", ClarifyOut)


def test_invalid_json_is_repaired_once():
    c = client(["не json", "```json\n" + GOOD + "\n```"])
    out = c.generate("clarifier", "s", "u", ClarifyOut)
    assert out.questions[0].options == ["a", "b"] and c.calls[0].attempts == 2


def test_truncated_answer_asks_for_shorter_json():
    c = client([('{"questions": [{"addressee": "Анали', {"truncated": True}), GOOD])
    c.generate("judge", "s", "u", ClarifyOut)
    repair_prompt = c._backend.prompts[1]
    assert "был обрезан" in repair_prompt and "короче" in repair_prompt


def test_invalid_json_twice_is_agent_error():
    c = client(["не json", "{}"])
    with pytest.raises(LLMError, match="не соответствует схеме"):
        c.generate("clarifier", "s", "u", ClarifyOut)


def test_overload_falls_back_to_next_model():
    c = client([ApiError(503, "UNAVAILABLE high demand")] * 3 + [GOOD], fallback_models=["m2"])
    c.generate("a", "s", "u", ClarifyOut)
    assert c._backend.models_used == ["m1", "m1", "m1", "m2"]
    assert c.calls[0].model == "m2" and c.switches == ["m1"]


def test_daily_quota_switches_and_then_exhausts():
    quota = ApiError(429, "RESOURCE_EXHAUSTED requests per day limit exceeded")
    c = client([quota, GOOD], fallback_models=["m2"])
    c.generate("a", "s", "u", ClarifyOut)
    assert c._backend.model == "m2"
    c2 = client([quota])
    with pytest.raises(QuotaExhausted):
        c2.generate("a", "s", "u", ClarifyOut)


def test_transient_errors_are_retried():
    c = client([ApiError(500, "internal"), ApiError(429, "per minute"), GOOD])
    c.generate("a", "s", "u", ClarifyOut)
    assert c.calls[0].attempts == 3


def test_missing_model_without_fallback_is_fatal():
    c = client([ApiError(404, "model not found")])
    with pytest.raises(FatalLLMError, match="модель не найдена"):
        c.generate("a", "s", "u", ClarifyOut)


def test_permission_denied_tries_fallback_then_fatal():
    denied = ApiError(403, "PERMISSION_DENIED")
    c = client([denied, GOOD], fallback_models=["m2"])
    c.generate("a", "s", "u", ClarifyOut)
    assert c._backend.model == "m2"
    with pytest.raises(FatalLLMError, match="нет доступа"):
        client([denied]).generate("a", "s", "u", ClarifyOut)


def test_non_retryable_error_is_reported():
    c = client([ApiError(400, "bad request")])
    with pytest.raises(LLMError, match="Ошибка вызова модели"):
        c.generate("a", "s", "u", ClarifyOut)


def test_gemini_client_has_no_hidden_sdk_retries():
    # иначе SDK сам ждёт до минуты на каждой исчерпанной или перегруженной модели
    from revizor.llm import GEMINI_TIMEOUT_MS, _gemini_client

    options = _gemini_client("test-key")._api_client._http_options
    assert options.retry_options.attempts == 1
    assert options.timeout == GEMINI_TIMEOUT_MS


def test_inline_schema_keeps_field_named_title():
    from revizor.llm import inline_schema
    from revizor.schemas import FindingOut

    schema = inline_schema(FindingOut)
    assert "title" in schema["properties"] and "$defs" not in str(schema)


def test_extract_json_variants():
    from revizor.llm import extract_json

    assert extract_json('Вот ответ: {"a": 1} спасибо') == '{"a": 1}'
    assert extract_json('```json\n{"a": 2}\n```') == '{"a": 2}'
