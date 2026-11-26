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


def test_deadline_exceeded_switches_model_immediately():
    # 504: модель не успела за таймаут — сразу следующая модель, без повторов на той же
    c = client([ApiError(504, "DEADLINE_EXCEEDED"), GOOD], fallback_models=["m2"])
    c.generate("a", "s", "u", ClarifyOut)
    assert c._backend.models_used == ["m1", "m2"]


def test_replay_finds_answer_saved_under_other_model_name(tmp_path):
    c = client([GOOD], tmp_path)  # сохранили ответ, полученный моделью m1
    c.generate("clarifier", "s", "u", ClarifyOut)
    other = LLMClient("gemini", "другая-модель-из-env", cache_dir=tmp_path, replay=True, rpm=0)
    assert other.generate("clarifier", "s", "u", ClarifyOut).questions[0].addressee == "Аналитик"
    not_replay = LLMClient("gemini", "другая-модель-из-env", cache_dir=tmp_path, rpm=0)
    not_replay._backend = ScriptedBackend([GOOD], model="другая-модель-из-env")
    not_replay.generate("clarifier", "s", "u", ClarifyOut)
    assert not_replay.real_calls == 1  # без --replay кэш другой модели не подменяет новую


class PingBackend(ScriptedBackend):
    def __init__(self, alive, **kw):
        super().__init__([GOOD], **kw)
        self.alive = set(alive)
        self.pinged = []

    def ping(self):
        self.pinged.append(self.model)
        if self.model not in self.alive:
            raise ApiError(504, "DEADLINE_EXCEEDED")


def test_preflight_picks_first_responsive_model():
    msgs = []
    c = LLMClient("gemini", "m1", cache_dir=None, rpm=0, fallback_models=["m2", "m3"], on_wait=msgs.append)
    c._backend = PingBackend(alive={"m3"})
    assert c.preflight() == "m3"
    assert c._backend.pinged == ["m1", "m2", "m3"] and c._backend.model == "m3"
    c.generate("a", "s", "u", ClarifyOut)
    assert c.calls[0].model == "m3" and any("работаем на m3" in m for m in msgs)


def test_real_gemini_backend_ping_is_instance_method(monkeypatch):
    # настоящий класс бэкенда: ping вызывается у экземпляра и идёт в API с коротким таймаутом
    from revizor.llm import PREFLIGHT_TIMEOUT_MS, _Gemini

    backend = _Gemini("test-key", "m1", None, None)
    sent = {}

    class Models:
        def generate_content(self, model, contents, config):
            sent.update(model=model, max_tokens=config.max_output_tokens)

    class Client:
        models = Models()

    monkeypatch.setattr(llm_mod, "_gemini_client", lambda key, timeout_ms=0: sent.update(timeout=timeout_ms) or Client())
    backend.ping()
    assert sent == {"timeout": PREFLIGHT_TIMEOUT_MS, "model": "m1", "max_tokens": 16}


def test_preflight_keeps_primary_when_nothing_answers():
    c = LLMClient("gemini", "m1", cache_dir=None, rpm=0, fallback_models=["m2"])
    c._backend = PingBackend(alive=set())
    assert c.preflight() is None and c._backend.model == "m1" and c.switches == []


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
