"""Клиент LLM: провайдеры, структурированный вывод, кэш, лимиты, ретраи, трассировка.

Каждый вызов агента проходит через LLMClient.generate():
  1. ключ кэша = хэш (провайдер, модель, промпты, схема) — повторный прогон бесплатен;
  2. бюджет вызовов (max_calls) защищает бесплатную квоту;
  3. ограничение частоты (rpm) и повторы при 429/5xx;
  4. ответ валидируется Pydantic-схемой, при ошибке — одна попытка починки;
  5. каждый вызов пишется в трассировку (агент, модель, токены, время, кэш).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


class CacheMiss(LLMError):
    pass


@dataclass
class CallRecord:
    agent: str
    model: str
    model_version: str
    cached: bool
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    attempts: int = 1


# ------------------------------------------------------------------ helpers
def inline_schema(model: type[BaseModel]) -> dict:
    """JSON Schema без $ref/$defs/title — максимально совместимо с провайдерами."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(dict(defs[name]))
            # "title"/"default" как аннотации схемы убираем, но не поля модели с таким именем
            return {
                k: walk(v)
                for k, v in node.items()
                if not (k in ("title", "default") and not isinstance(v, dict))
            }
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def extract_json(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _retry_delay(err: Exception) -> float | None:
    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", str(err), re.I)
    return float(m.group(1)) if m else None


def _is_daily_quota(err: Exception) -> bool:
    s = str(err)
    return bool(re.search(r"PerDay|per day|daily", s, re.I))


def _status(err: Exception) -> int | None:
    for attr in ("code", "status_code"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    return None


def _unavailable_reason(err: Exception) -> str | None:
    """Причина, по которой имеет смысл переключиться на резервную модель."""
    code = _status(err)
    if code == 404:
        return "модель не найдена"
    if code == 503 or re.search(r"high demand|overloaded|UNAVAILABLE", str(err)):
        return "модель перегружена (503)"
    if code == 429 and _is_daily_quota(err):
        return "исчерпана дневная квота"
    return None


# ------------------------------------------------------------------ backends
class _Gemini:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float | None, thinking_level: str | None):
        from google import genai

        if not api_key:
            raise LLMError("Не задан GEMINI_API_KEY (см. README, раздел «Ключ LLM»).")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.thinking_level = thinking_level
        self.schema_supported = True

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, dict]:
        from google.genai import errors, types

        kwargs = dict(system_instruction=system, response_mime_type="application/json", max_output_tokens=32768)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        if self.schema_supported:
            kwargs["response_json_schema"] = inline_schema(schema)
        else:
            user = user + "\n\nВерни JSON строго по схеме:\n" + json.dumps(inline_schema(schema), ensure_ascii=False)
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**kwargs)
            )
        except errors.ClientError as e:
            if e.code == 400 and self.schema_supported and "schema" in str(e).lower():
                self.schema_supported = False  # модель не поддерживает JSON Schema — схема уйдёт в промпт
                return self.complete(system, user, schema)
            raise
        usage = resp.usage_metadata
        meta = {
            "model_version": resp.model_version or self.model,
            "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "thoughts_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
        }
        text = resp.text
        if not text:
            reason = resp.candidates[0].finish_reason if resp.candidates else "нет кандидатов"
            raise LLMError(f"Пустой ответ модели (finish_reason={reason})")
        return text, meta

    @staticmethod
    def retryable(err: Exception) -> bool:
        from google.genai import errors

        if isinstance(err, errors.ServerError):
            return True
        return isinstance(err, errors.ClientError) and err.code == 429

    @staticmethod
    def list_models(api_key: str) -> list[str]:
        from google import genai

        client = genai.Client(api_key=api_key)
        out = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                out.append(m.name.removeprefix("models/"))
        return out


class _OpenAICompatible:
    """OpenAI и совместимые API: Ollama, LM Studio, vLLM, OpenRouter и т.п."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None, temperature: float | None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)
        self.model = model
        self.temperature = temperature
        self.mode = "json_schema"

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, dict]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": inline_schema(schema)},
            }
        else:
            messages[1]["content"] += "\n\nВерни JSON строго по схеме:\n" + json.dumps(
                inline_schema(schema), ensure_ascii=False
            )
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # провайдер не умеет json_schema — откатываемся на json_object
            if self.mode == "json_schema" and "response_format" in str(e):
                self.mode = "json_object"
                return self.complete(system, user, schema)
            raise
        usage = resp.usage
        meta = {
            "model_version": resp.model or self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return resp.choices[0].message.content or "", meta

    @staticmethod
    def retryable(err: Exception) -> bool:
        code = getattr(err, "status_code", None)
        return code == 429 or (isinstance(code, int) and code >= 500)


# ------------------------------------------------------------------ client
class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        cache_dir: Path | None = Path(".cache/llm"),
        replay: bool = False,
        rpm: float = 5,
        max_calls: int = 12,
        temperature: float | None = None,
        thinking_level: str | None = None,
        on_wait: Callable[[str], None] | None = None,
        fallback_models: list[str] | None = None,
        extra_cache_dirs: list[Path] | None = None,
    ):
        self.provider = provider
        self.model = model
        # резервные модели: при перегрузке/недоступности/исчерпании квоты переключаемся по цепочке
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.switches: list[str] = []
        self.cache_dir = cache_dir
        self.extra_cache_dirs = [d for d in (extra_cache_dirs or []) if d and d != cache_dir]
        self.replay = replay
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.max_calls = max_calls
        self.on_wait = on_wait or (lambda msg: None)
        self.calls: list[CallRecord] = []
        self._last_call = -1e9
        self._lock = threading.Lock()
        self._backend = None
        self._backend_args = (api_key, base_url, temperature, thinking_level)

    # ленивое создание: при --replay ключ не нужен
    @property
    def backend(self):
        if self._backend is None:
            api_key, base_url, temperature, thinking_level = self._backend_args
            if self.provider == "gemini":
                self._backend = _Gemini(api_key, self.model, temperature, thinking_level)
            elif self.provider == "openai":
                self._backend = _OpenAICompatible(api_key, self.model, base_url, temperature)
            else:
                raise LLMError(f"Неизвестный провайдер: {self.provider}")
        return self._backend

    @property
    def real_calls(self) -> int:
        return sum(1 for c in self.calls if not c.cached)

    @property
    def model_versions(self) -> list[str]:
        return sorted({c.model_version for c in self.calls if c.model_version})

    def _cache_key(self, system: str, user: str, schema: type[BaseModel]) -> str:
        payload = json.dumps(
            [self.provider, self.model, system, user, inline_schema(schema)], ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, agent: str, system: str, user: str, schema: type[T]) -> T:
        key = self._cache_key(system, user, schema)
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        # дополнительные каталоги кэша только для чтения (например, ответы финального прогона в репозитории)
        found = next((p for p in [path, *(d / f"{key}.json" for d in self.extra_cache_dirs)] if p and p.exists()), None)
        if found:
            data = json.loads(found.read_text(encoding="utf-8"))
            try:
                obj = schema.model_validate_json(extract_json(data["text"]))
                self.calls.append(CallRecord(agent=agent, model=self.model, cached=True, seconds=0.0,
                                             **{k: data["meta"].get(k, 0) for k in
                                                ("prompt_tokens", "output_tokens", "thoughts_tokens")},
                                             model_version=data["meta"].get("model_version", self.model)))
                return obj
            except (ValidationError, KeyError, json.JSONDecodeError):
                pass  # битый кэш — запрашиваем заново
        if self.replay:
            raise CacheMiss(f"Режим --replay: нет сохранённого ответа для агента «{agent}».")
        if self.real_calls >= self.max_calls:
            raise BudgetExceeded(f"Превышен бюджет вызовов LLM ({self.max_calls}). Увеличьте --max-calls.")

        started = time.monotonic()
        text, meta, attempts = self._call(system, user, schema)
        try:
            obj = schema.model_validate_json(extract_json(text))
        except ValidationError as err:
            repair = (
                user
                + "\n\nТвой предыдущий ответ не прошёл проверку схемы:\n"
                + str(err)[:1500]
                + "\nВерни исправленный JSON целиком."
            )
            text, meta, more = self._call(system, repair, schema)
            attempts += more
            try:
                obj = schema.model_validate_json(extract_json(text))
            except ValidationError as err2:
                raise LLMError(f"Ответ модели для агента «{agent}» не соответствует схеме: {str(err2)[:300]}") from err2
        record = CallRecord(
            agent=agent,
            model=self.backend.model,
            model_version=meta.get("model_version", self.model),
            cached=False,
            seconds=round(time.monotonic() - started, 1),
            prompt_tokens=meta.get("prompt_tokens", 0),
            output_tokens=meta.get("output_tokens", 0),
            thoughts_tokens=meta.get("thoughts_tokens", 0),
            attempts=attempts,
        )
        self.calls.append(record)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"agent": agent, "text": text, "meta": meta}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        return obj

    def _throttle(self):
        """Ограничение частоты, общее для всех потоков: каждый вызов получает свой слот времени."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last_call + self.min_interval)
            self._last_call = slot
        if slot > now:
            time.sleep(slot - now)

    def _switch_model(self, reason: str, failed_model: str) -> bool:
        """Переключиться на следующую резервную модель; False — резервы кончились."""
        with self._lock:
            if self.backend.model != failed_model:
                return True  # другой поток уже переключил модель
            tried = {self.backend.model, *self.switches}
            nxt = next((m for m in self.fallback_models if m not in tried), None)
            if not nxt:
                return False
            self.switches.append(self.backend.model)
            self.backend.model = nxt
        self.on_wait(f"{failed_model}: {reason} — переключаюсь на {nxt}")
        return True

    def _call(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, dict, int]:
        backend = self.backend
        attempts = 0
        overloaded = 0
        while True:
            attempts += 1
            self._throttle()
            used_model = backend.model
            try:
                text, meta = backend.complete(system, user, schema)
                return text, meta, attempts
            except Exception as err:  # noqa: BLE001 — решаем ниже, повторять ли
                reason = _unavailable_reason(err)
                if reason:
                    # перегрузку сначала пережидаем коротко, затем идём к резервной модели
                    if reason.startswith("модель перегружена") and overloaded < 2:
                        overloaded += 1
                        self.on_wait(f"{backend.model}: {reason}, повтор через {5 * overloaded} с")
                        time.sleep(5 * overloaded)
                        continue
                    if self._switch_model(reason, used_model):
                        overloaded = 0
                        continue
                    if reason == "исчерпана дневная квота":
                        raise QuotaExhausted(
                            f"Исчерпана дневная квота модели {backend.model} и резервных моделей. "
                            "Укажите другую модель (--model) или используйте кэш (--replay)."
                        ) from err
                    raise LLMError(f"{backend.model}: {reason}; резервные модели тоже недоступны.") from err
                if not backend.retryable(err) or attempts >= 6:
                    raise LLMError(f"Ошибка вызова модели {backend.model}: {err}") from err
                delay = _retry_delay(err) or min(60.0, 5.0 * 2 ** (attempts - 1))
                self.on_wait(f"лимит запросов/ошибка сервиса, повтор через {delay:.0f} с (попытка {attempts + 1})")
                time.sleep(delay)

    def trace(self) -> list[dict]:
        return [asdict(c) for c in self.calls]
