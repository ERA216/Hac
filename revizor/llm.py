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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# лимит выходных токенов (у моделей Gemini 3 в него входят и «размышления»): судья на большом ТЗ
# пишет длинный реестр — с запасом, чтобы JSON не обрывался
MAX_OUTPUT_TOKENS = 65536


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


class CacheMiss(LLMError):
    pass


class FatalLLMError(LLMError):
    """Ошибка настройки (неверный ключ, нет доступа, сервер модели не запущен).

    Продолжать прогон бессмысленно: оркестратор останавливается с понятным сообщением,
    а не выпускает отчёт, собранный одними правилами.
    """


def _fatal_reason(err: Exception) -> str | None:
    import urllib.error

    s = str(err)
    code = _status(err)
    if (
        re.search(r"API_KEY_INVALID|API key not valid|API key expired|invalid[_ ]api[_ ]key|Incorrect API key", s, re.I)
        or code == 401
    ):
        return "ключ API недействителен или просрочен — проверьте GEMINI_API_KEY / OPENAI_API_KEY в .env"
    if isinstance(err, urllib.error.URLError) and not isinstance(err, urllib.error.HTTPError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, ConnectionRefusedError) or re.search(r"refused|10061|actively refused", str(reason), re.I):
            return "сервер модели не отвечает (соединение отклонено) — запустите Ollama или проверьте OLLAMA_BASE_URL"
    return None


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
            return {k: walk(v) for k, v in node.items() if not (k in ("title", "default") and not isinstance(v, dict))}
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
    if code == 403 or "PERMISSION_DENIED" in str(err):
        return "нет доступа к модели"
    if code == 504 or "DEADLINE_EXCEEDED" in str(err) or isinstance(err, TimeoutError):
        # модель не успела ответить за таймаут запроса — повтор на ней же снова займёт минуты
        return "модель не ответила вовремя (504)"
    if code == 503 or re.search(r"high demand|overloaded|UNAVAILABLE", str(err)):
        return "модель перегружена (503)"
    if code == 429 and _is_daily_quota(err):
        return "исчерпана дневная квота"
    return None


# ------------------------------------------------------------------ backends
# таймаут одного запроса к Gemini: самый долгий вызов (нарезчик, судья) занимает 1–2 минуты; запас ×2,5
GEMINI_TIMEOUT_MS = 300_000


def _gemini_client(api_key: str, timeout_ms: int = GEMINI_TIMEOUT_MS):
    """Клиент Gemini без внутренних повторов SDK.

    SDK по умолчанию повторяет запрос до 5 раз с паузами до минуты (в том числе при 429 и 503) —
    это прячет исчерпанную квоту и перегрузку, и переход на резервную модель занимал минуты.
    Повторы и резервные модели делает LLMClient: коротко и с понятными сообщениями.
    """
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1)),
    )


# предполётная проверка: модель, не ответившая на «ok» за это время, сейчас перегружена
PREFLIGHT_TIMEOUT_MS = 20_000


class _Gemini:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float | None, thinking_level: str | None):
        if not api_key:
            raise LLMError("Не задан GEMINI_API_KEY (см. README, раздел «Ключ LLM»).")
        self._api_key = api_key
        self.client = _gemini_client(api_key)
        self.model = model
        self.temperature = temperature
        # «low» / «LOW» / «Low» → значение SDK; неизвестное значение игнорируем (модель решает сама)
        level = (thinking_level or "").strip().upper()
        valid = {lv.value for lv in __import__("google.genai.types", fromlist=["ThinkingLevel"]).ThinkingLevel}
        self.thinking_level = level if level in valid and level != "THINKING_LEVEL_UNSPECIFIED" else None
        self.schema_supported = True

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, dict]:
        from google.genai import errors, types

        if not self.schema_supported:
            user = user + "\n\nВерни JSON строго по схеме:\n" + json.dumps(inline_schema(schema), ensure_ascii=False)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=self.temperature,
            thinking_config=(
                types.ThinkingConfig(thinking_level=types.ThinkingLevel(self.thinking_level)) if self.thinking_level else None
            ),
            response_json_schema=inline_schema(schema) if self.schema_supported else None,
        )
        try:
            resp = self.client.models.generate_content(model=self.model, contents=user, config=config)
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
        reason = resp.candidates[0].finish_reason if resp.candidates else None
        meta["truncated"] = "MAX_TOKENS" in str(reason)
        text = resp.text
        if not text:
            raise LLMError(f"Пустой ответ модели (finish_reason={reason or 'нет кандидатов'})")
        return text, meta

    @staticmethod
    def retryable(err: Exception) -> bool:
        from google.genai import errors

        if isinstance(err, errors.ServerError):
            return True
        return isinstance(err, errors.ClientError) and err.code == 429

    def ping(self) -> None:
        """Короткий запрос с малым таймаутом: отвечает ли модель прямо сейчас."""
        from google.genai import types

        client = getattr(self, "_ping_client", None) or _gemini_client(self._api_key, PREFLIGHT_TIMEOUT_MS)
        self._ping_client = client
        client.models.generate_content(model=self.model, contents="ok", config=types.GenerateContentConfig(max_output_tokens=16))

    @staticmethod
    def list_models(api_key: str) -> list[str]:
        client = _gemini_client(api_key)
        out = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                if m.name:
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
            messages[1]["content"] += "\n\nВерни JSON строго по схеме:\n" + json.dumps(inline_schema(schema), ensure_ascii=False)
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
            "truncated": resp.choices[0].finish_reason == "length",
        }
        return resp.choices[0].message.content or "", meta

    @staticmethod
    def retryable(err: Exception) -> bool:
        code = getattr(err, "status_code", None)
        return code == 429 or (isinstance(code, int) and code >= 500)


class _Ollama:
    """Локальная модель через родной API Ollama: размер контекста, JSON-схема, без «размышлений».

    Данные не покидают контур: подходит для ТЗ, которые нельзя отправлять во внешние сервисы.
    """

    name = "ollama"

    def __init__(self, model: str, base_url: str | None, temperature: float | None, num_ctx: int):
        self.model = model
        base = (base_url or "http://localhost:11434").rstrip("/")
        self.base = base[:-3] if base.endswith("/v1") else base
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think_param = True

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, dict]:
        import urllib.error
        import urllib.request

        options: dict = {"num_ctx": self.num_ctx}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": inline_schema(schema),
            "stream": False,
            "options": options,
        }
        if self.think_param:
            body["think"] = False
        req = urllib.request.Request(
            f"{self.base}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code == 400 and self.think_param and "think" in detail.lower():
                self.think_param = False  # модель не поддерживает параметр think
                return self.complete(system, user, schema)
            err = LLMError(f"Ollama {e.code}: {detail}")
            err.status_code = e.code  # type: ignore[attr-defined]
            raise err from e
        meta = {
            "model_version": data.get("model", self.model),
            "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
            "output_tokens": data.get("eval_count", 0) or 0,
            "truncated": data.get("done_reason") == "length",
        }
        return data.get("message", {}).get("content", ""), meta

    @staticmethod
    def retryable(err: Exception) -> bool:
        import urllib.error

        code = getattr(err, "status_code", None)
        return isinstance(err, (urllib.error.URLError, TimeoutError)) or (isinstance(code, int) and code >= 500)


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
        num_ctx: int = 24576,
    ):
        self.provider = provider
        self.model = model
        # резервные модели: при перегрузке/недоступности/исчерпании квоты переключаемся по цепочке
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.switches: list[str] = []
        self.cache_dir = cache_dir
        self.extra_cache_dirs = [d for d in (extra_cache_dirs or []) if d and d != cache_dir]
        self.num_ctx = num_ctx  # окно контекста для локальной модели (Ollama)
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
            elif self.provider == "ollama":
                self._backend = _Ollama(self.model, base_url, temperature, self.num_ctx)
            else:
                raise LLMError(f"Неизвестный провайдер: {self.provider}")
        return self._backend

    @property
    def real_calls(self) -> int:
        return sum(1 for c in self.calls if not c.cached)

    @property
    def model_versions(self) -> list[str]:
        return sorted({c.model_version for c in self.calls if c.model_version})

    @property
    def current_model(self) -> str:
        """Модель, которая отвечает сейчас: после предполётной проверки или переключения это может быть резервная."""
        return self._backend.model if self._backend is not None else self.model

    def _cache_key(self, system: str, user: str, schema: type[BaseModel], model: str | None = None) -> str:
        """Ключ кэша — по модели, которая отвечает (а не по указанной в настройках): иначе ответ резервной
        модели, полученный при перегрузке основной, подставлялся бы потом вместо ответа основной."""
        model = model or self.current_model
        payload = json.dumps([self.provider, model, system, user, inline_schema(schema)], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _prompt_key(self, system: str, user: str, schema: type[BaseModel]) -> str:
        """Ключ без имени модели: --replay находит ответ, даже если в .env указана другая модель."""
        payload = json.dumps([self.provider, system, user, inline_schema(schema)], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _find_by_prompt(self, prompt_key: str) -> Path | None:
        for d in [self.cache_dir, *self.extra_cache_dirs]:
            if not d or not d.exists():
                continue
            for p in sorted(d.glob("*.json")):
                try:
                    if json.loads(p.read_text(encoding="utf-8")).get("prompt_key") == prompt_key:
                        return p
                except (OSError, json.JSONDecodeError):
                    continue
        return None

    def generate(self, agent: str, system: str, user: str, schema: type[T]) -> T:
        key = self._cache_key(system, user, schema)
        prompt_key = self._prompt_key(system, user, schema)
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        # дополнительные каталоги кэша только для чтения (например, ответы финального прогона в репозитории)
        found = next((p for p in [path, *(d / f"{key}.json" for d in self.extra_cache_dirs)] if p and p.exists()), None)
        if found is None and self.replay:
            found = self._find_by_prompt(prompt_key)
        if found:
            try:
                data = json.loads(found.read_text(encoding="utf-8"))
                obj = schema.model_validate_json(extract_json(data["text"]))
                self.calls.append(
                    CallRecord(
                        agent=agent,
                        model=data.get("model") or self.model,
                        cached=True,
                        seconds=0.0,
                        **{k: data["meta"].get(k, 0) for k in ("prompt_tokens", "output_tokens", "thoughts_tokens")},
                        model_version=data["meta"].get("model_version", self.model),
                    )
                )
                return obj
            except (ValidationError, KeyError, TypeError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass  # битый или нечитаемый файл кэша — запрашиваем модель заново
        if self.replay:
            raise CacheMiss(f"Режим --replay: нет сохранённого ответа для агента «{agent}».")
        if self.real_calls >= self.max_calls:
            raise BudgetExceeded(f"Превышен бюджет вызовов LLM ({self.max_calls}). Увеличьте --max-calls.")

        started = time.monotonic()
        text, meta, attempts = self._call(system, user, schema)
        try:
            obj = schema.model_validate_json(extract_json(text))
        except ValidationError as err:
            if meta.get("truncated"):
                # ответ упёрся в лимит длины — повтор того же запроса оборвётся так же; просим короче
                repair = (
                    user + "\n\nТвой предыдущий ответ был обрезан: он превысил лимит длины. Верни JSON заново и короче: "
                    "problem и recommendation — по 1–2 предложения, rewrite — только где он действительно нужен, "
                    "в списках оставь только самые важные элементы. JSON должен быть полным и корректным."
                )
            else:
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
            # ответила, возможно, уже резервная модель — сохраняем под её ключом
            path = path.with_name(f"{self._cache_key(system, user, schema, record.model)}.json")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {"agent": agent, "model": record.model, "prompt_key": prompt_key, "text": text, "meta": meta},
                        ensure_ascii=False,
                        indent=1,
                    ),
                    encoding="utf-8",
                )
            except OSError as err:  # ответ модели уже получен (и оплачен квотой) — не теряем его из-за кэша
                self.on_wait(f"не удалось сохранить ответ в кэш ({err.__class__.__name__}); прогон продолжается")
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
                fatal = _fatal_reason(err)
                if fatal:
                    raise FatalLLMError(f"{backend.model}: {fatal}.") from err
                reason = _unavailable_reason(err)
                if reason == "нет доступа к модели" and not self._switch_model(reason, used_model):
                    raise FatalLLMError(f"{backend.model}: нет доступа к модели (403) — проверьте ключ и права проекта.") from err
                if reason == "нет доступа к модели":
                    continue
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
                    if reason == "модель не найдена":  # ни одна модель из цепочки не существует — это настройка
                        raise FatalLLMError(
                            f"{backend.model}: модель не найдена, резервных нет — проверьте имя модели "
                            "(список: python -m revizor --list-models)."
                        ) from err
                    raise LLMError(f"{backend.model}: {reason}; резервные модели тоже недоступны.") from err
                if not backend.retryable(err) or attempts >= 6:
                    raise LLMError(f"Ошибка вызова модели {backend.model}: {err}") from err
                delay = _retry_delay(err) or min(60.0, 5.0 * 2 ** (attempts - 1))
                self.on_wait(f"лимит запросов/ошибка сервиса, повтор через {delay:.0f} с (попытка {attempts + 1})")
                time.sleep(delay)

    def preflight(self) -> str | None:
        """Выбрать первую модель цепочки, которая отвечает прямо сейчас (Gemini).

        При перегрузке основная модель не отвечает до таймаута запроса (минуты) — без проверки агент
        простаивал бы так на каждой перегруженной модели. Здесь каждой модели даётся «ok» с таймаутом
        20 с. Если не ответила ни одна — остаётся основная модель, дальше работают обычные повторы.
        """
        if self.replay or self.provider != "gemini":
            return None
        backend = self.backend
        if not hasattr(backend, "ping"):
            return None
        chain = [self.model, *[m for m in self.fallback_models if m != self.model]]
        for model in chain:
            backend.model = model
            try:
                backend.ping()
            except Exception as err:  # noqa: BLE001 — любая ошибка = модель сейчас не годится
                fatal = _fatal_reason(err)
                if fatal:
                    raise FatalLLMError(f"{model}: {fatal}.") from err
                reason = _unavailable_reason(err) or "не отвечает"
                self.on_wait(f"предполётная проверка: {model} — {reason}")
                if model not in self.switches:
                    self.switches.append(model)
                continue
            if model != self.model:
                self.on_wait(f"предполётная проверка: работаем на {model}")
            return model
        backend.model = self.model
        self.switches.clear()
        return None

    def trace(self) -> list[dict]:
        return [asdict(c) for c in self.calls]
