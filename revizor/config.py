"""Настройки LLM из окружения (.env) — общие для CLI и веб-интерфейса."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from .llm import LLMClient

ROOT = Path(__file__).resolve().parent.parent
REPLAY_CACHE = ROOT / "reports" / "llm-cache"
DEFAULT_MODELS = {"gemini": "gemini-3.6-flash", "openai": "gpt-4o-mini", "ollama": "qwen3:4b"}
DEFAULT_FALLBACKS = {
    "gemini": "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash",
    "openai": "",
}


# ключ из переменной окружения до чтения .env: у неё приоритет над .env, как у load_dotenv
_SHELL_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")


def _env_files() -> list[Path]:
    return [Path.cwd() / ".env", ROOT / ".env"]


def load_env() -> None:
    for path in _env_files():
        load_dotenv(path)


def usable_key(key: str | None) -> str:
    """Ключ без пробелов; пустая строка, если ключа нет или в .env осталась заглушка из .env.example."""
    key = (key or "").strip()
    return "" if not key or key.startswith("your_") else key


def gemini_key(entered: str = "") -> str:
    """Ключ Gemini для веб-интерфейса: поле ввода → переменная окружения → .env.

    .env перечитывается при каждом вызове, поэтому ключ, вписанный в файл при запущенном интерфейсе,
    подхватывается без перезапуска.
    """
    if key := usable_key(entered) or usable_key(_SHELL_GEMINI_KEY):
        return key
    for path in _env_files():
        if path.exists() and (key := usable_key(dotenv_values(path).get("GEMINI_API_KEY"))):
            return key
    return ""


def llm_from_env(
    provider: str | None = None,
    model: str | None = None,
    fallback_models: str | None = None,
    base_url: str | None = None,
    cache_dir: Path | None = ROOT / ".cache" / "llm",
    replay: bool = False,
    max_calls: int | None = None,
    rpm: float | None = None,
    on_wait: Callable[[str], None] | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """api_key — ключ, введённый в веб-интерфейсе; без него ключ берётся из окружения (.env)."""
    provider = provider or os.getenv("LLM_PROVIDER") or "gemini"  # пустая переменная окружения — тоже gemini
    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or ""
        model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
        fallbacks = fallback_models or os.getenv("GEMINI_FALLBACK_MODELS") or DEFAULT_FALLBACKS["gemini"]
    elif provider == "ollama":  # локальная модель: данные не покидают контур, лимита частоты нет
        key = ""
        model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODELS["ollama"]
        fallbacks = fallback_models or os.getenv("OLLAMA_FALLBACK_MODELS") or ""
        base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        rpm = 0 if rpm is None else rpm
    else:
        key = api_key or os.getenv("OPENAI_API_KEY") or ""
        model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
        fallbacks = fallback_models or os.getenv("OPENAI_FALLBACK_MODELS") or ""
    temperature = os.getenv("LLM_TEMPERATURE")
    return LLMClient(
        provider=provider,
        model=model,
        api_key=key,
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        cache_dir=cache_dir,
        replay=replay,
        rpm=rpm if rpm is not None else float(os.getenv("LLM_RPM", "5")),
        max_calls=max_calls if max_calls is not None else int(os.getenv("LLM_MAX_CALLS", "12")),
        temperature=float(temperature) if temperature else None,
        thinking_level=os.getenv("GEMINI_THINKING_LEVEL") or None,
        on_wait=on_wait,
        fallback_models=[m.strip() for m in fallbacks.split(",") if m.strip()],
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "24576")),
        # ответы модели финального прогона, сохранённые в репозитории: --replay работает сразу после клонирования
        # (только в режиме --replay: обычный запуск всегда обращается к модели)
        extra_cache_dirs=[REPLAY_CACHE] if replay else [],
    )
