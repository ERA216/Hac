"""Настройки LLM из окружения (.env) — общие для CLI и веб-интерфейса."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from .llm import LLMClient

ROOT = Path(__file__).resolve().parent.parent
REPLAY_CACHE = ROOT / "reports" / "llm-cache"
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "openai": "gpt-4o-mini"}
DEFAULT_FALLBACKS = {
    "gemini": "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash",
    "openai": "",
}


def load_env() -> None:
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(ROOT / ".env")


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
) -> LLMClient:
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
        fallbacks = fallback_models or os.getenv("GEMINI_FALLBACK_MODELS") or DEFAULT_FALLBACKS["gemini"]
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
        fallbacks = fallback_models or os.getenv("OPENAI_FALLBACK_MODELS") or ""
    temperature = os.getenv("LLM_TEMPERATURE")
    return LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        cache_dir=cache_dir,
        replay=replay,
        rpm=rpm if rpm is not None else float(os.getenv("LLM_RPM", "5")),
        max_calls=max_calls if max_calls is not None else int(os.getenv("LLM_MAX_CALLS", "12")),
        temperature=float(temperature) if temperature else None,
        thinking_level=os.getenv("GEMINI_THINKING_LEVEL") or None,
        on_wait=on_wait,
        fallback_models=[m.strip() for m in fallbacks.split(",") if m.strip()],
        # ответы модели финального прогона, сохранённые в репозитории: --replay работает сразу после клонирования
        # (только в режиме --replay: обычный запуск всегда обращается к модели)
        extra_cache_dirs=[REPLAY_CACHE] if replay else [],
    )
