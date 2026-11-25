"""Проверка окружения: python -m revizor --check.

Не тратит квоту генерации: модели проверяются по списку доступных моделей.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import urllib.request

from rich.console import Console
from rich.table import Table

from .config import DEFAULT_FALLBACKS, DEFAULT_MODELS, REPLAY_CACHE, ROOT

OK, WARN, FAIL = "[green]✔[/]", "[yellow]![/]", "[red]✘[/]"


def _row(table: Table, status: str, what: str, detail: str) -> None:
    table.add_row(status, what, detail)


def run_check(console: Console) -> int:
    table = Table(title="Ревизор ТЗ — проверка окружения", show_lines=False)
    table.add_column("", width=2)
    table.add_column("Проверка")
    table.add_column("Результат")
    critical = 0

    v = sys.version_info
    if v >= (3, 11):
        _row(table, OK, "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        _row(table, FAIL, "Python", f"{v.major}.{v.minor} — нужен 3.11 или новее")
        critical += 1

    for mod, name, required in [
        ("google.genai", "google-genai", True),
        ("pydantic", "pydantic", True),
        ("pymupdf4llm", "pymupdf4llm (PDF)", True),
        ("docx", "python-docx (DOCX)", True),
        ("rapidfuzz", "rapidfuzz (проверка цитат)", True),
        ("jinja2", "Jinja2", True),
        ("rich", "rich", True),
        ("dotenv", "python-dotenv", True),
        ("streamlit", "streamlit (веб-интерфейс)", False),
    ]:
        try:
            importlib.import_module(mod)
            _row(table, OK, name, "установлен")
        except Exception as e:  # noqa: BLE001
            _row(
                table,
                FAIL if required else WARN,
                name,
                f"не загружается: {e.__class__.__name__}. pip install -r requirements.txt",
            )
            critical += required

    env_path = ROOT / ".env"
    _row(
        table,
        OK if env_path.exists() else WARN,
        "Файл .env",
        "найден" if env_path.exists() else "нет — скопируйте .env.example в .env",
    )

    provider = os.getenv("LLM_PROVIDER", "gemini")
    _row(table, OK, "Провайдер", provider)
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key or key.startswith("your_"):
            _row(table, FAIL, "GEMINI_API_KEY", "не задан (без ключа работает только --replay)")
            critical += 1
        else:
            _row(table, OK, "GEMINI_API_KEY", f"задан (длина {len(key)})")
            model = os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
            chain = [model] + [
                m.strip() for m in (os.getenv("GEMINI_FALLBACK_MODELS") or DEFAULT_FALLBACKS["gemini"]).split(",") if m.strip()
            ]
            try:
                from .llm import _Gemini

                available = set(_Gemini.list_models(key))
                present = [m for m in chain if m in available]
                missing = [m for m in chain if m not in available]
                if model in available:
                    _row(table, OK, "Основная модель", model)
                else:
                    _row(table, WARN if present else FAIL, "Основная модель", f"{model} недоступна ключу")
                    critical += 0 if present else 1
                _row(
                    table,
                    OK if present else FAIL,
                    "Цепочка моделей",
                    f"доступно {len(present)} из {len(chain)}" + (f"; нет: {', '.join(missing)}" if missing else ""),
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                bad_key = "API_KEY_INVALID" in msg or "API key not valid" in msg
                _row(table, FAIL, "Доступ к Gemini", "ключ недействителен" if bad_key else f"ошибка: {msg[:120]}")
                critical += 1
    elif provider == "ollama":
        base = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/").removesuffix("/v1")
        model = os.getenv("OLLAMA_MODEL") or DEFAULT_MODELS["ollama"]
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
                names = {m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])}
            _row(table, OK, "Ollama", f"отвечает: {base}")
            _row(
                table,
                OK if model in names else FAIL,
                "Модель Ollama",
                model if model in names else f"{model} не скачана — ollama pull {model}",
            )
            critical += model not in names
        except Exception as e:  # noqa: BLE001
            _row(table, FAIL, "Ollama", f"не отвечает по адресу {base} ({e.__class__.__name__}) — запустите Ollama")
            critical += 1
    else:
        key = os.getenv("OPENAI_API_KEY", "")
        _row(table, OK if key else WARN, "OPENAI_API_KEY", "задан" if key else "не задан (нужен большинству провайдеров)")

    cached = len(list(REPLAY_CACHE.glob("*.json"))) if REPLAY_CACHE.exists() else 0
    _row(table, OK if cached else WARN, "Кэш для --replay", f"{cached} ответов модели в reports/llm-cache")

    reports = ROOT / "reports"
    try:
        reports.mkdir(exist_ok=True)
        probe = reports / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        _row(table, OK, "Запись отчётов", str(reports))
    except OSError as e:
        _row(table, FAIL, "Запись отчётов", f"нет доступа к {reports}: {e}")
        critical += 1

    console.print(table)
    if critical:
        console.print(f"[red]Есть проблемы: {critical}. Исправьте отмеченные ✘ и запустите проверку снова.[/]")
        return 1
    console.print("[green]Всё готово.[/] Запуск: [bold]python -m revizor examples/requirements.md reports/report.md[/]")
    return 0


if __name__ == "__main__":
    from .config import load_env

    load_env()
    sys.exit(run_check(Console()))
