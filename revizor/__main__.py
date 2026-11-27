"""CLI: python -m revizor <файл требований> <файл отчёта> [опции]."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from . import __version__
from .config import ROOT, llm_from_env, load_env
from .document import SUPPORTED, load_document
from .llm import LLMError, _Gemini
from .orchestrator import Pipeline
from .report import write_reports
from .schemas import Question


class ConsoleProgress:
    """Одна строка-индикатор для всех активных шагов (ревьюеры работают параллельно)."""

    def __init__(self, console: Console):
        self.console = console
        self.active: list[str] = []
        self.status = None
        self.lock = threading.Lock()

    def _refresh(self):
        if self.active:
            label = "[bold cyan]Работают: " + ", ".join(self.active) + "…"
            if self.status is None:
                self.status = self.console.status(label)
                self.status.start()
            else:
                self.status.update(label)
        elif self.status is not None:
            self.status.stop()
            self.status = None

    def start(self, title):
        with self.lock:
            self.active.append(title)
            self._refresh()

    def _finish(self, title, line):
        with self.lock:
            if title in self.active:
                self.active.remove(title)
            if self.status is not None:
                self.status.stop()
                self.status = None
            self.console.print(line)
            self._refresh()

    def done(self, title, note):
        self._finish(title, f"[green]✔[/] {title}: {note}")

    def warn(self, message):
        title = message.split(":", 1)[0]
        self._finish(title, f"[yellow]⚠ {message}[/]")


def _ask_in_console(console: Console):
    def ask(questions: list[Question]) -> dict[str, str]:
        console.print(
            Panel.fit(
                "Агент подготовил уточняющие вопросы. Выберите вариант цифрой, введите свой ответ "
                "или нажмите Enter, чтобы пропустить.",
                title="Уточнения",
                border_style="cyan",
            )
        )
        answers = {}
        for q in questions:
            mark = "[red]⛔ блокирующий[/] " if q.blocking else ""
            console.print(f"\n[bold]{q.qid}[/] {mark}[dim]({q.addressee})[/]\n{q.question}")
            for i, o in enumerate(q.options, start=1):
                console.print(f"  [cyan]{i}[/]. {o}")
            raw = Prompt.ask("Ответ", default="", show_default=False).strip()
            if raw:
                answers[q.qid] = raw
        return answers

    return ask


def _answers_from_file(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        answers = {k: str(v) for k, v in json.loads(text).items()}
    except json.JSONDecodeError:  # формат «Q-01: ответ» построчно
        answers = {m.group(1): m.group(2).strip() for m in re.finditer(r"^(Q-\d+)\s*[:.)-]\s*(.+)$", text, re.M)}
    return lambda questions: answers


class _RussianHelp(argparse.RawDescriptionHelpFormatter):
    """Справка целиком по-русски: argparse по умолчанию пишет «usage:» и заголовки разделов по-английски."""

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, prefix="Запуск: ")


EXAMPLES = """примеры:
  python -m revizor examples/requirements.md reports/report.md
      проверить ТЗ; рядом с отчётом появятся .html, .json и .csv
  python -m revizor examples/requirements.md reports/report.md --replay
      пересобрать финальный отчёт из сохранённых ответов модели, без ключа и сети
  python -m revizor examples/requirements.md reports/report.md --answers reports/answers.json
      учесть ответы на уточняющие вопросы и перепроверить дефекты
  python -m revizor --check
      проверить окружение перед первым запуском (квоту не тратит)"""


_ARGPARSE_RU = [
    ("unrecognized arguments", "неизвестные параметры"),
    ("invalid int value", "нужно целое число, получено"),
    ("invalid float value", "нужно число, получено"),
    ("invalid choice", "недопустимое значение"),
    ("choose from", "допустимо"),
    ("expected one argument", "нужно значение"),
    ("argument ", "параметр "),
]


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        for en, ru in _ARGPARSE_RU:
            message = message.replace(en, ru)
        self.print_usage(sys.stderr)
        self.exit(2, f"Ошибка в параметрах: {message}\nСправка: python -m revizor --help\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="python -m revizor",
        description="Ревизор ТЗ — мультиагентная проверка документа требований (ISO/IEC/IEEE 29148, IEEE 830).",
        epilog=EXAMPLES,
        formatter_class=_RussianHelp,
        add_help=False,
    )
    p._positionals.title = "аргументы"
    p._optionals.title = "параметры"
    p.add_argument("-h", "--help", action="help", help="показать эту справку и выйти")
    p.add_argument("input", nargs="?", help=f"файл требований ({', '.join(sorted(SUPPORTED))})")
    p.add_argument("report", nargs="?", help="путь к отчёту (.md); рядом появятся .html, .json и .csv")
    p.add_argument(
        "--provider",
        choices=["gemini", "openai", "ollama"],
        default=None,
        help="провайдер LLM: gemini, openai (любой OpenAI-совместимый API) или ollama (локальная модель)",
    )
    p.add_argument("--model", help="модель (по умолчанию из .env)")
    p.add_argument("--fallback-models", help="резервные модели через запятую (при перегрузке, 404 или исчерпании квоты)")
    p.add_argument("--base-url", help="адрес OpenAI-совместимого API")
    p.add_argument("--interactive", "-i", action="store_true", help="задать уточняющие вопросы в консоли и перепроверить")
    p.add_argument("--answers", type=Path, help='файл ответов на вопросы (JSON {"Q-01": "…"} или строки «Q-01: …»)')
    p.add_argument("--replay", action="store_true", help="только сохранённые ответы модели (без сети и ключа)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "llm", help="каталог кэша ответов модели")
    p.add_argument("--max-calls", type=int, default=None, help="бюджет новых вызовов LLM на прогон (по умолчанию 12)")
    p.add_argument("--rpm", type=float, default=None, help="ограничение запросов в минуту (бесплатный Gemini — 5)")
    p.add_argument(
        "--parallel",
        type=int,
        default=int(os.getenv("LLM_PARALLEL", "4")),
        help="сколько ревьюеров работают одновременно (1 — последовательно)",
    )
    p.add_argument("--list-models", action="store_true", help="показать доступные модели и выйти")
    p.add_argument(
        "--check", action="store_true", help="проверить окружение: зависимости, ключ, модели, Ollama (квоту не тратит)"
    )
    p.add_argument("--version", action="version", version=f"revizor {__version__}", help="показать версию и выйти")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.check:
        from .check import run_check

        return run_check(console)

    if args.list_models:
        for name in _Gemini.list_models(os.getenv("GEMINI_API_KEY", "")):
            console.print(name)
        return 0

    if not args.input or not args.report:
        parser.print_usage()
        console.print("[red]Укажите файл требований и путь к отчёту.[/]")
        return 2

    llm = llm_from_env(
        provider=args.provider,
        model=args.model,
        fallback_models=args.fallback_models,
        base_url=args.base_url,
        cache_dir=None if args.no_cache else args.cache_dir,
        replay=args.replay,
        max_calls=args.max_calls,
        rpm=args.rpm,
        on_wait=lambda msg: console.print(f"[dim]… {msg}[/]"),
    )

    key = os.getenv("GEMINI_API_KEY", "") if llm.provider == "gemini" else "ok"
    if not args.replay and (not key or key.startswith("your_")):
        console.print(
            "[red]Не задан ключ GEMINI_API_KEY.[/] Скопируйте .env.example в .env и впишите ключ "
            "(см. README, раздел «Ключ LLM»).\nБез ключа можно воспроизвести сохранённый прогон: "
            "[bold]python -m revizor examples/requirements.md reports/replay.md --replay[/]"
        )
        return 2

    try:
        doc = load_document(args.input)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return 2

    console.print(
        Panel.fit(
            f"[bold]Ревизор ТЗ[/] {__version__}\nДокумент: [cyan]{doc.title}[/] ({doc.path.name}, {len(doc.lines)} строк)\n"
            f"Модель: {llm.provider} / {llm.model}" + (" [dim](replay)[/]" if args.replay else ""),
            border_style="cyan",
        )
    )

    ask = _answers_from_file(args.answers) if args.answers else _ask_in_console(console) if args.interactive else None
    pipeline = Pipeline(llm, progress=ConsoleProgress(console), ask_answers=ask, parallel=args.parallel)
    try:
        state = pipeline.run(doc)
    except LLMError as e:
        console.print(f"[red]Прогон остановлен: {e}[/]")
        return 3

    report_path = Path(args.report)
    paths = write_reports(state, llm, report_path)
    answered = {q.qid: q.answer for q in state.questions if q.answer}
    if answered:
        ans_path = report_path.with_suffix(".answers.json")
        ans_path.write_text(json.dumps(answered, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(ans_path)

    verdict = "[green]можно начинать разработку[/]" if state.ready_for_development else "[red]не готов к разработке[/]"
    resolved = sum(1 for f in state.findings if not f.is_open)
    console.print(
        Panel.fit(
            f"Требований: {len(state.requirements)} · дефектов: {len(state.open_findings)} "
            f"(🔴 {state.count('critical')} · 🟠 {state.count('major')} · 🟡 {state.count('minor')}) · "
            f"вопросов: {len(state.questions)}"
            + (f" · закрыто уточнениями: {resolved}" if resolved else "")
            + f"\nИтог: {verdict}\n"
            f"Вызовов LLM: {llm.real_calls} новых, {len(llm.calls) - llm.real_calls} из кэша\n"
            + "\n".join(f"Отчёт: {p}" for p in paths),
            title="Готово",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
