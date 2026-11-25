"""CLI: python -m revizor <файл требований> <файл отчёта> [опции]."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from . import __version__
from .document import SUPPORTED, load_document
from .llm import LLMClient, LLMError, _Gemini
from .orchestrator import Pipeline
from .report import write_reports
from .schemas import Question

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = {"gemini": "gemini-flash-latest", "openai": "gpt-4o-mini"}


class ConsoleProgress:
    def __init__(self, console: Console):
        self.console = console
        self._status = None

    def start(self, title):
        self._status = self.console.status(f"[bold cyan]{title}…")
        self._status.start()

    def _stop(self):
        if self._status:
            self._status.stop()
            self._status = None

    def done(self, title, note):
        self._stop()
        self.console.print(f"[green]✔[/] {title}: {note}")

    def warn(self, message):
        self._stop()
        self.console.print(f"[yellow]⚠ {message}[/]")


def _ask_in_console(console: Console):
    def ask(questions: list[Question]) -> dict[str, str]:
        console.print(Panel.fit(
            "Агент подготовил уточняющие вопросы. Выберите вариант цифрой, введите свой ответ "
            "или нажмите Enter, чтобы пропустить.", title="Уточнения", border_style="cyan"))
        answers = {}
        for q in questions:
            mark = "[red]⛔ блокирующий[/] " if q.blocking else ""
            console.print(f"\n[bold]{q.qid}[/] {mark}[dim]({q.addressee})[/]\n{q.question}")
            for i, o in enumerate(q.options, start=1):
                console.print(f"  [cyan]{i}[/]. {o}")
            raw = Prompt.ask("Ответ", default="", show_default=False).strip()
            if raw.isdigit() and 1 <= int(raw) <= len(q.options):
                raw = q.options[int(raw) - 1]
            if raw:
                answers[q.qid] = raw
        return answers

    return ask


def _answers_from_file(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        answers = {k: str(v) for k, v in data.items()}
    except json.JSONDecodeError:  # формат «Q-01: ответ» построчно
        answers = {m.group(1): m.group(2).strip() for m in re.finditer(r"^(Q-\d+)\s*[:.)-]\s*(.+)$", text, re.M)}

    def ask(questions: list[Question]) -> dict[str, str]:
        out = {}
        for q in questions:
            a = answers.get(q.qid, "")
            if a.isdigit() and 1 <= int(a) <= len(q.options):
                a = q.options[int(a) - 1]
            if a:
                out[q.qid] = a
        return out

    return ask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m revizor",
        description="Ревизор ТЗ — мультиагентная проверка документа требований (ISO/IEC/IEEE 29148, IEEE 830).",
    )
    p.add_argument("input", nargs="?", help=f"файл требований ({', '.join(sorted(SUPPORTED))})")
    p.add_argument("report", nargs="?", help="путь к файлу отчёта (.md); рядом будет .json")
    p.add_argument("--provider", choices=["gemini", "openai"], default=os.getenv("LLM_PROVIDER", "gemini"),
                   help="провайдер LLM (openai — любой OpenAI-совместимый API, включая Ollama)")
    p.add_argument("--model", help="модель (по умолчанию из .env)")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"), help="адрес OpenAI-совместимого API")
    p.add_argument("--interactive", "-i", action="store_true", help="задать уточняющие вопросы в консоли и перепроверить")
    p.add_argument("--answers", type=Path, help="файл ответов на вопросы (JSON {\"Q-01\": \"…\"} или строки «Q-01: …»)")
    p.add_argument("--replay", action="store_true", help="только сохранённые ответы модели (без сети и ключа)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "llm")
    p.add_argument("--max-calls", type=int, default=int(os.getenv("LLM_MAX_CALLS", "12")),
                   help="бюджет новых вызовов LLM на прогон (по умолчанию 12)")
    p.add_argument("--rpm", type=float, default=float(os.getenv("LLM_RPM", "5")),
                   help="ограничение запросов в минуту (бесплатный Gemini — 5)")
    p.add_argument("--list-models", action="store_true", help="показать доступные модели и выйти")
    p.add_argument("--version", action="version", version=f"revizor {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(ROOT / ".env")
    args = build_parser().parse_args(argv)
    console = Console()

    if args.provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODELS["openai"]

    if args.list_models:
        if args.provider != "gemini":
            console.print("[yellow]--list-models поддерживается для Gemini.[/]")
            return 2
        for name in _Gemini.list_models(api_key):
            console.print(name)
        return 0

    if not args.input or not args.report:
        build_parser().print_usage()
        console.print("[red]Укажите файл требований и путь к отчёту.[/]")
        return 2

    temperature = os.getenv("LLM_TEMPERATURE")
    llm = LLMClient(
        provider=args.provider,
        model=model,
        api_key=api_key,
        base_url=args.base_url,
        cache_dir=None if args.no_cache else args.cache_dir,
        replay=args.replay,
        rpm=args.rpm,
        max_calls=args.max_calls,
        temperature=float(temperature) if temperature else None,
        thinking_level=os.getenv("GEMINI_THINKING_LEVEL") or None,
        on_wait=lambda msg: console.print(f"[dim]… {msg}[/]"),
    )

    try:
        doc = load_document(args.input)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return 2

    console.print(Panel.fit(
        f"[bold]Ревизор ТЗ[/] {__version__}\nДокумент: [cyan]{doc.title}[/] ({doc.path.name}, {len(doc.lines)} строк)\n"
        f"Модель: {args.provider} / {model}" + (" [dim](replay)[/]" if args.replay else ""),
        border_style="cyan"))

    ask = None
    if args.answers:
        ask = _answers_from_file(args.answers)
    elif args.interactive:
        ask = _ask_in_console(console)

    pipeline = Pipeline(llm, progress=ConsoleProgress(console), ask_answers=ask)
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
    console.print(Panel.fit(
        f"Требований: {len(state.requirements)} · дефектов: {len(state.open_findings)} "
        f"(🔴 {state.count('critical')} · 🟠 {state.count('major')} · 🟡 {state.count('minor')}) · "
        f"вопросов: {len(state.questions)}\nИтог: {verdict}\n"
        f"Вызовов LLM: {llm.real_calls} новых, {len(llm.calls) - llm.real_calls} из кэша\n"
        + "\n".join(f"Отчёт: {p}" for p in paths),
        title="Готово", border_style="green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
