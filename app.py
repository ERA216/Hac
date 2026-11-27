"""Веб-интерфейс «Ревизора ТЗ» для демонстрации: streamlit run app.py"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from revizor import __version__
from revizor.config import ROOT, gemini_key, llm_from_env, load_env
from revizor.document import SUPPORTED, load_document
from revizor.llm import LLMError
from revizor.orchestrator import Pipeline
from revizor.report import render_html_report, render_markdown, to_json
from revizor.report.pdf import export_pdf, find_browser

load_env()
st.set_page_config(page_title="Ревизор ТЗ", page_icon="🔎", layout="wide")

EXAMPLE = ROOT / "examples" / "requirements.md"


class StatusProgress:
    """Живой журнал работы агентов внутри st.status."""

    def __init__(self, box):
        self.box = box

    def start(self, title):
        self.box.update(label=f"Работает: {title}…")

    def done(self, title, note):
        self.box.write(f"✅ **{title}** — {note}")

    def warn(self, message):
        self.box.write(f"⚠️ {message}")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Ревизор ТЗ")
    st.caption(f"Версия {__version__}. Проверка требований по ISO/IEC/IEEE 29148 и IEEE 830.")
    MODES = {
        "Gemini (облако)": "gemini",
        "Локальная модель Ollama": "ollama",
        "Только кэш (без сети и ключа)": "replay",
    }
    default_mode = "Локальная модель Ollama" if os.getenv("LLM_PROVIDER") == "ollama" else "Gemini (облако)"
    mode_label = st.radio(
        "Где работает модель",
        list(MODES),
        index=list(MODES).index(default_mode),
        help="Ollama: документ не покидает компьютер (контур банка). Кэш: сохранённые ответы, для демо без ключа.",
    )
    mode = MODES[mode_label]
    replay = mode == "replay"
    provider = "ollama" if mode == "ollama" else "gemini"
    placeholder = os.getenv("OLLAMA_MODEL", "qwen3:4b") if provider == "ollama" else os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    model = st.text_input("Модель", value="", placeholder=f"из .env ({placeholder})", disabled=replay)
    key_found = bool(gemini_key())
    entered_key = st.text_input(
        "Ключ Gemini",
        type="password",
        key="api_key_input",
        placeholder="уже задан — можно не заполнять" if key_found else "вставьте ключ из AI Studio",
        disabled=provider != "gemini" or replay,
        help="Ключ хранится только в памяти этой вкладки и никуда не записывается. Пустое поле — ключ из .env "
        "или переменной окружения GEMINI_API_KEY; правку .env интерфейс подхватывает без перезапуска.",
    )
    if provider == "gemini" and not replay and not key_found and not entered_key.strip():
        st.caption("Нет ключа? [Получить бесплатно в Google AI Studio](https://aistudio.google.com/apikey)")
    use_cache = st.toggle("Кэш ответов модели", value=True, help="Повторная проверка того же документа не тратит квоту")
    st.divider()
    st.markdown(
        "**Агенты:** нарезчик → правила → аналитик · тестировщик · архитектор · ИБ → "
        "согласованность → проверка цитат → судья → уточнения ⇄ человек → перепроверка"
    )

st.title("🔎 Ревизор ТЗ")
st.write(
    "Мультиагентная статическая проверка технического задания: агенты находят противоречия, "
    "непроверяемые и неполные требования, **задают уточняющие вопросы** и выпускают отчёт, "
    "где каждая цитата сверена с документом."
)

# ------------------------------------------------------------------ input
col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader("Документ требований", type=[s.lstrip(".") for s in sorted(SUPPORTED)])
with col2:
    st.write("")
    st.write("")
    use_example = st.button("Взять пример ТЗ хакатона", use_container_width=True)

if use_example:
    st.session_state["doc_path"] = str(EXAMPLE)
    st.session_state["doc_name"] = EXAMPLE.name
elif uploaded is not None and st.session_state.get("upload_id") != uploaded.file_id:
    # только новая загрузка меняет документ: иначе при любом следующем клике виджет с уже загруженным
    # файлом снова подменял бы выбранный пример
    st.session_state["upload_id"] = uploaded.file_id
    # у каждой сессии свой временный файл — одновременные пользователи не перезаписывают документы друг друга
    fd, tmp_name = tempfile.mkstemp(prefix="revizor_", suffix=Path(uploaded.name).suffix.lower())
    with os.fdopen(fd, "wb") as fh:
        fh.write(uploaded.getvalue())
    st.session_state["doc_path"] = tmp_name
    st.session_state["doc_name"] = uploaded.name

doc_path = st.session_state.get("doc_path")
if doc_path:
    st.info(f"Документ: **{st.session_state.get('doc_name') or Path(doc_path).name}**")


def make_pipeline(llm, box) -> Pipeline:
    ctx = get_script_run_ctx()  # потоки параллельных ревьюеров должны писать в тот же интерфейс
    return Pipeline(
        llm,
        progress=StatusProgress(box),
        thread_initializer=lambda: add_script_run_ctx(threading.current_thread(), ctx),
    )


def make_llm(box):
    return llm_from_env(
        provider=provider,
        model=model or None,
        cache_dir=(ROOT / ".cache" / "llm") if use_cache or replay else None,
        replay=replay,
        on_wait=lambda msg: box.write(f"⏳ {msg}"),
        api_key=gemini_key(entered_key) if provider == "gemini" else None,
    )


DEMO_ANSWERS = ROOT / "reports" / "answers.json"


def demo_answers(state) -> dict[str, str] | None:
    """Ответы демо-прогона — только для примера ТЗ хакатона (к другому документу они не относятся)."""
    if not DEMO_ANSWERS.exists() or state.doc.sha256 != load_document(EXAMPLE).sha256:
        return None
    try:
        return {k: str(v) for k, v in json.loads(DEMO_ANSWERS.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return None


def key_problem() -> str | None:
    """Нет ключа Gemini и не включён режим «только кэш» — объясняем сразу, а не после минуты ожидания."""
    if replay or provider != "gemini":
        return None
    if not gemini_key(entered_key):
        return (
            "Нет ключа Gemini. Вставьте ключ в поле «Ключ Gemini» слева (бесплатно: https://aistudio.google.com/apikey) "
            "или впишите GEMINI_API_KEY в .env — перезапуск не нужен. Без ключа выберите режим «Только кэш» "
            "и возьмите пример ТЗ хакатона."
        )
    return None


if st.button("Проверить документ", type="primary", disabled=not doc_path):
    if (problem := key_problem()) is not None:
        st.error(problem)
        st.stop()
    with st.status("Агенты анализируют документ…", expanded=True) as box:
        try:
            assert doc_path is not None  # кнопка неактивна, пока документ не выбран
            doc = load_document(doc_path)
            box.write(
                f"📄 Загружен документ «{doc.title}»: {len(doc.lines)} строк" + (f", {doc.pages} стр." if doc.pages else "")
            )
            llm = make_llm(box)
            state = make_pipeline(llm, box).run(doc)
            st.session_state["run"] = (state, llm)
            box.update(label="Проверка завершена", state="complete", expanded=False)
        except (LLMError, ValueError, FileNotFoundError) as e:
            box.update(label="Ошибка", state="error", expanded=True)  # причина сразу на виду, а не в свёрнутом блоке
            st.error(str(e))

# ------------------------------------------------------------------ results
run = st.session_state.get("run")
if run:
    state, llm = run
    c = st.columns(3)
    c[0].metric("Требований", len(state.requirements))
    c[1].metric(
        "Уточняющих вопросов", len(state.questions), f"блокирующих: {sum(q.blocking for q in state.questions)}", delta_color="off"
    )
    c[2].metric("Цитат подтверждено", f"{state.quotes_verified}/{state.quotes_checked}")
    c = st.columns(3)
    c[0].metric("🔴 Критичных", state.count("critical"))
    c[1].metric("🟠 Существенных", state.count("major"))
    c[2].metric("🟡 Замечаний", state.count("minor"))
    resolved = sum(1 for f in state.findings if not f.is_open)
    if resolved:
        st.info(f"Закрыто уточнениями: {resolved} дефектов")
    if state.ready_for_development:
        st.success("✅ Документ можно передавать в разработку")
    else:
        st.error("❌ Документ не готов к разработке — см. критичные дефекты и блокирующие вопросы")

    tab_q, tab_report, tab_dl = st.tabs(["💬 Уточняющие вопросы", "📋 Отчёт", "⬇️ Скачать"])

    with tab_q:
        if not state.questions:
            st.write("Вопросов нет.")
        else:
            st.write("Ответьте на вопросы — агент перепроверит связанные дефекты и закроет те, что сняты ответами.")
            demo = demo_answers(state)
            if demo and st.button(
                "Подставить ответы демо-прогона",
                help="Ответы из reports/answers.json. Перепроверка с ними сохранена в reports/llm-cache, "
                "поэтому работает и в режиме «Только кэш».",
            ):
                for q in state.questions:
                    a = demo.get(q.qid, "").strip()
                    if a.isdigit() and 1 <= int(a) <= len(q.options):
                        st.session_state[f"r_{q.qid}"], st.session_state[f"t_{q.qid}"] = q.options[int(a) - 1], ""
                    else:
                        st.session_state[f"r_{q.qid}"], st.session_state[f"t_{q.qid}"] = "— пропустить —", a
                st.rerun()
            with st.form("answers"):
                answers = {}
                for q in state.questions:
                    st.markdown(
                        f"**{q.qid}** {'⛔ блокирующий · ' if q.blocking else ''}*{q.addressee}* · "
                        f"дефекты: {', '.join(q.finding_ids) or '—'}"
                    )
                    st.write(q.question)
                    choice = st.radio("Вариант", ["— пропустить —", *q.options], key=f"r_{q.qid}", label_visibility="collapsed")
                    own = st.text_input("Свой ответ", key=f"t_{q.qid}", placeholder="или свой ответ")
                    st.caption(f"Зачем: {q.why}")
                    a = own.strip() or (choice if choice != "— пропустить —" else "")
                    if a:
                        answers[q.qid] = a
                    st.divider()
                submitted = st.form_submit_button("Перепроверить с ответами", type="primary")
            if submitted:
                if not answers:
                    st.warning("Нет ни одного ответа.")
                else:
                    with st.status("Агент перепроверки учитывает ответы…", expanded=True) as box:
                        try:
                            llm.on_wait = lambda msg: box.write(f"⏳ {msg}")
                            note = make_pipeline(llm, box).apply_answers(state, answers)
                            box.update(label=f"Перепроверка завершена: {note}", state="complete")
                            st.rerun()
                        except LLMError as e:
                            box.update(label="Ошибка", state="error", expanded=True)
                            st.error(str(e))

    with tab_report:
        # data:-URL — отдельный origin: отчёт (текст документа и ответы модели) изолирован от приложения
        b64 = base64.b64encode(render_html_report(state, llm).encode("utf-8")).decode("ascii")
        st.iframe(f"data:text/html;base64,{b64}", height=1100)

    with tab_dl:
        stem = Path(state.doc.path).stem
        st.download_button("Отчёт Markdown", render_markdown(state, llm), file_name=f"{stem}_report.md", mime="text/markdown")
        st.download_button(
            "Отчёт HTML (интерактивный)", render_html_report(state, llm), file_name=f"{stem}_report.html", mime="text/html"
        )
        st.download_button(
            "Данные JSON",
            json.dumps(to_json(state, llm), ensure_ascii=False, indent=2, default=str),
            file_name=f"{stem}_report.json",
            mime="application/json",
        )
        # PDF печатает браузер (несколько секунд) — по кнопке, а не при каждом обновлении страницы;
        # после перепроверки по ответам ключ меняется и PDF собирается заново
        pdf_key = (state.doc.sha256, len(state.findings), sum(1 for f in state.findings if not f.is_open))
        if find_browser():
            if st.button("Подготовить PDF"):
                with st.spinner("Печатаю PDF…"), tempfile.TemporaryDirectory() as tmp:
                    md_file = Path(tmp) / f"{stem}_report.md"
                    md_file.write_text(render_markdown(state, llm), encoding="utf-8")
                    try:
                        st.session_state["pdf"] = (pdf_key, export_pdf(md_file).read_bytes())
                    except RuntimeError as e:
                        st.error(str(e))
            saved = st.session_state.get("pdf")
            if saved and saved[0] == pdf_key:
                st.download_button("Отчёт PDF", saved[1], file_name=f"{stem}_report.pdf", mime="application/pdf")
        else:
            st.caption("Для PDF нужен установленный Chrome, Edge или Chromium.")
