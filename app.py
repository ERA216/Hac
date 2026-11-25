"""Веб-интерфейс «Ревизора ТЗ» для демонстрации: streamlit run app.py"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from revizor import __version__
from revizor.config import ROOT, llm_from_env, load_env
from revizor.document import SUPPORTED, load_document
from revizor.llm import LLMError
from revizor.orchestrator import Pipeline
from revizor.report import render_html_report, render_markdown, to_json

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
    model = st.text_input("Модель", value="", placeholder="из .env (gemini-3.8-flash)")
    use_cache = st.toggle("Кэш ответов модели", value=True, help="Повторная проверка того же документа не тратит квоту")
    replay = st.toggle("Только кэш (без сети)", value=False, help="Для демо без ключа: использовать сохранённые ответы")
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
elif uploaded is not None:
    tmp = Path(tempfile.gettempdir()) / f"revizor_upload{Path(uploaded.name).suffix.lower()}"
    tmp.write_bytes(uploaded.getvalue())
    st.session_state["doc_path"] = str(tmp)
    st.session_state["doc_name"] = uploaded.name

doc_path = st.session_state.get("doc_path")
if doc_path:
    st.info(f"Документ: **{st.session_state.get('doc_name') or Path(doc_path).name}**")


def make_pipeline(llm, box) -> Pipeline:
    ctx = get_script_run_ctx()  # потоки параллельных ревьюеров должны писать в тот же интерфейс
    return Pipeline(
        llm, progress=StatusProgress(box),
        thread_initializer=lambda: add_script_run_ctx(threading.current_thread(), ctx),
    )


def make_llm(box):
    return llm_from_env(
        model=model or None,
        cache_dir=(ROOT / ".cache" / "llm") if use_cache or replay else None,
        replay=replay,
        on_wait=lambda msg: box.write(f"⏳ {msg}"),
    )


if st.button("Проверить документ", type="primary", disabled=not doc_path):
    with st.status("Агенты анализируют документ…", expanded=True) as box:
        try:
            doc = load_document(doc_path)
            box.write(f"📄 Загружен документ «{doc.title}»: {len(doc.lines)} строк" + (f", {doc.pages} стр." if doc.pages else ""))
            llm = make_llm(box)
            state = make_pipeline(llm, box).run(doc)
            st.session_state["run"] = (state, llm)
            box.update(label="Проверка завершена", state="complete", expanded=False)
        except (LLMError, ValueError, FileNotFoundError) as e:
            box.update(label="Ошибка", state="error")
            st.error(str(e))

# ------------------------------------------------------------------ results
run = st.session_state.get("run")
if run:
    state, llm = run
    c = st.columns(3)
    c[0].metric("Требований", len(state.requirements))
    c[1].metric("Уточняющих вопросов", len(state.questions), f"блокирующих: {sum(q.blocking for q in state.questions)}", delta_color="off")
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
            with st.form("answers"):
                answers = {}
                for q in state.questions:
                    st.markdown(f"**{q.qid}** {'⛔ блокирующий · ' if q.blocking else ''}*{q.addressee}* · дефекты: {', '.join(q.finding_ids) or '—'}")
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
                            box.update(label="Ошибка", state="error")
                            st.error(str(e))

    with tab_report:
        # data:-URL — отдельный origin: отчёт (текст документа и ответы модели) изолирован от приложения
        b64 = base64.b64encode(render_html_report(state, llm).encode("utf-8")).decode("ascii")
        st.iframe(f"data:text/html;base64,{b64}", height=1100)

    with tab_dl:
        stem = Path(state.doc.path).stem
        st.download_button("Отчёт Markdown", render_markdown(state, llm), file_name=f"{stem}_report.md", mime="text/markdown")
        st.download_button("Отчёт HTML (интерактивный)", render_html_report(state, llm), file_name=f"{stem}_report.html", mime="text/html")
        st.download_button(
            "Данные JSON", json.dumps(to_json(state, llm), ensure_ascii=False, indent=2, default=str),
            file_name=f"{stem}_report.json", mime="application/json",
        )
