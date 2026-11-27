"""Слова со смешанной раскладкой в итоговом тексте отчёта."""

from revizor.document import load_document
from revizor.orchestrator import Pipeline
from revizor.textfix import fix_text

from .fake_llm import FakeLLM
from .test_pipeline import DOC


def test_latin_lookalikes_inside_russian_words_are_fixed():
    assert fix_text("Явно указать, является ли сменa порога") == "Явно указать, является ли смена порога"
    assert fix_text("Тестирoвание") == "Тестирование"


def test_latin_words_and_identifiers_are_kept():
    text = "push-уведомление, SMS-код, cardId, GET /alerts/saveLimit, limit_kzt, R-05, ISO/IEC 29148, Ф-3"
    assert fix_text(text) == text


class MixedScript(FakeLLM):
    """Модель пишет «сменa» с латинской «a» и в формулировке, и (ошибочно) в цитате."""

    def generate(self, agent, system, user, schema):
        out = super().generate(agent, system, user, schema)
        if agent == "judge":
            out.findings[0].title = "Сменa порога"
        return out


def test_report_text_is_fixed_but_prompts_and_quotes_are_not():
    llm = MixedScript()
    state = Pipeline(llm).run(load_document(DOC))
    assert any(f.title == "Смена порога" for f in state.findings)
    assert "Сменa порога" in llm.prompts["clarifier"]  # агенты видят ответ как есть: кэш и --replay не ломаются
    state.findings[0].evidence[0].quote = "сменa"
    from revizor.orchestrator import polish_texts

    polish_texts(state)
    assert state.findings[0].evidence[0].quote == "сменa"  # цитата — буква в букву как в документе


def test_cli_and_app_answers_give_the_same_recheck_prompt():
    # CLI (--answers) применяет ответы внутри прогона, интерфейс — после него. Промпт перепроверки должен
    # совпадать: тогда сохранённый ответ (reports/llm-cache) находится и в демо без ключа
    answers = {"Q-01": "1", "Q-02": "свой ответ"}
    cli = MixedScript()
    Pipeline(cli, ask_answers=lambda qs: answers).run(load_document(DOC))
    app = MixedScript()
    pipe = Pipeline(app)
    state = pipe.run(load_document(DOC))
    pipe.apply_answers(state, answers)
    assert cli.prompts["recheck"] == app.prompts["recheck"]
    assert "Смена порога" in app.prompts["recheck"]  # исправленный текст отчёта
