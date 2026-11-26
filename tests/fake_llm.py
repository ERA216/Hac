"""Фейковый LLM для тестов: детерминированные ответы агентам без сети и ключа."""

from __future__ import annotations

from revizor.llm import CallRecord, LLMClient
from revizor.schemas import (
    ClarifyOut,
    JudgeOut,
    RecheckOut,
    ReviewOut,
    SegmentationOut,
    SetReviewOut,
)


def _seg():
    return SegmentationOut(
        document_title="Уведомление о расходах по карте",
        requirements=[
            dict(
                source_id="Ф-1",
                section="Функциональные требования",
                kind="functional",
                summary="Порог от 1 000 до 1 000 000 ₸",
                quote="Держатель активной карты KZT задаёт целый порог",
                line_start=37,
                line_end=37,
            ),
            dict(
                source_id="Ф-2",
                section="Функциональные требования",
                kind="functional",
                summary="Месяц по дате покупки в Алматы",
                quote="Месяц определяется по дате совершения покупки в Алматы",
                line_start=38,
                line_end=38,
            ),
            dict(
                source_id="Ф-3",
                section="Функциональные требования",
                kind="functional",
                summary="Push при достижении порога, один в месяц",
                quote="клиент получает push",
                line_start=39,
                line_end=39,
            ),
            dict(
                source_id="Ф-4",
                section="Функциональные требования",
                kind="functional",
                summary="Смена порога сохраняет сумму",
                quote="При изменении порога накопленная сумма сохраняется",
                line_start=40,
                line_end=40,
            ),
            dict(
                source_id="",
                section="Нефункциональные требования",
                kind="nonfunctional",
                summary="Push в течение минуты",
                quote="Сервис передаёт push на отправку в течение минуты",
                line_start=46,
                line_end=46,
            ),
            dict(
                source_id="Критерии приёмки, строка 2",
                section="Критерии приемки",
                kind="acceptance",
                summary="Покупка на стыке месяцев",
                quote="Февральский счётчик вырос на 4 000 ₸",
                line_start=64,
                line_end=64,
            ),
            dict(
                source_id="",
                section="Общие параметры",
                kind="api",
                summary="Метод GET сохранения порога",
                quote="ВЫДУМАННАЯ ЦИТАТА, которой нет",
                line_start=104,
                line_end=104,
            ),
        ],
    )


def _review(name):
    findings = []
    tests = []
    if name == "qa":
        findings.append(
            dict(
                requirement_ids=["R-02", "R-06"],
                criterion="consistent",
                severity="critical",
                title="Критерий приёмки противоречит Ф-2",
                evidence=[
                    dict(quote="Месяц определяется по дате совершения покупки в Алматы", line=38),
                    dict(quote="Февральский счётчик вырос на 4 000 ₸", line=64),
                ],
                problem="По Ф-2 покупка 31 января относится к январю, а критерий ожидает февраль.",
                recommendation="Согласовать правило отнесения к месяцу.",
                rewrite="",
            )
        )
        findings.append(
            dict(
                requirement_ids=["R-05"],
                criterion="verifiable",
                severity="critical",
                title="НФТ без точки измерения",
                evidence=[dict(quote="в течение минуты после получения события", line=46)],
                problem="Не указан перцентиль и нагрузка.",
                recommendation="Указать p95 и нагрузку.",
                rewrite="95% push передаются на отправку не позднее 60 с после получения события при нагрузке до N событий/с.",
            )
        )
        tests.append(
            dict(
                requirement_ids=["R-01"],
                title="Граница порога 999",
                preconditions="Карта KZT активна",
                action="Задать порог 999 ₸",
                expected="Ошибка «Проверьте сумму»",
                gap="Нет граничных значений",
            )
        )
    if name == "analyst":
        findings.append(
            dict(
                requirement_ids=["R-03", "R-04"],
                criterion="consistent",
                severity="critical",
                title="Ф-3 и Ф-4 противоречат по числу push",
                evidence=[
                    dict(quote="На карту приходится максимум один такой push за календарный месяц", line=39),
                    dict(quote="следующее достижение вызывает push независимо от истории предыдущей настройки", line=40),
                ],
                problem="После смены порога возможен второй push.",
                recommendation="Выбрать одно правило.",
                rewrite="",
            )
        )
        findings.append(
            dict(
                requirement_ids=["R-01"],
                criterion="unambiguous",
                severity="major",
                title="Галлюцинация",
                evidence=[dict(quote="Порог задаётся в долларах США", line=37)],
                problem="—",
                recommendation="—",
                rewrite="",
            )
        )
    checks = []
    if name == "qa":  # построчная сверка критериев приёмки: одна строка совпадает, одна противоречит Ф-2
        checks = [
            dict(
                acceptance_quote="Итог 50 000 ₸ и один push",
                acceptance_line=63,
                row_facts="Расходы 48 500, порог 50 000; покупка 1 500",
                requirement_ids=["R-03"],
                requirement_quote="клиент получает push",
                requirement_line=39,
                expected_by_requirements="Итог 50 000 ₸, один push",
                stated_expected="Итог 50 000 ₸ и один push",
                verdict="совпадает",
                explanation="Совпадает.",
            ),
            dict(
                acceptance_quote="Февральский счётчик вырос на 4 000 ₸",
                acceptance_line=64,
                row_facts="Покупка совершена 31.01 в 23:58; проведена 01.02 в 00:03",
                requirement_ids=["R-02"],
                requirement_quote="Месяц определяется по дате совершения покупки в Алматы",
                requirement_line=38,
                expected_by_requirements="Покупка 31 января — январский счётчик",
                stated_expected="Февральский счётчик вырос на 4 000 ₸",
                verdict="противоречит",
                explanation="По Ф-2 месяц — по дате покупки (январь), строка относит её к февралю.",
            ),
        ]
    return ReviewOut(findings=findings, suggested_tests=tests, acceptance_checks=checks)


class FakeLLM(LLMClient):
    def __init__(self):
        super().__init__(provider="fake", model="fake-model", cache_dir=None, rpm=0)
        self.prompts: dict[str, str] = {}

    def generate(self, agent, system, user, schema):
        self.prompts[agent] = system + "\n" + user
        self.calls.append(
            CallRecord(
                agent=agent,
                model=self.model,
                model_version="fake-model-001",
                cached=False,
                seconds=0.0,
                prompt_tokens=len(user) // 4,
                output_tokens=100,
            )
        )
        if agent == "segmenter":
            return _seg()
        if agent in ("analyst", "qa", "architect", "security"):
            return _review(agent)
        if agent == "consistency":
            return SetReviewOut(
                findings=[],
                trace=[
                    dict(requirement_id="R-01", verified_by=[], implemented_by=["R-07"], note="нет критерия приёмки на границы")
                ],
            )
        if agent == "judge":
            # объединяем всё, что пришло, по одному; первый кандидат отклоняем
            import re

            cids = re.findall(r"^(C-\d{3}) ", user, re.M)
            return JudgeOut(
                findings=[
                    dict(
                        candidate_ids=[c],
                        criterion="consistent" if i == 1 else "complete",
                        severity="critical" if i < 3 else "minor",
                        title=f"Итог {c}",
                        problem="p",
                        recommendation="r",
                        rewrite="",
                        priority=i if i < 4 else 0,
                    )
                    for i, c in enumerate(cids[1:], start=1)
                ],
                rejected=[dict(candidate_id=cids[0], reason="ложное срабатывание")],
                checklist=[
                    dict(item="Требование может быть протестировано", answer="Нет", comment="см. НФТ", candidate_ids=cids[1:2])
                ],
                verdict=dict(ready_for_development=False, ready_for_testing=False, summary="Есть противоречия."),
            )
        if agent == "clarifier":
            return ClarifyOut(
                questions=[
                    dict(
                        addressee="Аналитик",
                        question="Сколько push в месяц?",
                        why="спор",
                        options=["Один", "Один на значение порога"],
                        finding_ids=["D-01", "D-99"],
                        requirement_ids=["R-03"],
                        blocking=True,
                    )
                ]
            )
        if agent == "recheck":
            return RecheckOut(
                statuses=[dict(finding_id="D-01", status="resolved", comment="Внести: строго один push")], new_findings=[]
            )
        raise AssertionError(f"unexpected agent {agent}")
