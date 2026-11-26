"""Оркестратор: граф шагов, общее состояние прогона, устойчивость к сбоям агентов.

Граф:  нарезка → правила → 4 ревьюера → согласованность набора → проверка цитат
       → судья → уточняющие вопросы → (ответы → перепроверка) → итог.
Каждый шаг пишет запись в трассировку; сбой одного LLM-агента не обрывает прогон —
отчёт выпускается с пометкой, какой шаг не выполнен.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .agents import (
    LENSES,
    ClarifierAgent,
    ConsistencyAgent,
    JudgeAgent,
    RecheckAgent,
    ReviewerAgent,
    SegmenterAgent,
)
from .agents.context import full_findings, questions_answers, requirements_table, short_findings
from .criteria import CRITERIA
from .document import Document
from .llm import BudgetExceeded, CacheMiss, FatalLLMError, LLMClient, LLMError, QuotaExhausted
from .rules import run_rules
from .schemas import (
    AcceptanceCheck,
    AcceptanceCheckOut,
    ChecklistRow,
    Evidence,
    Finding,
    FindingOut,
    PairCheck,
    PairCheckOut,
    Question,
    Rejected,
    Requirement,
    TestCase,
    TraceLinkOut,
    VerdictOut,
)
from .segmentation import chunk_ranges, numbered_range, rule_segments
from .verify import QuoteLocator, clean_line, dedupe_evidence

ALMATY = timezone(timedelta(hours=5), "Алматы")
# обоснование, противоречащее статусу «закрыт уточнением»
_NOT_RESOLVED = re.compile(
    r"не\s+(?:уточня|снима|закрыва|устраня|отвеча\w*\s+на|определя|решает|покрыва)|"
    r"оста[её]тся\s+(?:открыт|неопредел|нерешён|нерешен)|по-прежнему\s+не|частично",
    re.I,
)
SEV_ORDER = {"critical": 0, "major": 1, "minor": 2}


class Progress(Protocol):
    def start(self, title: str) -> None: ...
    def done(self, title: str, note: str) -> None: ...
    def warn(self, message: str) -> None: ...


class _Silent:
    def start(self, title):
        pass

    def done(self, title, note):
        pass

    def warn(self, message):
        pass


@dataclass
class StepRecord:
    title: str
    kind: str  # "LLM" | "код" | "человек"
    seconds: float
    note: str
    ok: bool = True


@dataclass
class RunState:
    doc: Document
    started_at: datetime = field(default_factory=lambda: datetime.now(ALMATY))
    finished_at: datetime | None = None
    requirements: list[Requirement] = field(default_factory=list)
    candidates: list[Finding] = field(default_factory=list)
    unverified: list[Finding] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    tests: list[TestCase] = field(default_factory=list)
    acceptance: list[AcceptanceCheck] = field(default_factory=list)  # построчная сверка критериев приёмки
    pairs: list[PairCheck] = field(default_factory=list)  # попарная сверка требований об одном действии
    trace: list[TraceLinkOut] = field(default_factory=list)
    checklist: list[ChecklistRow] = field(default_factory=list)
    verdict: VerdictOut | None = None
    steps: list[StepRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    quotes_checked: int = 0
    quotes_verified: int = 0

    @property
    def open_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.is_open]

    def count(self, severity: str) -> int:
        return sum(1 for f in self.open_findings if f.severity == severity)

    @property
    def ready_for_development(self) -> bool:
        llm_ok = self.verdict.ready_for_development if self.verdict else True
        return llm_ok and self.count("critical") == 0

    @property
    def ready_for_testing(self) -> bool:
        blocking = [
            f
            for f in self.open_findings
            if f.severity == "critical" and f.criterion in ("verifiable", "consistent", "validatable")
        ]
        llm_ok = self.verdict.ready_for_testing if self.verdict else True
        return llm_ok and not blocking


AskAnswers = Callable[[list[Question]], dict[str, str]]


class Pipeline:
    def __init__(
        self,
        llm: LLMClient,
        progress: Progress | None = None,
        ask_answers: AskAnswers | None = None,
        max_findings_per_agent: int = 15,
        max_questions: int = 10,
        parallel: int = 4,
        thread_initializer: Callable[[], object] | None = None,
        segment_chunk_chars: int = 24000,
    ):
        self.llm = llm
        self.segment_chunk_chars = segment_chunk_chars
        self.parallel = max(1, parallel)
        self.thread_initializer = thread_initializer
        self.progress = progress or _Silent()
        self.ask_answers = ask_answers
        self.max_findings = max_findings_per_agent
        self.max_questions = max_questions
        self._cid = 0

    # ------------------------------------------------------------ helpers
    def _step(self, state: RunState, title: str, kind: str, fn: Callable[[], str], required: bool = False) -> bool:
        self.progress.start(title)
        t0 = time.monotonic()
        try:
            note = fn() or ""
            ok = True
        except FatalLLMError:
            self.progress.warn(f"{title}: прогон остановлен")
            raise  # неверный ключ, нет доступа, сервер модели не запущен — продолжать бессмысленно
        except (QuotaExhausted, BudgetExceeded, CacheMiss) as e:
            note, ok = f"не выполнен: {e}", False
            if required:
                raise
        except LLMError as e:
            note, ok = f"не выполнен: {e}", False
            if required:
                raise
        except Exception as e:  # noqa: BLE001 — неожиданный сбой одного шага не должен терять весь отчёт
            if required:
                raise
            note, ok = f"не выполнен: внутренняя ошибка {e.__class__.__name__}: {str(e)[:200]}", False
        dt = round(time.monotonic() - t0, 1)
        state.steps.append(StepRecord(title=title, kind=kind, seconds=dt, note=note, ok=ok))
        if ok:
            self.progress.done(title, note)
        else:
            state.errors.append(f"{title}: {note}")
            self.progress.warn(f"{title}: {note}")
        return ok

    def _next_cid(self) -> str:
        self._cid += 1
        return f"C-{self._cid:03d}"

    def _req_ids_for_lines(self, state: RunState, evidence: list[Evidence]) -> list[str]:
        ids = []
        for e in evidence:
            if not e.line_start:
                continue
            for r in state.requirements:
                rs, re_ = r.evidence.line_start or 0, r.evidence.line_end or r.evidence.line_start or 0
                if rs <= e.line_start <= max(rs, re_) and r.rid not in ids:
                    ids.append(r.rid)
        return ids

    def _from_out(self, state: RunState, loc: QuoteLocator, out: FindingOut, source: str) -> Finding:
        evidence = []
        for ev in out.evidence:
            state.quotes_checked += 1
            located = loc.locate(ev.quote, ev.line)
            if located.verified:
                state.quotes_verified += 1
            evidence.append(located)
        known = {r.rid for r in state.requirements}
        req_ids = [r for r in out.requirement_ids if r in known]
        if not req_ids:
            req_ids = self._req_ids_for_lines(state, [e for e in evidence if e.verified])
        criterion = out.criterion if out.criterion in CRITERIA else "complete"
        return Finding(
            fid=self._next_cid(),
            requirement_ids=req_ids,
            criterion=criterion,
            severity=out.severity,
            title=out.title.strip(),
            problem=out.problem.strip(),
            recommendation=out.recommendation.strip(),
            rewrite=out.rewrite.strip(),
            evidence=dedupe_evidence(evidence),
            sources=[source],
        )

    # ------------------------------------------------------------ steps
    def run(self, doc: Document) -> RunState:
        state = RunState(doc=doc)
        loc = QuoteLocator(doc)
        numbered = doc.numbered()
        base_ctx = {"doc": doc, "numbered": numbered}

        # 0. Предполётная проверка: начинаем с модели цепочки, которая отвечает прямо сейчас
        if self.llm.provider == "gemini" and not self.llm.replay:

            def preflight():
                model = self.llm.preflight()
                return f"работаем на {model}" if model else "ни одна модель не ответила быстро — начинаем с основной"

            self._step(state, "Проверка доступности модели", "LLM", preflight)

        # 1. Нарезка
        def segment():
            # длинный документ режем на части по заголовкам; если LLM недоступна — нарезка по структуре
            chunks = chunk_ranges(doc, self.segment_chunk_chars)
            # бюджет вызовов — страховка от зацикливания: длинному документу нужно больше вызовов нарезчика
            needed = len(chunks) + 8  # части + 4 ревьюера + согласованность + судья + уточнения + перепроверка
            if self.llm.max_calls < needed:
                self.llm.max_calls = needed
            items, by_rules = [], []
            for n, (a, b) in enumerate(chunks, start=1):
                ctx = dict(base_ctx)
                if len(chunks) > 1:
                    ctx["numbered"] = numbered_range(doc, a, b)
                    ctx["part"] = f"часть {n} из {len(chunks)} (строки {a}–{b}); выдели требования только из этой части"
                try:
                    part = SegmenterAgent(self.llm).call(**ctx).requirements
                except FatalLLMError:
                    raise
                except LLMError as e:
                    by_rules.append(f"{a}–{b}")
                    items.extend(rule_segments(doc, (a, b)))
                    state.errors.append(
                        f"Нарезка требований (строки {a}–{b}): LLM недоступна ({e}); использована нарезка по структуре документа"
                    )
                    continue
                by_structure = rule_segments(doc, (a, b))
                if not part and by_structure:  # модель ничего не выделила там, где структура видит требования
                    by_rules.append(f"{a}–{b}")
                    part = by_structure
                    state.errors.append(
                        f"Нарезка требований (строки {a}–{b}): модель не выделила ни одного требования; "
                        "использована нарезка по структуре документа"
                    )
                items.extend(part)
            unverified = 0
            for i, r in enumerate(items, start=1):
                ev = loc.locate(r.quote, r.line_start)
                if not ev.verified:
                    unverified += 1
                    start = r.line_start if doc.line(r.line_start) else None
                    if start:
                        ev.line_start, ev.line_end = start, r.line_end if doc.line(r.line_end) else start
                        ev.location = doc.location(ev.line_start, ev.line_end)
                        ev.quote = clean_line(doc.line(start).text)
                state.requirements.append(
                    Requirement(
                        rid=f"R-{i:02d}",
                        source_id=r.source_id.strip(),
                        section=r.section.strip() or (doc.line(ev.line_start).section if ev.line_start else ""),
                        kind=r.kind,
                        summary=r.summary.strip(),
                        evidence=ev,
                    )
                )
            note = f"{len(state.requirements)} требований"
            if len(chunks) > 1:
                note += f", документ обработан частями: {len(chunks)}"
            if by_rules:
                note += f"; по структуре (без LLM) нарезаны строки {', '.join(by_rules)}"
            return note + (f", {unverified} цитат восстановлено по номерам строк" if unverified else "")

        self._step(state, "Нарезка требований", "LLM", segment, required=True)

        # 2. Детерминированные правила
        def rules():
            hits = run_rules(doc, loc)
            for h in hits:
                state.candidates.append(
                    Finding(
                        fid=self._next_cid(),
                        requirement_ids=self._req_ids_for_lines(state, h.evidence),
                        criterion=h.criterion,
                        severity=h.severity,
                        title=h.title,
                        problem=h.problem,
                        recommendation=h.recommendation,
                        evidence=h.evidence,
                        sources=[f"правила:{h.rule}"],
                    )
                )
            return f"{len(hits)} срабатываний"

        self._step(state, "Детерминированные правила", "код", rules)
        rule_findings = short_findings(state.candidates)
        reqs_text = requirements_table(state.requirements)

        # 3. Ревьюеры: вызовы параллельно, обработка результатов — в фиксированном порядке,
        #    чтобы идентификаторы кандидатов и промпты следующих агентов были детерминированы (кэш, --replay)
        def call_lens(lens):
            t0 = time.monotonic()
            try:
                out = ReviewerAgent(self.llm, lens).call(
                    **base_ctx, requirements=reqs_text, rule_findings=rule_findings, max_findings=self.max_findings
                )
                return out, None, round(time.monotonic() - t0, 1)
            except FatalLLMError:
                raise
            except LLMError as e:
                return None, e, round(time.monotonic() - t0, 1)

        for lens in LENSES:
            self.progress.start(lens.title)
        results: dict[str, tuple] = {}
        if self.parallel > 1:
            with ThreadPoolExecutor(max_workers=min(self.parallel, len(LENSES)), initializer=self.thread_initializer) as ex:
                futures = {ex.submit(call_lens, lens): lens for lens in LENSES}
                for fut in as_completed(futures):
                    lens = futures[fut]
                    results[lens.name] = fut.result()
                    out, err, _ = results[lens.name]
                    if err is None:
                        self.progress.done(lens.title, f"{len(out.findings)} находок")
                    else:
                        self.progress.warn(f"{lens.title}: не выполнен: {err}")
        else:
            for lens in LENSES:
                results[lens.name] = call_lens(lens)
                out, err, _ = results[lens.name]
                if err is None:
                    self.progress.done(lens.title, f"{len(out.findings)} находок")
                else:
                    self.progress.warn(f"{lens.title}: не выполнен: {err}")
        for lens in LENSES:
            out, err, dt = results[lens.name]
            if err is not None:
                note = f"не выполнен: {err}"
                state.errors.append(f"{lens.title}: {note}")
                state.steps.append(StepRecord(title=lens.title, kind="LLM", seconds=dt, note=note, ok=False))
                continue
            for f in out.findings:
                state.candidates.append(self._from_out(state, loc, f, lens.name))
            for tc in out.suggested_tests:
                state.tests.append(TestCase(tid=f"T-{len(state.tests) + 1:02d}", **tc.model_dump()))
            for ac in out.acceptance_checks:
                self._add_acceptance_check(state, loc, ac, lens.name)
            note = f"{len(out.findings)} находок" + (f", {len(out.suggested_tests)} тест-кейсов" if out.suggested_tests else "")
            state.steps.append(StepRecord(title=lens.title, kind="LLM", seconds=dt, note=note))

        # 4. Согласованность набора
        def consistency():
            out = ConsistencyAgent(self.llm).call(
                **base_ctx,
                requirements=reqs_text,
                known_findings=short_findings(state.candidates),
                max_findings=self.max_findings,
            )
            for pc in out.pair_checks:
                self._add_pair_check(state, loc, pc)
            for f in out.findings:
                state.candidates.append(self._from_out(state, loc, f, "consistency"))
            known = {r.rid for r in state.requirements}
            state.trace = [t for t in out.trace if t.requirement_id in known]
            conflicts = sum(p.verdict == "противоречат" for p in state.pairs)
            return (
                f"{len(out.findings)} находок, пар сверено {len(state.pairs)} (противоречий {conflicts}), "
                f"{len(state.trace)} связей трассировки"
            )

        self._step(state, "Агент согласованности набора", "LLM", consistency)

        # 5. Проверка цитат
        def verify():
            kept = []
            for c in state.candidates:
                good = [e for e in c.evidence if e.verified]
                if good:
                    c.evidence = good
                    kept.append(c)
                else:
                    state.unverified.append(c)
            state.candidates = kept
            return (
                f"цитат проверено {state.quotes_checked}, подтверждено {state.quotes_verified}; "
                f"отброшено находок без подтверждения: {len(state.unverified)}"
            )

        self._step(state, "Проверка цитат по документу", "код", verify)

        # 6. Судья
        judged = self._step(state, "Судья-верификатор", "LLM", lambda: self._judge(state, base_ctx, reqs_text))
        if not judged:  # запасной путь: кандидаты без слияния
            state.findings = [c.model_copy(update={"candidate_ids": [c.fid]}) for c in state.candidates]
            self._number_findings(state, {})

        # 7. Уточняющие вопросы
        self._step(state, "Агент уточнений", "LLM", lambda: self._clarify(state, base_ctx, reqs_text))

        # 8. Ответы человека и перепроверка
        if self.ask_answers and state.questions:
            self.apply_answers(state, self.ask_answers(state.questions))

        state.finished_at = datetime.now(ALMATY)
        return state

    def apply_answers(self, state: RunState, answers: dict[str, str]) -> str:
        """Учесть ответы на уточняющие вопросы: агент перепроверки закрывает или меняет дефекты."""
        for q in state.questions:
            a = (answers.get(q.qid) or "").strip()
            if a.isdigit() and 1 <= int(a) <= len(q.options):
                a = q.options[int(a) - 1]
            if a:
                q.answer = a
        answered = [q for q in state.questions if q.answer]
        if not answered:
            return "ответов нет"
        doc = state.doc
        loc = QuoteLocator(doc)
        base_ctx = {"doc": doc, "numbered": doc.numbered()}
        state.steps.append(StepRecord("Ответы на уточняющие вопросы", "человек", 0.0, f"получено ответов: {len(answered)}"))
        self._step(
            state,
            "Агент перепроверки",
            "LLM",
            lambda: self._recheck(state, loc, base_ctx, requirements_table(state.requirements), answered),
        )
        state.finished_at = datetime.now(ALMATY)
        return state.steps[-1].note

    @staticmethod
    def _fix_ids(text: str, cid_to_d: dict[str, str]) -> str:
        """Заменить внутренние ID кандидатов (C-xxx) в текстах судьи на номера итоговых дефектов."""
        t = re.sub(r"\bC-\d{3}\b", lambda m: cid_to_d.get(m.group(0), ""), text or "")
        t = re.sub(r"\b(D-\d{2})(?:,\s*\1\b)+", r"\1", t)  # «D-01, D-01» → «D-01»
        t = re.sub(r"\(\s*[,;]?\s*\)", "", t)  # пустые скобки от удалённых ID
        t = re.sub(r"(?:,\s*){2,}", ", ", t)
        t = re.sub(r"\(\s*,\s*", "(", t)
        t = re.sub(r"\s*,\s*\)", ")", t)
        t = re.sub(r"\s+([,.;:])", r"\1", t)
        return re.sub(r"\s{2,}", " ", t).strip()

    def _judge(self, state: RunState, base_ctx: dict, reqs_text: str) -> str:
        out = JudgeAgent(self.llm).call(**base_ctx, requirements=reqs_text, candidates=full_findings(state.candidates))
        by_id = {c.fid: c for c in state.candidates}
        used: set[str] = set()
        findings: list[Finding] = []
        for m in out.findings:
            cands = [by_id[c] for c in m.candidate_ids if c in by_id and c not in used]
            if not cands:
                continue
            used.update(c.fid for c in cands)
            evidence, seen = [], set()
            for c in cands:
                for e in c.evidence:
                    key = (e.line_start, e.quote[:60])
                    if key not in seen:
                        seen.add(key)
                        evidence.append(e)
            req_ids = []
            for c in cands:
                req_ids += [r for r in c.requirement_ids if r not in req_ids]
            sources = []
            for c in cands:
                sources += [s for s in c.sources if s not in sources]
            findings.append(
                Finding(
                    fid="",
                    requirement_ids=req_ids,
                    criterion=m.criterion if m.criterion in CRITERIA else cands[0].criterion,
                    severity=m.severity,
                    title=m.title.strip() or cands[0].title,
                    problem=m.problem.strip() or cands[0].problem,
                    recommendation=m.recommendation.strip() or cands[0].recommendation,
                    rewrite=m.rewrite.strip() or next((c.rewrite for c in cands if c.rewrite), ""),
                    evidence=dedupe_evidence(evidence)[:4],
                    sources=sources,
                    candidate_ids=[c.fid for c in cands],
                    priority=max(0, m.priority),
                )
            )
        for r in out.rejected:
            rej = by_id.get(r.candidate_id)
            if rej and rej.fid not in used:
                used.add(rej.fid)
                state.rejected.append(
                    Rejected(candidate_id=rej.fid, title=rej.title, source=", ".join(rej.sources), reason=r.reason)
                )
        # кандидаты, о которых судья забыл, не теряем
        orphans = [c for c in state.candidates if c.fid not in used]
        for c in orphans:
            f = c.model_copy()
            f.candidate_ids = [c.fid]
            findings.append(f)
        state.findings = findings
        cid_to_d = self._number_findings(state, {})
        state.checklist = [
            ChecklistRow(
                item=row.item,
                answer=row.answer,
                comment=row.comment,
                finding_ids=sorted({cid_to_d[c] for c in row.candidate_ids if c in cid_to_d}),
            )
            for row in out.checklist
        ]
        state.verdict = out.verdict
        fix = lambda s: self._fix_ids(s, cid_to_d)  # noqa: E731
        for f in state.findings:
            f.title, f.problem = fix(f.title), fix(f.problem)
            f.recommendation, f.rewrite = fix(f.recommendation), fix(f.rewrite)
        for row in state.checklist:
            row.comment = fix(row.comment)
        state.verdict.summary = fix(state.verdict.summary)
        return f"итог {len(state.findings)} дефектов из {len(state.candidates)} кандидатов; отклонено {len(state.rejected)}" + (
            f"; добавлено пропущенных судьёй {len(orphans)}" if orphans else ""
        )

    def _add_acceptance_check(self, state: RunState, loc: QuoteLocator, ac: AcceptanceCheckOut, source: str) -> None:
        """Строка критерия приёмки, сверенная с требованием. «Противоречит» — сразу кандидат в критичный дефект."""
        state.quotes_checked += 2
        acc = loc.locate(ac.acceptance_quote, ac.acceptance_line)
        req = loc.locate(ac.requirement_quote, ac.requirement_line)
        state.quotes_verified += acc.verified + req.verified
        known = {r.rid for r in state.requirements}
        req_ids = [r for r in ac.requirement_ids if r in known] or self._req_ids_for_lines(state, [acc, req])
        check = AcceptanceCheck(
            acceptance=acc,
            requirement=req,
            requirement_ids=req_ids,
            expected_by_requirements=ac.expected_by_requirements.strip(),
            stated_expected=ac.stated_expected.strip(),
            verdict=ac.verdict,
            explanation=ac.explanation.strip(),
        )
        state.acceptance.append(check)
        if ac.verdict != "противоречит":
            return
        req_ref = next((r.source_id for r in state.requirements if r.rid in req_ids and r.source_id), "требованию")
        cand = Finding(
            fid=self._next_cid(),
            requirement_ids=req_ids,
            criterion="consistent",
            severity="critical",
            title=f"Критерий приёмки противоречит {req_ref}",
            problem=(
                f"По требованию ожидается: {check.expected_by_requirements}. В критерии приёмки указано: "
                f"{check.stated_expected}. {check.explanation}"
            ),
            recommendation="Согласовать требование и критерий приёмки: исправить одно из них и зафиксировать правило явно.",
            evidence=dedupe_evidence([acc, req]),
            sources=[source],
        )
        check.finding_id = cand.fid  # после судьи заменится на номер итогового дефекта
        state.candidates.append(cand)

    def _add_pair_check(self, state: RunState, loc: QuoteLocator, pc: PairCheckOut) -> None:
        """Пара требований об одном действии. «Противоречат» — сразу кандидат в критичный дефект."""
        state.quotes_checked += 2
        first = loc.locate(pc.first_quote, pc.first_line)
        second = loc.locate(pc.second_quote, pc.second_line)
        state.quotes_verified += first.verified + second.verified
        known = {r.rid for r in state.requirements}
        req_ids = [r for r in pc.requirement_ids if r in known] or self._req_ids_for_lines(state, [first, second])
        check = PairCheck(
            subject=pc.subject.strip(),
            first=first,
            second=second,
            requirement_ids=req_ids,
            scenario=pc.scenario.strip(),
            first_result=pc.first_result.strip(),
            second_result=pc.second_result.strip(),
            verdict=pc.verdict,
            explanation=pc.explanation.strip(),
        )
        state.pairs.append(check)
        if pc.verdict != "противоречат":
            return
        refs = [r.source_id for r in state.requirements if r.rid in req_ids and r.source_id]
        names = " и ".join(dict.fromkeys(refs)) if len(set(refs)) > 1 else "Требования"
        cand = Finding(
            fid=self._next_cid(),
            requirement_ids=req_ids,
            criterion="consistent",
            severity="critical",
            title=f"{names} противоречат друг другу ({check.subject})",
            problem=(
                f"Сценарий: {check.scenario}. По первому требованию: {check.first_result}. "
                f"По второму: {check.second_result}. {check.explanation}"
            ),
            recommendation="Согласовать требования: выбрать одно правило для этого сценария и записать его явно в обоих местах.",
            evidence=dedupe_evidence([first, second]),
            sources=["consistency"],
        )
        check.finding_id = cand.fid  # после судьи заменится на номер итогового дефекта
        state.candidates.append(cand)

    def _number_findings(self, state: RunState, _) -> dict[str, str]:
        req_order = {r.rid: i for i, r in enumerate(state.requirements)}

        def key(f: Finding):
            first_req = min((req_order.get(r, 999) for r in f.requirement_ids), default=999)
            first_line = min((e.line_start or 10**6 for e in f.evidence), default=10**6)
            return (SEV_ORDER.get(f.severity, 3), first_req, first_line)

        state.findings.sort(key=key)
        cid_to_d = {}
        for i, f in enumerate(state.findings, start=1):
            f.fid = f"D-{i:02d}"
            for c in f.candidate_ids or []:
                cid_to_d[c] = f.fid
        checks: list[AcceptanceCheck | PairCheck] = [*state.acceptance, *state.pairs]
        for check in checks:
            if check.finding_id.startswith("C-"):
                check.finding_id = cid_to_d.get(check.finding_id, "")
        return cid_to_d

    def _clarify(self, state: RunState, base_ctx: dict, reqs_text: str) -> str:
        if not state.findings:
            return "дефектов нет — вопросов нет"
        out = ClarifierAgent(self.llm).call(
            **base_ctx,
            requirements=reqs_text,
            findings=full_findings(state.open_findings),
            max_questions=self.max_questions,
        )
        known_f = {f.fid for f in state.findings}
        known_r = {r.rid for r in state.requirements}
        qs = sorted(out.questions, key=lambda q: not q.blocking)[: self.max_questions]
        state.questions = [
            Question(
                qid=f"Q-{i:02d}",
                addressee=q.addressee,
                question=q.question.strip(),
                why=q.why.strip(),
                options=[o.strip() for o in q.options if o.strip()],
                finding_ids=[f for f in q.finding_ids if f in known_f],
                requirement_ids=[r for r in q.requirement_ids if r in known_r],
                blocking=q.blocking,
            )
            for i, q in enumerate(qs, start=1)
        ]
        blocking = sum(q.blocking for q in state.questions)
        return f"{len(state.questions)} вопросов, из них блокирующих {blocking}"

    def _recheck(self, state: RunState, loc: QuoteLocator, base_ctx: dict, reqs_text: str, answered: list[Question]) -> str:
        affected_ids = {f for q in answered for f in q.finding_ids}
        affected = [f for f in state.findings if f.fid in affected_ids]
        if not affected:  # ответы не привязаны ни к одному дефекту — перепроверять нечего, запрос модели не нужен
            return "ответы не затрагивают дефекты — перепроверка не нужна"
        out = RecheckAgent(self.llm).call(
            **base_ctx, requirements=reqs_text, qa=questions_answers(answered), findings=full_findings(affected)
        )
        by_id = {f.fid: f for f in state.findings}
        changed = downgraded = 0
        for s in out.statuses:
            f = by_id.get(s.finding_id)
            if f and s.finding_id in affected_ids:
                status, comment = s.status, s.comment.strip()
                # статус должен совпадать с обоснованием: «закрыт», но «ответ не уточняет» — значит открыт
                if status == "resolved" and _NOT_RESOLVED.search(comment):
                    status, downgraded = "open", downgraded + 1
                    comment = "Ответ не снимает дефект полностью. " + comment
                f.status, f.status_comment = status, comment
                changed += status != "open"
        added = 0
        for nf in out.new_findings:
            new = self._from_out(state, loc, nf, "recheck")
            new.evidence = [e for e in new.evidence if e.verified]
            if new.evidence:
                new.fid = f"D-{len(state.findings) + 1:02d}"
                state.findings.append(new)
                added += 1
        note = f"закрыто/изменено {changed} из {len(affected)} затронутых дефектов, новых: {added}"
        return note + (f"; возвращено в открытые (обоснование противоречило статусу): {downgraded}" if downgraded else "")
