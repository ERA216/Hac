"""Оркестратор: граф шагов, общее состояние прогона, устойчивость к сбоям агентов.

Граф:  нарезка → правила → 4 ревьюера → согласованность набора → проверка цитат
       → судья → уточняющие вопросы → (ответы → перепроверка) → итог.
Каждый шаг пишет запись в трассировку; сбой одного LLM-агента не обрывает прогон —
отчёт выпускается с пометкой, какой шаг не выполнен.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

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
from .llm import BudgetExceeded, CacheMiss, LLMClient, LLMError, QuotaExhausted
from .rules import run_rules
from .schemas import (
    ChecklistRow,
    Evidence,
    Finding,
    FindingOut,
    Question,
    Rejected,
    Requirement,
    TestCase,
    TraceLinkOut,
    VerdictOut,
)
from .verify import QuoteLocator, clean_line, dedupe_evidence

ALMATY = timezone(timedelta(hours=5), "Алматы")
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
        blocking = [f for f in self.open_findings if f.severity == "critical" and f.criterion in ("verifiable", "consistent", "validatable")]
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
    ):
        self.llm = llm
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
        except (QuotaExhausted, BudgetExceeded, CacheMiss) as e:
            note, ok = f"не выполнен: {e}", False
            if required:
                raise
        except LLMError as e:
            note, ok = f"не выполнен: {e}", False
            if required:
                raise
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

        # 1. Нарезка
        def segment():
            out = SegmenterAgent(self.llm).call(**base_ctx)
            unverified = 0
            for i, r in enumerate(out.requirements, start=1):
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

        # 3. Ревьюеры
        for lens in LENSES:
            def review(lens=lens):
                out = ReviewerAgent(self.llm, lens).call(
                    **base_ctx, requirements=reqs_text, rule_findings=rule_findings, max_findings=self.max_findings
                )
                for f in out.findings:
                    state.candidates.append(self._from_out(state, loc, f, lens.name))
                for t in out.suggested_tests:
                    state.tests.append(
                        TestCase(tid=f"T-{len(state.tests) + 1:02d}", **t.model_dump())
                    )
                return f"{len(out.findings)} находок" + (f", {len(out.suggested_tests)} тест-кейсов" if out.suggested_tests else "")

            self._step(state, lens.title, "LLM", review)

        # 4. Согласованность набора
        def consistency():
            out = ConsistencyAgent(self.llm).call(
                **base_ctx, requirements=reqs_text, known_findings=short_findings(state.candidates),
                max_findings=self.max_findings,
            )
            for f in out.findings:
                state.candidates.append(self._from_out(state, loc, f, "consistency"))
            known = {r.rid for r in state.requirements}
            state.trace = [t for t in out.trace if t.requirement_id in known]
            return f"{len(out.findings)} находок, {len(state.trace)} связей трассировки"

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
            answers = self.ask_answers(state.questions)
            for q in state.questions:
                q.answer = answers.get(q.qid, "").strip()
            answered = [q for q in state.questions if q.answer]
            if answered:
                state.steps.append(StepRecord("Ответы на уточняющие вопросы", "человек", 0.0, f"получено ответов: {len(answered)}"))
                self._step(state, "Агент перепроверки", "LLM", lambda: self._recheck(state, loc, base_ctx, reqs_text, answered))

        state.finished_at = datetime.now(ALMATY)
        return state

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
            c = by_id.get(r.candidate_id)
            if c and c.fid not in used:
                used.add(c.fid)
                state.rejected.append(Rejected(candidate_id=c.fid, title=c.title, source=", ".join(c.sources), reason=r.reason))
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
                item=row.item, answer=row.answer, comment=row.comment,
                finding_ids=sorted({cid_to_d[c] for c in row.candidate_ids if c in cid_to_d}),
            )
            for row in out.checklist
        ]
        state.verdict = out.verdict
        return (
            f"итог {len(state.findings)} дефектов из {len(state.candidates)} кандидатов; "
            f"отклонено {len(state.rejected)}" + (f"; добавлено пропущенных судьёй {len(orphans)}" if orphans else "")
        )

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
        return cid_to_d

    def _clarify(self, state: RunState, base_ctx: dict, reqs_text: str) -> str:
        if not state.findings:
            return "дефектов нет — вопросов нет"
        out = ClarifierAgent(self.llm).call(
            **base_ctx, requirements=reqs_text, findings=full_findings(state.open_findings),
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
        out = RecheckAgent(self.llm).call(
            **base_ctx, requirements=reqs_text, qa=questions_answers(answered), findings=full_findings(affected)
        )
        by_id = {f.fid: f for f in state.findings}
        changed = 0
        for s in out.statuses:
            f = by_id.get(s.finding_id)
            if f and s.finding_id in affected_ids:
                f.status = s.status
                f.status_comment = s.comment.strip()
                changed += s.status != "open"
        added = 0
        for nf in out.new_findings:
            f = self._from_out(state, loc, nf, "recheck")
            f.evidence = [e for e in f.evidence if e.verified]
            if f.evidence:
                f.fid = f"D-{len(state.findings) + 1:02d}"
                state.findings.append(f)
                added += 1
        return f"закрыто/изменено {changed} из {len(affected)} затронутых дефектов, новых: {added}"
