"""Детерминированные проверки без LLM.

Правила общие для любых ТЗ (не подогнаны под конкретный документ):
язык требований (29148 п. 5.2.7), незаполненные шаблоны и пустые разделы,
смешение денежных единиц, семантика HTTP-методов и кодов ответа, расхождение
типов параметров API, идентификаторы в строке запроса.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .document import Document
from .schemas import Evidence
from .verify import QuoteLocator, clean_line


@dataclass
class RuleHit:
    rule: str
    criterion: str
    severity: str
    title: str
    problem: str
    recommendation: str
    evidence: list[Evidence] = field(default_factory=list)


def _w(pattern: str) -> re.Pattern:
    return re.compile(rf"(?<![\w-])(?:{pattern})(?![\w-])", re.I)


# --- 29148 п. 5.2.7: субъективные, размытые и незамкнутые формулировки
VAGUE = {
    "неизмеримое время": _w(
        r"быстр\w*|оперативн\w*|своевременн\w*|мгновенн\w*|немедленн\w*|сразу|в момент|в реальном времени|как можно скорее"
    ),
    "субъективная оценка": _w(r"удобн\w*|интуитивн\w*|дружелюбн\w*|простой|просто|легко|понятн\w*|красив\w*|современн\w*"),
    "неизмеримое качество": _w(
        r"достаточн\w*|приемлем\w*|разумн\w*|оптимальн\w*|эффективн\w*|надёжн\w*|надежн\w*|гибк\w*|"
        r"высок\w* (?:производительн\w*|доступн\w*|качеств\w*)"
    ),
    "лазейка": _w(r"при необходимости|по возможности|если возможно|как правило|обычно|в основном|по усмотрению|желательно"),
    "незамкнутое перечисление": _w(
        r"и т\.\s?д\.?|и т\.\s?п\.?|и др\.|и прочее|и тому подобное|и так далее|и далее|в том числе|включая,? но не ограничиваясь"
    ),
    "неопределённое количество": _w(r"несколько|некотор\w*|многие|много|мало|около|примерно|порядка|большое количество"),
    "неоднозначная связка": _w(r"и/или"),
    "неопределённая ссылка": _w(r"соответствующ\w*|вышеуказанн\w*|вышеперечисленн\w*|данный|этот|оный"),
}
# слова, которые часто встречаются в нейтральном смысле — считаем только в требованиях
_WEAK = {"неопределённая ссылка"}

PLACEHOLDER = [
    (re.compile(r"\b(TBD|TBA|TODO|FIXME|XXX)\b"), "маркер «не определено»"),
    (
        re.compile(
            r"заполня\w+ (позже|тестировщик\w*|аналитик\w*|разработчик\w*)|"
            r"будет (заполнен|определ\w+|уточн\w+)|уточняется|не заполнен\w*",
            re.I,
        ),
        "раздел не заполнен",
    ),
    (re.compile(r"Да\s*/\s*Нет"), "неотмеченный вариант шаблона «Да / Нет»"),
]
_EMPTY_ROW = re.compile(r"^\|(\s*\|)+\s*$")
_DASH_CELL = re.compile(r"^\|\s*\**[^|]+?\**\s*\|\s*[—–-]\s*\|\s*$")
_PROD_DASH = re.compile(r"\b(Prod|Production|Прод)\w*\s*:\s*[—–-]", re.I)

_MINOR_UNIT = _w(r"тиын\w*|тийын\w*|копе\w*|цент\w*|minor units?")
_MAJOR_UNIT = re.compile(r"(?<![\w-])(тенге|рубл\w*|доллар\w*)(?![\w-])|₸|\bKZT\b|\bRUB\b|\bUSD\b", re.I)

_HTTP_METHOD = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b")
_MUTATING_URL = re.compile(r"/[\w/{}-]*(save|update|create|delete|remove|set|add|edit|change)\w*", re.I)
_MUTATING_WORD = re.compile(r"сохранен\w*|сохранить|изменен\w*|изменить|удален\w*|удалить|создан\w*|создать|заменить", re.I)
_SENSITIVE_PARAM = re.compile(
    r"card|\bpan\b|account|iin|иин|phone|телефон|passport|паспорт|token|password|cvv|iban|e-?mail|fio|фио|birth|snils|inn\b",
    re.I,
)
_NAME_COLS = {"parameter", "параметр", "поле", "field", "name", "имя", "атрибут", "attribute", "наименование", "название"}
_TYPE_COLS = {"format", "тип", "формат", "type", "тип данных", "data type", "datatype"}
_LOCATION_COLS = {"in", "где", "расположение", "location", "место", "передача", "источник"}
_HTTP_CODE = re.compile(r"(?<![\d\s])(?<!\d )\b([1-5]\d\d)\b(?![\s ]?\d{3})")


def _is_table_sep(t: str) -> bool:
    return bool(re.match(r"^\|?\s*:?-{2,}", t.strip()))


def _cells(t: str) -> list[str]:
    return [re.sub(r"[*`]", "", c).strip() for c in t.strip().strip("|").split("|")]


def check_language(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    by_kind: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ln in doc.lines:
        t = ln.text
        if not t.strip() or ln.heading or t.lstrip().startswith("#") or _is_table_sep(t) or t.strip().startswith("```"):
            continue
        for kind, rx in VAGUE.items():
            if kind in _WEAK:
                continue
            for m in rx.finditer(t):
                by_kind[kind].append((ln.no, m.group(0)))
    hits = []
    for kind, items in by_kind.items():
        for no, word in items:
            hits.append(
                RuleHit(
                    rule="R-LANG",
                    criterion="language",
                    severity="major" if kind in ("неизмеримое время", "незамкнутое перечисление", "лазейка") else "minor",
                    title=f"Размытая формулировка: «{word}» ({kind})",
                    problem=(
                        f"Слово «{word}» относится к категории «{kind}» (ISO/IEC/IEEE 29148, п. 5.2.7): "
                        "оно не задаёт измеримого и однозначного условия, поэтому разные участники "
                        "поймут его по-разному, а проверить выполнение объективно нельзя."
                    ),
                    recommendation="Заменить на измеримое условие: точное значение, границу, перечень или момент времени.",
                    evidence=[loc.line_evidence(no)],
                )
            )
    return hits


def check_placeholders(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ln in doc.lines:
        t = ln.text
        if _is_table_sep(t):
            continue
        for rx, kind in PLACEHOLDER:
            if rx.search(t):
                grouped[(ln.section, kind)].append(ln.no)
                break
        else:
            if _EMPTY_ROW.match(t.strip()):
                grouped[(ln.section, "пустая строка таблицы (не заполнено)")].append(ln.no)
            elif _DASH_CELL.match(t.strip()) or _PROD_DASH.search(t):
                grouped[(ln.section, "значение поля не указано («—»)")].append(ln.no)
    hits = []
    for (section, kind), lines in grouped.items():
        evid = [loc.line_evidence(n) for n in lines[:3]]
        for e in evid:
            if not e.quote:
                e.quote = "(пустая строка таблицы)"
        hits.append(
            RuleHit(
                rule="R-PLACEHOLDER",
                criterion="set_complete" if kind != "значение поля не указано («—»)" else "conforming",
                severity="minor" if "Да / Нет" in kind else "major",
                title=f"Незаполненный шаблон: {kind}" + (f" — {len(lines)} мест" if len(lines) > 1 else ""),
                problem=(
                    f"В разделе «{section or 'без раздела'}» обнаружено: {kind} "
                    f"({len(lines)} шт.). Документ в таком виде нельзя считать завершённым: "
                    "информация, которую должен дать шаблон, отсутствует."
                ),
                recommendation="Заполнить раздел или явно указать «не применимо» с обоснованием.",
                evidence=evid,
            )
        )
    return hits


def check_empty_sections(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    hits = []
    heads = []
    for ln in doc.lines:
        m = re.match(r"^(#{1,6})\s", ln.text)
        if m:
            heads.append((ln.no, len(m.group(1))))
        elif ln.heading:  # распознанный заголовок без разметки: уровень по глубине нумерации
            num = re.match(r"^\s*(\d+(?:\.\d+)*)", ln.text)
            heads.append((ln.no, (num.group(1).count(".") + 2) if num else 2))
    empty: list[tuple[int, int]] = []
    for i, (no, level) in enumerate(heads):
        nxt_no, nxt_level = heads[i + 1] if i + 1 < len(heads) else (len(doc.lines) + 1, 0)
        if nxt_level > level:
            continue  # есть подразделы
        body = [bl.text for n in range(no + 1, nxt_no) if (bl := doc.line(n)) is not None]
        head_line = doc.line(no)
        title = re.sub(r"^#{1,6}\s*|[*_`]", "", head_line.text if head_line else "").strip()
        if re.match(r"^[1-5]\d\d\b", title):
            continue  # «#### 200. Лимит сохранён» — заголовок и есть описание кода ответа
        if not any(t.strip() and t.strip() != "---" for t in body):
            empty.append((i, no))
    # три и более пустых заголовка подряд одного уровня — это список, оформленный заголовками
    runs: list[list[tuple[int, int]]] = []
    for item in empty:
        if runs and item[0] == runs[-1][-1][0] + 1 and heads[item[0]][1] == heads[runs[-1][-1][0]][1]:
            runs[-1].append(item)
        else:
            runs.append([item])
    for run in runs:
        if len(run) >= 3:
            continue
        for _, no in run:
            hits.append(
                RuleHit(
                    rule="R-EMPTY",
                    criterion="set_complete",
                    severity="major",
                    title="Пустой раздел",
                    problem="Раздел объявлен в документе, но не содержит ни одного требования или описания.",
                    recommendation="Заполнить раздел или удалить его с пометкой «не применимо».",
                    evidence=[loc.line_evidence(no)],
                )
            )
    return hits


def check_money_units(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    minor = [ln.no for ln in doc.lines if _MINOR_UNIT.search(ln.text)]
    major = [ln.no for ln in doc.lines if _MAJOR_UNIT.search(ln.text) and not _is_table_sep(ln.text)]
    if not minor or not major:
        return []
    # в первую очередь показываем строки, где обе единицы встречаются рядом (одна сущность)
    both = [n for n in minor if (ln := doc.line(n)) is not None and _MAJOR_UNIT.search(ln.text)]
    evid_lines = (both or minor[:1]) + [n for n in major if n not in both][:1]
    return [
        RuleHit(
            rule="R-UNITS",
            criterion="consistent",
            severity="major",
            title="Денежные величины заданы в разных единицах",
            problem=(
                "Часть сумм задана в мелких единицах (тиын/копейки), часть — в основных (тенге/₸). "
                "Без явного правила пересчёта и округления сравнение суммы с порогом может дать "
                "ошибку в 100 раз или на границе."
            ),
            recommendation="Зафиксировать единую единицу хранения и сравнения (например, тиын, целое) и правило конвертации.",
            evidence=[loc.line_evidence(n) for n in evid_lines[:3]],
        )
    ]


def check_http(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    hits = []
    method_lines = [
        ln
        for ln in doc.lines
        if (re.search(r"REST type|Method|Метод|Тип запроса|HTTP[- ]метод|Verb", ln.text, re.I) and _HTTP_METHOD.search(ln.text))
        or re.match(r"^\s*[*`]*(GET|POST|PUT|PATCH|DELETE)[*`]*\s+`?/[\w/{}.-]+", ln.text)  # «GET /transfers/create»
    ]
    for ml in method_lines:
        m_method = _HTTP_METHOD.search(ml.text)
        if not m_method or m_method.group(1) != "GET":
            continue
        # ищем признаки изменения состояния в том же разделе API
        scope = [ln for ln in doc.lines if ln.section.split(" › ")[0] == ml.section.split(" › ")[0]]
        mut = [
            ln
            for ln in scope
            if _MUTATING_URL.search(ln.text)
            or (re.search(r"Description|Описание|Назначение", ln.text, re.I) and _MUTATING_WORD.search(ln.text))
        ]
        if mut:
            hits.append(
                RuleHit(
                    rule="R-HTTP-GET",
                    criterion="appropriate",
                    severity="critical",
                    title="GET используется для изменяющей операции",
                    problem=(
                        "Метод GET по HTTP-семантике безопасен и идемпотентен: его кэшируют прокси, "
                        "повторяют браузеры и сканеры, параметры попадают в логи и историю. "
                        "Изменение настроек через GET приводит к непреднамеренным сохранениям и утечке параметров."
                    ),
                    recommendation="Использовать POST/PUT (или PATCH) с телом запроса; GET оставить только для чтения.",
                    evidence=[loc.line_evidence(ml.no)] + [loc.line_evidence(m.no) for m in mut[:2]],
                )
            )
        body = [ln for ln in scope if re.search(r"Content-Type", ln.text, re.I) and "json" in ln.text.lower()]
        if body:
            hits.append(
                RuleHit(
                    rule="R-HTTP-BODY",
                    criterion="consistent",
                    severity="minor",
                    title="Для GET-запроса указан Content-Type тела",
                    problem="У GET-запроса нет тела, поэтому указание Content-Type запроса противоречит выбранному методу.",
                    recommendation="Убрать Content-Type запроса или сменить метод на POST/PUT с JSON-телом.",
                    evidence=[loc.line_evidence(body[0].no)],
                )
            )
    return hits


def check_status_codes(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    described: set[str] = set()
    mentioned: dict[str, int] = {}
    for ln in doc.lines:
        t = ln.text.strip()
        m = re.match(r"^(?:#{1,6}\s*|\*\*)\s*([1-5]\d\d)\b", t)
        if m:
            described.add(m.group(1))
            continue
        if re.search(r"Ответ|Статус|Status|Response|Код", ln.section, re.I) and not t.startswith(("{", "|", "`")):
            for code in re.findall(r"\b([1-5]\d\d)\b", t):
                if re.search(r"результат|статус|код|возвращ|ответ|status|code", t, re.I):
                    mentioned.setdefault(code, ln.no)
    hits = []
    for code, no in sorted(mentioned.items()):
        if code not in described:
            hits.append(
                RuleHit(
                    rule="R-HTTP-CODE",
                    criterion="complete",
                    severity="major",
                    title=f"Код ответа {code} упомянут, но не описан",
                    problem=f"Код {code} фигурирует в тексте, но для него нет описания ситуации и примера ответа.",
                    recommendation=f"Описать, когда возвращается {code}, и привести пример тела ответа — или убрать упоминание.",
                    evidence=[loc.line_evidence(no)],
                )
            )
    return hits


def check_api_tables(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    """Тип одного и того же параметра в разных таблицах API; чувствительные данные в query."""
    formats: dict[str, dict[str, int]] = defaultdict(dict)
    sensitive: list[tuple[int, str]] = []
    header: list[str] | None = None
    for ln in doc.lines:
        t = ln.text.strip()
        if not t.startswith("|"):
            header = None
            continue
        if _is_table_sep(t):
            continue
        cells = _cells(t)
        low = [c.lower() for c in cells]
        if header is None:
            header = low
            continue
        pi = next((i for i, h in enumerate(header) if h in _NAME_COLS or h.startswith("param")), None)
        if pi is not None:
            fi = next((i for i, h in enumerate(header) if h in _TYPE_COLS), None)
            li = next((i for i, h in enumerate(header) if h in _LOCATION_COLS), None)
            if pi < len(cells) and cells[pi]:
                name = cells[pi]
                if fi is not None and fi < len(cells) and cells[fi]:
                    formats[name].setdefault(cells[fi].lower(), ln.no)
                in_query = bool(re.search(r"query|строк\w* запроса", ln.section, re.I)) or (
                    li is not None and li < len(cells) and re.search(r"query|url|строк", cells[li], re.I)
                )
                if in_query and _SENSITIVE_PARAM.search(name):
                    sensitive.append((ln.no, name))
    # чувствительные параметры прямо в примерах URL: «?phone=…», «&cardId=…»
    for ln in doc.lines:
        for m in re.finditer(r"[?&]([A-Za-z_][\w-]{1,40})=", ln.text):
            if _SENSITIVE_PARAM.search(m.group(1)) and all(m.group(1) != n for _, n in sensitive):
                sensitive.append((ln.no, m.group(1)))
    hits = []
    for name, fmts in formats.items():
        if len(fmts) > 1:
            hits.append(
                RuleHit(
                    rule="R-API-TYPE",
                    criterion="consistent",
                    severity="major",
                    title=f"Параметр «{name}» имеет разные типы: {', '.join(fmts)}",
                    problem=(
                        f"Один и тот же параметр «{name}» описан с разными форматами ({', '.join(fmts)}). "
                        "Клиент и сервер по-разному сериализуют значение, что ведёт к ошибкам парсинга и сравнения."
                    ),
                    recommendation="Привести формат параметра к одному типу во всех таблицах запроса и ответа.",
                    evidence=[loc.line_evidence(n) for n in list(fmts.values())[:3]],
                )
            )
    for no, name in sensitive:
        hits.append(
            RuleHit(
                rule="R-API-QUERY-PII",
                criterion="risk",
                severity="major",
                title=f"Идентификатор «{name}» передаётся в строке запроса",
                problem=(
                    "Параметры строки запроса сохраняются в логах веб-серверов, прокси, APM и истории — "
                    "идентификаторы карт/счетов/клиентов в URL создают риск утечки."
                ),
                recommendation=(
                    "Передавать идентификатор в теле запроса или в пути с маскированием в логах; "
                    "описать требования к логированию."
                ),
                evidence=[loc.line_evidence(no)],
            )
        )
    return hits


# --- шаблон ТЗ финтех-проекта (основная часть и дочерняя спецификация REST API)
TEMPLATE_MAIN = [
    ("общие сведения", "minor"),
    ("цели", "major"),
    ("архитектура решения", "minor"),
    ("дизайн процесса ui", "minor"),
    ("требования к изменениям", "minor"),
    ("bpmn", "minor"),
    ("функциональные требования", "critical"),
    ("sequence", "minor"),
    ("нефункциональные требования", "major"),
    ("требования к данным", "major"),
    ("требования к интеграции", "major"),
    ("требования к тестированию", "minor"),
    ("критерии приемки", "major"),
    ("ревью требований", "minor"),
    ("блок согласующих", "minor"),
]
TEMPLATE_API = [
    ("общие параметры", "minor"),
    ("логика работы метода", "major"),
    ("параметры запроса", "major"),
    ("ответ", "major"),
    ("статусы", "major"),
]


def _norm_heading(t: str) -> str:
    t = re.sub(r"^#{1,6}\s*", "", t)
    return re.sub(r"[*_`]", "", t).lower().replace("ё", "е").strip()


def check_template_sections(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    """Если документ оформлен по шаблону ТЗ, проверяем, что обязательные разделы на месте."""
    # заголовком считаем и строку, целиком выделенную жирным (PDF/DOCX часто теряют уровни заголовков)
    heads = [
        (ln.no, _norm_heading(ln.text))
        for ln in doc.lines
        if ln.heading or re.match(r"^#{1,6}\s", ln.text) or re.fullmatch(r"\*\*[^*|]{3,80}\*\*", ln.text.strip())
    ]
    if not heads:
        return []

    def present(key: str) -> bool:
        return any(key in h or (len(h) > 4 and h in key) for _, h in heads)

    hits = []
    for name, template, min_found in (("основной части ТЗ", TEMPLATE_MAIN, 6), ("спецификации API", TEMPLATE_API, 3)):
        found = [k for k, _ in template if present(k)]
        if len(found) < min_found:
            continue  # документ не по этому шаблону — не навязываем структуру
        missing = [(k, sev) for k, sev in template if k not in found]
        if not missing:
            continue
        worst = min((sev for _, sev in missing), key=lambda s: {"critical": 0, "major": 1, "minor": 2}[s])
        anchor = (
            heads[0][0]
            if name == "основной части ТЗ"
            else next((no for no, h in heads if any(k in h for k, _ in template)), heads[0][0])
        )
        names = ", ".join(f"«{k}»" for k, _ in missing)
        hits.append(
            RuleHit(
                rule="R-TEMPLATE",
                criterion="set_complete",
                severity=worst,
                title=f"В {name} нет обязательных разделов шаблона: {names}",
                problem=(
                    f"Документ оформлен по шаблону ТЗ (найдено разделов шаблона: {len(found)} из {len(template)}), "
                    f"но разделы {names} отсутствуют. Требования, которые должны в них находиться, не заданы — "
                    "их нельзя реализовать и проверить."
                ),
                recommendation="Добавить недостающие разделы и заполнить их или явно указать «не применимо» с обоснованием.",
                evidence=[loc.line_evidence(anchor)],
            )
        )
    return hits


_REQ_ID = re.compile(r"(?<![\w-])([A-ZА-ЯЁ]{1,6})[-‑–](\d{1,3})(?![\w-])")
_DEF_ID = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?\**\s*([A-ZА-ЯЁ]{1,6})[-‑–](\d{1,3})\.?\**(?=[\s.:)]|$)")


def check_requirement_ids(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    """Нумерация требований: дубли, пропуски, ссылки на несуществующие требования."""
    defined: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for ln in doc.lines:
        t = ln.text.strip()
        if t.startswith("|"):
            cells = _cells(t)
            m = _DEF_ID.match(cells[0]) if cells else None
        else:
            m = _DEF_ID.match(t)
        if m:
            defined[m.group(1)][int(m.group(2))].append(ln.no)
    hits = []
    for prefix, nums in defined.items():
        if len(nums) < 2:
            continue  # одиночный «ID» — скорее всего не система нумерации
        for n, lines in sorted(nums.items()):
            if len(lines) > 1:
                hits.append(
                    RuleHit(
                        rule="R-ID-DUP",
                        criterion="traceable",
                        severity="major",
                        title=f"Идентификатор {prefix}-{n} используется несколько раз",
                        problem=(
                            f"Требование {prefix}-{n} определено в {len(lines)} местах: "
                            "ссылки на него неоднозначны, трассировка ломается."
                        ),
                        recommendation="Присвоить требованиям уникальные идентификаторы.",
                        evidence=[loc.line_evidence(x) for x in lines[:3]],
                    )
                )
        present = sorted(nums)
        gaps = [n for n in range(present[0], present[-1] + 1) if n not in nums]
        if gaps:
            after = max(n for n in present if n < gaps[0])
            hits.append(
                RuleHit(
                    rule="R-ID-GAP",
                    criterion="traceable",
                    severity="minor",
                    title=f"Пропуск в нумерации требований: нет {', '.join(f'{prefix}-{g}' for g in gaps)}",
                    problem=(
                        f"Требования {prefix} пронумерованы с пропуском ({', '.join(f'{prefix}-{g}' for g in gaps)}). "
                        "Возможно, требование удалено или потеряно при редактировании; "
                        "ссылки на него в смежных документах станут висячими."
                    ),
                    recommendation="Восстановить пропущенное требование или явно пометить номер как исключённый.",
                    evidence=[loc.line_evidence(nums[after][0])],
                )
            )
        # ссылки на несуществующие требования того же префикса
        for ln in doc.lines:
            for m in _REQ_ID.finditer(ln.text):
                if m.group(1) == prefix and int(m.group(2)) not in nums:
                    hits.append(
                        RuleHit(
                            rule="R-ID-REF",
                            criterion="traceable",
                            severity="major",
                            title=f"Ссылка на несуществующее требование {m.group(0)}",
                            problem=(
                                f"В тексте упоминается {m.group(0)}, "
                                "но требование с таким идентификатором в документе не определено."
                            ),
                            recommendation="Исправить ссылку или добавить требование.",
                            evidence=[loc.line_evidence(ln.no)],
                        )
                    )
    return hits


# --- чек-лист ревью финтех-ТЗ: измеримость НФТ, часовой пояс, идемпотентность, этап обработки операции
_NFR_SECTION = re.compile(r"нефункциональн|производительн|надёжност|надежност|доступност|\bNFR\b|\bSLA\b", re.I)
_PERIOD = re.compile(
    r"(?<![\w-])(?:месяц\w*|сутк\w*|календарн\w*|ежедневн\w*|ежемесячн\w*|в день|за день|начал\w* (?:дня|месяца|суток)|"
    r"конц\w* (?:дня|месяца|суток)|полуноч\w*)(?![\w-])",
    re.I,
)
_TIMEZONE = re.compile(
    r"UTC|GMT|часов\w* пояс|по времени|Алматы|Астан|Нур-Султан|MSK|московск\w* времен|[+−-]0?[1-9]:?00\b", re.I
)
_REPEAT = re.compile(
    r"повторн\w* (?:доставк|нажати|отправк|запрос|вызов|событи|сохранени|списани|платеж|перевод)\w*|"
    r"двойн\w* (?:списани|начислени|отправк|перевод|платеж)\w*|дубл\w* (?:операц|событи|списани|платеж)\w*|идемпотент\w*",
    re.I,
)
_IDEMPOTENCY_MECHANISM = re.compile(
    r"idempotency|idempotent[-_ ]?key|X-Request-Id|request[-_]?id|operation[-_]?id|external[-_]?id|"
    r"ключ\w* (?:идемпотент|дедупликац)|уникальн\w* (?:ключ|идентификатор\w*) (?:запроса|операции|события)",
    re.I,
)
_TXN_VAGUE = re.compile(
    r"проведённ\w* (?:покупк|операц|транзакц|платеж)\w*|проведенн\w* (?:покупк|операц|транзакц|платеж)\w*|"
    r"успешн\w* (?:покупк|операц|транзакц|платеж)\w*",
    re.I,
)
_TXN_STAGE_DEFINED = re.compile(
    r"авторизаци\w* (?:покупк|операц|транзакц|платеж|по карт)|авторизованн\w* (?:операц|покупк|транзакц)|клиринг|"
    r"\bholds?\b|холд|блокировк\w* (?:средств|суммы)|settlement|clearing|финансов\w* подтвержд",
    re.I,
)
_CARD_DOMAIN = re.compile(r"карт\w*|card|процессинг|POS|эквайринг", re.I)


def _content_lines(doc: Document):
    """Строки с содержанием: без заголовков, разделителей таблиц, кода и служебных разделов."""
    in_code = False
    for ln in doc.lines:
        t = ln.text.strip()
        if t.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not t or ln.heading or t.startswith("#") or _is_table_sep(t) or t == "---":
            continue
        if re.search(r"общие сведения|ревью требований|блок согласующ", ln.section, re.I):
            continue
        yield ln


def check_fintech_checklist(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    lines = list(_content_lines(doc))
    full = "\n".join(ln.text for ln in doc.lines)
    hits = []

    nfr = [
        ln
        for ln in lines
        if _NFR_SECTION.search(ln.section)
        and not re.search(r"\d", ln.text)
        and not any(rx.search(ln.text) for rx, _ in PLACEHOLDER)
    ]
    for ln in nfr[:3]:
        hits.append(
            RuleHit(
                rule="R-NFR-METRIC",
                criterion="verifiable",
                severity="major",
                title="Нефункциональное требование без числового показателя",
                problem=(
                    "В требовании нет ни одного числа: не задан измеримый порог (время, доля запросов, нагрузка, "
                    "доступность), поэтому выполнение нельзя объективно проверить нагрузочным или иным тестом."
                ),
                recommendation="Задать метрику, порог, долю (например, p95), нагрузку и точку измерения.",
                evidence=[loc.line_evidence(ln.no)],
            )
        )

    if not _TIMEZONE.search(full):
        period = [ln for ln in lines if _PERIOD.search(ln.text) and not _NFR_SECTION.search(ln.section)]
        if period:
            hits.append(
                RuleHit(
                    rule="R-TIMEZONE",
                    criterion="unambiguous",
                    severity="major",
                    title="Границы периода заданы без часового пояса",
                    problem=(
                        "Требования опираются на календарный период (день, месяц), но в документе не указан часовой пояс. "
                        "Операции около полуночи будут отнесены к разным периодам в зависимости от сервера — лимиты и "
                        "счётчики разойдутся."
                    ),
                    recommendation="Указать часовой пояс и точное правило границы периода (например, 00:00:00–23:59:59 UTC+5).",
                    evidence=[loc.line_evidence(ln.no) for ln in period[:2]],
                )
            )

    repeats = [ln for ln in lines if _REPEAT.search(ln.text)]
    if repeats and not _IDEMPOTENCY_MECHANISM.search(full):
        hits.append(
            RuleHit(
                rule="R-IDEMPOTENCY",
                criterion="complete",
                severity="major",
                title="Защита от повторов заявлена, но механизм не задан",
                problem=(
                    "Документ требует корректно обрабатывать повторы (повторная доставка, повторное нажатие, двойное "
                    "списание), но не определяет ключ идемпотентности или дедупликации и где он передаётся. "
                    "Реализации будут разными, а двойные операции — возможны."
                ),
                recommendation=(
                    "Определить ключ идемпотентности (например, идентификатор операции или заголовок Idempotency-Key), "
                    "срок его хранения и ответ на повтор."
                ),
                evidence=[loc.line_evidence(ln.no) for ln in repeats[:2]],
            )
        )

    vague = [ln for ln in lines if _TXN_VAGUE.search(ln.text)]
    if vague and _CARD_DOMAIN.search(full) and not _TXN_STAGE_DEFINED.search(full):
        m_word = _TXN_VAGUE.search(vague[0].text)
        word = m_word.group(0) if m_word else vague[0].text
        hits.append(
            RuleHit(
                rule="R-TXN-STAGE",
                criterion="unambiguous",
                severity="major",
                title=f"Не определён этап обработки операции: «{word}»",
                problem=(
                    f"Формулировка «{word}» не говорит, учитывается ли операция при авторизации (блокировка средств) "
                    "или после клиринга (финансовое подтверждение). От этого зависят суммы, даты отнесения к периоду "
                    "и обработка отмен."
                ),
                recommendation=(
                    "Явно указать этап (авторизация или клиринг) и правило для отменённых и частично подтверждённых операций."
                ),
                evidence=[loc.line_evidence(vague[0].no)],
            )
        )
    return hits


# --- одна и та же денежная величина с разными границами в разных местах документа
_NUM = r"\d{1,3}(?:[   ]\d{3})+|\d+"
_RANGE = re.compile(rf"(?<![\d:.,])(?:от\s+)?({_NUM})\s*(?:–|—|-|до)\s*({_NUM})(?![\d:.,]\d)")
_MONEY_MARK = re.compile(r"₸|тенге|\bKZT\b|тиын|руб|\$|сумм|порог|лимит|комисси|amount|limit|fee", re.I)
_QUANTITY_FAMILIES = {
    "сумма": re.compile(r"сумм|amount|\bsum\b", re.I),
    "порог/лимит": re.compile(r"порог|лимит|threshold|\blimit", re.I),
    "комиссия": re.compile(r"комисси|\bfee\b", re.I),
}
_FAMILY_PRIORITY = ["комиссия", "порог/лимит", "сумма"]  # от самой конкретной величины к самой общей


def _to_int(s: str) -> int:
    return int(re.sub(r"\D", "", s))


def check_range_consistency(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    """«Ф-2: от 100 до 500 000 ₸» против «API: amount 1–1 000 000» — у одной величины разные границы."""
    by_family: dict[str, list[tuple[tuple[int, int], int, str]]] = defaultdict(list)
    for ln in _content_lines(doc):
        if not _MONEY_MARK.search(ln.text):
            continue
        for m in _RANGE.finditer(ln.text):
            lo, hi = _to_int(m.group(1)), _to_int(m.group(2))
            if lo >= hi or hi < 10:  # «1–3», перевёрнутые и вырожденные диапазоны — не границы сумм
                continue
            # диапазон относится к самой конкретной величине, названной в строке: «Комиссия 1% от суммы,
            # от 50 до 1 000» — это комиссия; «amount … Порог в тенге» — порог; «сумма» — только если иного нет
            family = next((f for f in _FAMILY_PRIORITY if _QUANTITY_FAMILIES[f].search(ln.text)), None)
            if family:
                by_family[family].append(((lo, hi), ln.no, m.group(0).strip()))
    hits = []
    reported: set[frozenset[int]] = set()
    for family, items in by_family.items():
        bounds = {b for b, _, _ in items}
        if len(bounds) < 2:
            continue
        lines = frozenset(no for _, no, _ in items)
        if lines in reported:  # та же пара строк уже описана для другой величины
            continue
        reported.add(lines)
        first_by_bounds: dict[tuple[int, int], tuple[int, str]] = {}
        for b, no, text in items:
            first_by_bounds.setdefault(b, (no, text))
        shown = list(first_by_bounds.items())[:3]
        ranges = "; ".join(f"«{text}» (строка {no})" for _, (no, text) in shown)
        hits.append(
            RuleHit(
                rule="R-RANGE",
                criterion="consistent",
                severity="critical",
                title=f"Разные границы величины «{family}» в разных местах документа",
                problem=(
                    f"Для одной и той же величины ({family}) документ задаёт разные диапазоны: {ranges}. "
                    "Валидация в интерфейсе, API и проверке требований будет расходиться: значение, допустимое "
                    "в одном месте, будет отклонено в другом."
                ),
                recommendation="Оставить один диапазон, указать его в одном месте и ссылаться на него из остальных разделов.",
                evidence=[loc.line_evidence(no) for _, (no, _) in shown],
            )
        )
    return hits


ALL_RULES = [
    check_language,
    check_placeholders,
    check_empty_sections,
    check_template_sections,
    check_requirement_ids,
    check_fintech_checklist,
    check_range_consistency,
    check_money_units,
    check_http,
    check_status_codes,
    check_api_tables,
]


def run_rules(doc: Document, loc: QuoteLocator) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for rule in ALL_RULES:
        hits.extend(rule(doc, loc))
    for h in hits:
        for e in h.evidence:
            if not e.quote:
                ln = doc.line(e.line_start) if e.line_start else None
                e.quote = clean_line(ln.text) if ln else ""
    return hits
