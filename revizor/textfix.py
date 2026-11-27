"""Исправление слов со смешанной раскладкой в свободном тексте модели.

Модель иногда пишет русское слово с латинской буквой-двойником («сменa» с латинской «a»): такое слово
не находится поиском и выглядит как опечатка. В словах, где кириллицы больше, латинские двойники
заменяются кириллическими. Цитаты и идентификаторы документа не трогаются — они должны совпадать
с документом буква в букву.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

_LATIN_TO_CYR = str.maketrans("aeopcxyAEOPCXBMHKT", "аеорсхуАЕОРСХВМНКТ")
_WORD = re.compile(r"[^\W\d_]+")
_CYR = re.compile(r"[а-яёА-ЯЁ]")
_LAT = re.compile(r"[a-zA-Z]")
_KEEP_FIELDS = ("quote", "source_id", "location")  # дословные цитаты, номера пунктов и разделы документа


def fix_word(word: str) -> str:
    cyr, lat = len(_CYR.findall(word)), len(_LAT.findall(word))
    if not cyr or not lat or lat >= cyr:
        return word
    fixed = word.translate(_LATIN_TO_CYR)
    return fixed if not _LAT.search(fixed) else word  # латиница без двойников (например, «z») — не трогаем


def fix_text(text: str) -> str:
    return _WORD.sub(lambda m: fix_word(m.group(0)), text)


def _fix(value: Any) -> Any:
    if isinstance(value, str):
        return fix_text(value)
    if isinstance(value, list):
        return [_fix(v) for v in value]
    if isinstance(value, BaseModel):
        fix_model(value)
    return value


def fix_model(obj: BaseModel) -> BaseModel:
    """Исправить строковые поля (на месте), кроме цитат и идентификаторов документа."""
    for name in type(obj).model_fields:
        if any(k in name for k in _KEEP_FIELDS):
            continue
        value = getattr(obj, name)
        fixed = _fix(value)
        if fixed is not value:
            object.__setattr__(obj, name, fixed)
    return obj
