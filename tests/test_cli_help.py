"""Справка и ошибки параметров командной строки — по-русски и с подсказкой."""

import pytest

from revizor.__main__ import build_parser


def test_help_is_russian_with_examples():
    text = build_parser().format_help()
    assert text.startswith("Запуск: python -m revizor")
    assert "аргументы:" in text and "параметры:" in text and "примеры:" in text
    assert "--replay" in text and "--answers reports/answers.json" in text
    for english in ("usage:", "show this help", "positional arguments", "options:"):
        assert english not in text


def test_bad_parameter_value_is_explained_in_russian(capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["in.md", "out.md", "--max-calls", "abc"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "параметр --max-calls: нужно целое число" in err and "python -m revizor --help" in err
