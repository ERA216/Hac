"""Ключ Gemini для веб-интерфейса: поле ввода, переменная окружения и .env без перезапуска."""

import pytest

from revizor import config


@pytest.fixture
def env_dir(tmp_path, monkeypatch):
    """Вместо настоящего .env проекта — пустой каталог; переменной окружения с ключом нет."""
    monkeypatch.setattr(config, "_env_files", lambda: [tmp_path / ".env"])
    monkeypatch.setattr(config, "_SHELL_GEMINI_KEY", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return tmp_path


def test_entered_key_wins(env_dir):
    (env_dir / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert config.gemini_key("  typed-key  ") == "typed-key"


def test_placeholder_is_not_a_key(env_dir):
    (env_dir / ".env").write_text("GEMINI_API_KEY=your_gemini_api_key_here\n", encoding="utf-8")
    assert config.gemini_key() == ""
    assert config.gemini_key("your_gemini_api_key_here") == ""


def test_env_file_is_reread_without_restart(env_dir):
    assert config.gemini_key() == ""
    (env_dir / ".env").write_text("GEMINI_API_KEY=added-later\n", encoding="utf-8")
    assert config.gemini_key() == "added-later"


def test_shell_variable_beats_env_file(env_dir, monkeypatch):
    (env_dir / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setattr(config, "_SHELL_GEMINI_KEY", "from-shell")
    assert config.gemini_key() == "from-shell"


def test_llm_uses_key_from_web_field(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert config.llm_from_env(provider="gemini", cache_dir=None, api_key="typed-key")._backend_args[0] == "typed-key"
    assert config.llm_from_env(provider="gemini", cache_dir=None)._backend_args[0] == "from-env"


def test_web_app_key_field(env_dir):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=60).run()
    field = next(w for w in at.sidebar.text_input if w.label == "Ключ Gemini")
    assert field.proto.type == field.proto.PASSWORD  # ключ не виден на экране
    assert any("aistudio.google.com" in c.value for c in at.sidebar.caption)  # подсказка, где взять ключ

    at.button[0].click().run()  # «Взять пример ТЗ хакатона»
    next(b for b in at.button if b.label == "Проверить документ").click().run()
    assert any("Нет ключа Gemini" in e.value for e in at.error)

    (env_dir / ".env").write_text("GEMINI_API_KEY=added-later\n", encoding="utf-8")  # ключ вписали в .env на ходу
    at.run()
    field = next(w for w in at.sidebar.text_input if w.label == "Ключ Gemini")
    assert "уже задан" in field.placeholder
    assert not any("aistudio.google.com" in c.value for c in at.sidebar.caption)
