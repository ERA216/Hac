"""Версия агента для паспорта прогона: коммит (и тег) git + версия пакета."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent

# выходные файлы прогона: их изменение не меняет версию кода агента
OUTPUT_PATHS = [":(exclude)reports", ":(exclude)eval/RESULTS.md", ":(exclude)eval/cases/*/RESULTS.md"]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def code_is_dirty() -> bool:
    """Есть ли незакоммиченные изменения кода (без учёта отчётов и кэша ответов модели в reports/)."""
    return _git("status", "--porcelain", "--untracked-files=no", "--", ".", *OUTPUT_PATHS) is not None


def agent_version() -> str:
    commit = _git("rev-parse", "--short", "HEAD")
    if commit:
        tag = _git("describe", "--tags", "--exact-match", "HEAD")
        ref = f"{tag}, коммит {commit}" if tag else commit
        dirty = ", есть незакоммиченные изменения кода" if code_is_dirty() else ""
        return f"revizor {__version__} (git {ref}{dirty})"
    if _git("rev-parse", "--git-dir"):
        return f"revizor {__version__} (git: коммитов ещё нет)"
    return f"revizor {__version__} (вне git-репозитория)"
