"""Версия агента для паспорта прогона: тег/коммит git + версия пакета."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5, encoding="utf-8"
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def agent_version() -> str:
    describe = _git("describe", "--tags", "--always", "--dirty")
    if describe:
        commit = _git("rev-parse", "--short", "HEAD") or ""
        extra = f", коммит {commit}" if commit and commit not in describe else ""
        return f"revizor {__version__} (git {describe}{extra})"
    if _git("rev-parse", "--git-dir"):
        return f"revizor {__version__} (git: коммитов ещё нет)"
    return f"revizor {__version__} (вне git-репозитория)"
