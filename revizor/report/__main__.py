"""Пересборка отчёта (HTML и Markdown) из сохранённого report.json, без обращения к модели:

python -m revizor.report reports/report.json examples/requirements.md [reports/report.html]
"""

import sys

from .html import main

sys.exit(main())
