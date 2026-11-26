"""Пересборка интерактивного HTML из сохранённого отчёта:

python -m revizor.report reports/report.json examples/requirements.md [reports/report.html]
"""

import sys

from .html import main

sys.exit(main())
