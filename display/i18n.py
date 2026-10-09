"""String wrapping for future translation: `tr("Save")`. Falls back to the English text when no translation is loaded."""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication


def tr(text: str, context: str = "app") -> str:
    return QCoreApplication.translate(context, text)
