"""'Export diagnostics bundle': a zip of the logs (always PII-scrubbed), the activity feed, the redacted settings,
the last-operation breadcrumb and system information. Meant to be attached to a bug report."""
from __future__ import annotations

import json
import platform
import sys
import zipfile
from pathlib import Path

from applog import scrub_pii

_SKIP_SETTINGS = ("resume.path", "knowledge.folder")      # file-system paths can contain a user name


def _scrub(text: str) -> str:
    return scrub_pii(text)


def export_diagnostics_bundle(dest, log_dir, bus=None, settings=None, redact: bool = True) -> list[str]:
    names: list[str] = []
    log_dir = Path(log_dir) if log_dir else None
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name: str, text: str) -> None:
            z.writestr(name, _scrub(text) if redact else text)
            names.append(name)
        if log_dir and log_dir.is_dir():
            for f in sorted(log_dir.iterdir()):
                if f.is_file() and f.suffix in (".log", ".json", ".txt") or f.name.endswith(".log.1"):
                    try:
                        put(f"logs/{f.name}", f.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        pass
        if bus is not None:
            put("activity.txt", "\n".join(e.line() + (f"\n    {e.detail}" if e.detail and e.detail != e.message else "")
                                          for e in bus.events()))
        if settings is not None:
            from settings.settings_manager import DEFAULTS
            data = {f"{s}.{k}": settings.get(f"{s}.{k}") for s, d in DEFAULTS.items() for k in d
                    if f"{s}.{k}" not in _SKIP_SETTINGS}
            put("settings.json", json.dumps(data, indent=1, default=str))
        import PySide6
        put("system.txt", f"python {sys.version}\nplatform {platform.platform()}\nPySide6 {PySide6.__version__}\n"
                          f"frozen {getattr(sys, 'frozen', False)}\n")
    return names
