"""Filesystem locations. Portable mode: create 'portable.flag' next to the exe."""
import os
import sys
from pathlib import Path


def _exe_dir() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    if (_exe_dir() / "portable.flag").exists():
        p = _exe_dir() / "data"
    elif os.environ.get("APPDATA"):
        p = Path(os.environ["APPDATA"]) / "DualScreenAI"
    else:
        p = Path.home() / ".dualscreen_ai"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resource_dir() -> Path:
    """Read-only bundled files (icons, fonts). PyInstaller unpacks them under sys._MEIPASS."""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else Path(__file__).resolve().parents[1]


def cache_dir() -> Path:
    p = app_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
