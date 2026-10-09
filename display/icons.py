"""Offline line-icon set: bundled SVG files (display/assets/icons) recoloured to the current theme at run time.

    icon("mic")                       QIcon in the theme's text colour (disabled / hover states included)
    icon("check", color="#fff")       explicit colour
    pixmap("warning", "#f5a623", 16)  QPixmap (cached)
    icon_file("chevron_down", color)  path of a recoloured SVG on disk (for QSS `image: url(...)`)
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from app.paths import resource_dir

log = logging.getLogger("app.icons")
ICON_DIR = resource_dir() / "display" / "assets" / "icons"
_cache_dir: Path | None = None


def icon_names() -> list[str]:
    return sorted(p.stem for p in ICON_DIR.glob("*.svg"))


@lru_cache(maxsize=None)
def svg_text(name: str) -> str:
    p = ICON_DIR / f"{name}.svg"
    if not p.exists():
        log.warning("missing icon %r", name)
        p = ICON_DIR / "info.svg"
    return p.read_text(encoding="utf-8")


def has_icon(name: str) -> bool:
    return (ICON_DIR / f"{name}.svg").exists()


def _colored(name: str, color: str) -> bytes:
    return svg_text(name).replace("currentColor", color).encode("utf-8")


@lru_cache(maxsize=1024)
def _pixmap(name: str, color: str, size: int, dpr: float) -> QPixmap:
    px = int(round(size * dpr))
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    r = QSvgRenderer(QByteArray(_colored(name, color)))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, px, px))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(dpr)
    return pm


def pixmap(name: str, color: str | None = None, size: int = 18, dpr: float | None = None) -> QPixmap:
    if color is None:
        from display.theme import current
        color = current()["text_muted"]
    if dpr is None:
        from PySide6.QtGui import QGuiApplication
        s = QGuiApplication.primaryScreen()
        dpr = float(s.devicePixelRatio()) if s else 1.0
    return _pixmap(name, color, int(size), float(dpr))


def icon(name: str, color: str | None = None, size: int = 18, disabled: str | None = None, active: str | None = None) -> QIcon:
    """Theme-aware QIcon. Without arguments uses the current theme's text colours."""
    if color is None:
        from display.theme import current
        t = current()
        color, disabled, active = t["text_muted"], t["text_faint"], active or t["accent"]
    ic = QIcon()
    ic.addPixmap(pixmap(name, color, size), QIcon.Normal)
    ic.addPixmap(pixmap(name, active or color, size), QIcon.Active)
    ic.addPixmap(pixmap(name, active or color, size), QIcon.Selected)
    ic.addPixmap(pixmap(name, disabled or color, size), QIcon.Disabled)
    return ic


def icon_file(name: str, color: str) -> str:
    """A recoloured copy of the icon on disk, for stylesheets (`image: url(...)`)."""
    global _cache_dir
    if _cache_dir is None:
        try:
            from app.paths import cache_dir
            _cache_dir = cache_dir() / "icons"
        except Exception:
            import tempfile
            _cache_dir = Path(tempfile.gettempdir()) / "dsai_icons"
        _cache_dir.mkdir(parents=True, exist_ok=True)
    data = _colored(name, color)
    f = _cache_dir / f"{name}-{hashlib.sha1(data).hexdigest()[:10]}.svg"
    if not f.exists():
        f.write_bytes(data)
    return f.as_posix()
