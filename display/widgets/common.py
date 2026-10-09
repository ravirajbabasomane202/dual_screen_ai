"""Small building blocks: labels, buttons with theme-aware icons, badges, chips, pulse dot, confidence ring, flow layout."""
from __future__ import annotations

import math
import weakref

from PySide6.QtCore import (QEasingCurve, QPoint, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer, Signal)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLayout, QLayoutItem, QLineEdit,
                               QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from display import icons, theme

try:
    import shiboken6
    _valid = shiboken6.isValid
except Exception:                                  # pragma: no cover
    _valid = lambda o: True

_BOUND: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_connected = [False]


def _refresh_icons(_t=None) -> None:
    for w, (name, size, token) in list(_BOUND.items()):
        if _valid(w):
            col = theme.current()[token] if token else None
            w.setIcon(icons.icon(name, col, size) if col else icons.icon(name, size=size))


def bind_icon(w, name: str, size: int = 18, token: str | None = None) -> None:
    """Give a button an icon that follows the theme (re-coloured when the theme / accent changes)."""
    _BOUND[w] = (name, size, token)
    col = theme.current()[token] if token else None
    w.setIcon(icons.icon(name, col, size) if col else icons.icon(name, size=size))
    w.setIconSize(QSize(size, size))
    if not _connected[0]:
        theme.manager().changed.connect(_refresh_icons)
        _connected[0] = True


def set_prop(w, name: str, value) -> None:
    if w.property(name) != value:
        w.setProperty(name, value)
        theme.repolish(w)


def label(text: str = "", role: str = "", wrap: bool = False, selectable: bool = False) -> QLabel:
    l = QLabel(text)
    if role:
        l.setProperty("role", role)
    l.setWordWrap(wrap)
    if selectable:
        l.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return l


def button(text: str = "", variant: str = "", icon: str = "", tooltip: str = "", shortcut: str = "", checkable: bool = False,
           on_click=None, size: int = 16) -> QPushButton:
    b = QPushButton(text)
    if variant:
        b.setProperty("variant", variant)
    if icon:
        bind_icon(b, icon, size, "accent_text" if variant == "primary" else ("danger" if variant == "danger" else None))
    tip = tooltip or text
    if shortcut:
        tip = f"{tip}  ({shortcut})" if tip else shortcut
    if tip:
        b.setToolTip(tip)
    if not text and tip:
        b.setAccessibleName(tooltip or icon)
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    if on_click:
        b.clicked.connect(on_click)
    return b


def icon_button(icon: str, tooltip: str, on_click=None, variant: str = "ghost", checkable: bool = False, size: int = 18,
                shortcut: str = "") -> QToolButton:
    b = QToolButton()
    b.setProperty("variant", variant)
    b.setProperty("icononly", True)
    bind_icon(b, icon, size)
    b.setToolTip(f"{tooltip}  ({shortcut})" if shortcut else tooltip)
    b.setAccessibleName(tooltip)
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    b.setFocusPolicy(Qt.StrongFocus)
    if on_click:
        b.clicked.connect(on_click)
    return b


def badge(text: str, tone: str = "neutral") -> QLabel:
    l = QLabel(text)
    l.setProperty("badge", True)
    l.setProperty("tone", tone)
    l.setAlignment(Qt.AlignCenter)
    return l


def set_badge(l: QLabel, text: str, tone: str) -> None:
    l.setText(text)
    set_prop(l, "tone", tone)


def divider(vertical: bool = False) -> QFrame:
    f = QFrame()
    f.setObjectName("vdivider" if vertical else "divider")
    return f


def card(elevated: bool = False) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    f.setProperty("elevated", elevated)
    lay = QVBoxLayout(f)
    lay.setContentsMargins(theme.SPACE["lg"], theme.SPACE["md"], theme.SPACE["lg"], theme.SPACE["md"])
    lay.setSpacing(theme.SPACE["sm"])
    return f, lay


def fade_in(w: QWidget, ms: int | None = None) -> None:
    """150–200 ms fade; a no-op with reduced motion."""
    dur = theme.duration("normal") if ms is None else ms
    if dur <= 0 or not w.isVisible():
        return
    eff = QGraphicsOpacityEffect(w)
    w.setGraphicsEffect(eff)
    a = QPropertyAnimation(eff, b"opacity", w)
    a.setDuration(dur)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.OutCubic)
    a.finished.connect(lambda: w.setGraphicsEffect(None) if _valid(w) else None)
    a.start(QPropertyAnimation.DeleteWhenStopped)


class PulseDot(QWidget):
    """A small status dot that pulses gently while something is happening (static with reduced motion)."""

    def __init__(self, diameter: int = 10, parent=None):
        super().__init__(parent)
        self._d, self._color, self._pulse, self._phase = diameter, theme.current()["text_faint"], False, 0.0
        self.setFixedSize(diameter + 8, diameter + 8)
        self._timer = QTimer(self, interval=60, timeout=self._tick)

    def set_state(self, color: str, pulsing: bool) -> None:
        assert_gui_thread("PulseDot.set_state")
        self._color, self._pulse = color, pulsing and not theme.reduced_motion()
        if self._pulse and not self._timer.isActive():
            self._timer.start()
        elif not self._pulse:
            self._timer.stop()
            self._phase = 0.0
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.08) % 1.0
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self._color)
        cx = cy = self.width() / 2
        if self._pulse:
            ring = QColor(c)
            ring.setAlphaF(max(0.0, 0.45 * (1 - self._phase)))
            r = self._d / 2 + 4 * self._phase
            p.setBrush(ring)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        p.setBrush(c)
        p.setPen(Qt.NoPen)
        r = self._d / 2
        p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))


class ConfidenceRing(QWidget):
    def __init__(self, size: int = 44, parent=None):
        super().__init__(parent)
        self._v, self._size = 0.0, size
        self.setFixedSize(size, size)
        self.setAccessibleName("Detection confidence")

    def set_value(self, v: float) -> None:
        assert_gui_thread("ConfidenceRing.set_value")
        self._v = max(0.0, min(1.0, float(v)))
        self.setToolTip(f"Detection confidence {self._v:.0%}")
        self.update()

    def value(self) -> float:
        return self._v

    def paintEvent(self, _e) -> None:
        t = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = 4
        r = QRectF(w / 2 + 1, w / 2 + 1, self._size - w - 2, self._size - w - 2)
        p.setPen(QPen(QColor(t["border_strong"]), w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(r, 0, 360 * 16)
        col = t["success"] if self._v >= 0.8 else (t["warning"] if self._v >= 0.55 else t["danger"])
        p.setPen(QPen(QColor(col), w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(r, 90 * 16, -int(360 * 16 * self._v))
        p.setPen(QColor(t["text"]))
        f = QFont(self.font())
        f.setPixelSize(max(9, self._size // 4))
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter, f"{round(self._v * 100)}")


class FlowLayout(QLayout):
    """Wrapping layout for chips (Qt's flow-layout example)."""

    def __init__(self, parent=None, h: int = 8, v: int = 8):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h, self._v = h, v
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return self._layout(QRect(0, 0, w, 0), True)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect, test: bool) -> int:
        x, y, line_h = rect.x(), rect.y(), 0
        for it in self._items:
            sz = it.sizeHint()
            if x + sz.width() > rect.right() and line_h > 0:
                x, y, line_h = rect.x(), y + line_h + self._v, 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), sz))
            x += sz.width() + self._h
            line_h = max(line_h, sz.height())
        return y + line_h - rect.y()

    def clear(self) -> None:
        while self._items:
            it = self._items.pop()
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()


class Chip(QFrame):
    """A small clickable pill with an optional icon: skills, sources, tags."""
    clicked = Signal(object)

    def __init__(self, text: str, icon: str = "", payload=None, tooltip: str = "", clickable: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("chip")
        self.payload = payload
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 10, 3)
        lay.setSpacing(6)
        if icon:
            ic = QLabel()
            ic.setPixmap(icons.pixmap(icon, theme.current()["text_muted"], 14))
            lay.addWidget(ic)
        self.text_label = QLabel(text)
        self.text_label.setProperty("role", "caption")
        lay.addWidget(self.text_label)
        if tooltip:
            self.setToolTip(tooltip)
        if clickable:
            self.setCursor(Qt.PointingHandCursor)
            self.setFocusPolicy(Qt.StrongFocus)
        self._clickable = clickable

    def mouseReleaseEvent(self, e) -> None:
        if self._clickable and e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.payload)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e) -> None:
        if self._clickable and e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.payload)
        else:
            super().keyPressEvent(e)


class SearchBox(QLineEdit):
    def __init__(self, placeholder: str = "Search…", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setAccessibleName(placeholder)
        from PySide6.QtGui import QAction
        self._act = QAction(self)
        self.addAction(self._act, QLineEdit.LeadingPosition)
        self._paint()
        theme.on_changed(self, self._paint)

    def _paint(self) -> None:
        self._act.setIcon(icons.icon("search", theme.current()["text_faint"], 16))


def hbox(*items, spacing: int = 8, margins=(0, 0, 0, 0), stretch_end: bool = False) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for it in items:
        if it is None:
            lay.addStretch(1)
        elif isinstance(it, QLayout):
            lay.addLayout(it)
        else:
            lay.addWidget(it)
    if stretch_end:
        lay.addStretch(1)
    return lay


def vbox(*items, spacing: int = 8, margins=(0, 0, 0, 0)) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for it in items:
        if it is None:
            lay.addStretch(1)
        elif isinstance(it, QLayout):
            lay.addLayout(it)
        else:
            lay.addWidget(it)
    return lay
