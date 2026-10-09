"""Header health indicators: Mic · System audio · STT · LLM · Resume · Knowledge base — each a small icon with a
green / amber / red / grey status dot; click to open the relevant settings page or run its quick test."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QHBoxLayout, QToolButton, QWidget

from app.gui_thread import assert_gui_thread
from app.health import HealthItem
from display import icons, theme

STATE_WORD = {"ok": "OK", "warn": "needs attention", "error": "problem", "idle": "inactive"}


class HealthButton(QToolButton):
    def __init__(self, item: HealthItem, parent=None):
        super().__init__(parent)
        self.item = item
        self.setProperty("variant", "ghost")
        self.setProperty("icononly", True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(34, 34)
        self.apply(item)

    def apply(self, item: HealthItem) -> None:
        self.item = item
        t = theme.current()
        self.setIcon(icons.icon(item.icon or "info", t["text_muted"], 18))
        self.setToolTip(f"{item.label}: {item.message}\n({STATE_WORD.get(item.state, item.state)} — click for details)")
        self.setAccessibleName(f"{item.label}: {STATE_WORD.get(item.state, item.state)}. {item.message}")
        self.update()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        t = theme.current()
        col = {"ok": t["success"], "warn": t["warning"], "error": t["danger"]}.get(self.item.state, t["text_faint"])
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(col))
        p.setPen(QColor(t["surface"]))
        p.drawEllipse(QRectF(self.width() - 14, self.height() - 14, 10, 10))
        if self.item.state in ("error", "warn"):           # shape as well as colour
            p.setPen(QColor(t["surface"]))
            f = p.font()
            f.setPixelSize(8)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(self.width() - 14, self.height() - 14, 10, 10), Qt.AlignCenter, "!")


class HealthBar(QWidget):
    activated = Signal(str)                      # "page:Resume" / "test:llm"

    def __init__(self, health_fn, bus=None, parent=None):
        super().__init__(parent)
        self.health_fn = health_fn
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.buttons: dict[str, HealthButton] = {}
        self._lay = lay
        self._debounce = QTimer(self, singleShot=True, interval=120, timeout=self.refresh)
        self.setAccessibleName("System health")
        self.refresh()
        if bus is not None:
            unsub = bus.subscribe(self._on_event)
            self.destroyed.connect(lambda *_: unsub())
        theme.on_changed(self, self.refresh)

    def _on_event(self, ev) -> None:
        if ev.stage in ("audio", "stt_model", "llm_model", "kb", "resume", "vad_model"):
            self._debounce.start()

    def refresh(self) -> None:
        assert_gui_thread("HealthBar.refresh")
        try:
            items = self.health_fn()
        except Exception:
            items = []
        for it in items:
            b = self.buttons.get(it.key)
            if b is None:
                b = HealthButton(it)
                b.clicked.connect(lambda _=False, k=it.key: self.activated.emit(self.buttons[k].item.action))
                self.buttons[it.key] = b
                self._lay.addWidget(b)
            else:
                b.apply(it)

    def states(self) -> dict[str, str]:
        return {k: b.item.state for k, b in self.buttons.items()}
