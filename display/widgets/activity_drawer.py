"""Activity drawer (Ctrl+L): a slide-out, filterable live feed of everything the app does."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from app.status import LEVELS, StatusBus, StatusEvent
from display import icons, theme
from display.widgets.common import SearchBox, button, icon_button, label
from display.widgets.feedback import copy_text

LEVEL_ICON = {"DEBUG": "terminal", "INFO": "info", "WARNING": "warning", "ERROR": "error"}
LEVEL_TOKEN = {"DEBUG": "text_faint", "INFO": "info", "WARNING": "warning", "ERROR": "danger"}
FILTERS = (("All events", "DEBUG"), ("Info and above", "INFO"), ("Warnings and errors", "WARNING"), ("Errors only", "ERROR"))
MAX_ROWS = 1200


class ActivityDrawer(QFrame):
    open_logs_requested = Signal()
    export_requested = Signal()

    WIDTH = 440

    def __init__(self, bus: StatusBus, host: QWidget):
        super().__init__(host)
        self.bus, self.host = bus, host
        self.setObjectName("card")
        self.setProperty("elevated", True)
        self._open = False
        self._stage = ""
        self._events: list[StatusEvent] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label("Activity", "section"))
        head.addStretch(1)
        self.stage_label = label("", "caption")
        head.addWidget(self.stage_label)
        head.addWidget(icon_button("x", "Close activity feed (Esc)", self.close_drawer))
        lay.addLayout(head)
        row = QHBoxLayout()
        self.level = QComboBox()
        for text, lv in FILTERS:
            self.level.addItem(text, lv)
        self.level.setCurrentIndex(1)
        self.level.setAccessibleName("Level filter")
        self.search = SearchBox("Filter events…")
        row.addWidget(self.level)
        row.addWidget(self.search, 1)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.setUniformItemSizes(True)
        self.list.setAccessibleName("Activity events")
        lay.addWidget(self.list, 1)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setProperty("mono", True)
        self.detail.setMaximumHeight(120)
        self.detail.setPlaceholderText("Select an event to see its details.")
        lay.addWidget(self.detail)
        btns = QHBoxLayout()
        self.btn_copy = button("Copy", "", "copy", "Copy the visible events")
        self.btn_clear = button("Clear", "ghost", "trash")
        self.btn_logs = button("Open log folder", "", "folder")
        self.btn_export = button("Export diagnostics", "", "download", "Export a redacted diagnostics bundle (zip)")
        for b in (self.btn_copy, self.btn_clear, self.btn_logs, self.btn_export):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.btn_copy.clicked.connect(lambda: copy_text(self.visible_text()))
        self.btn_clear.clicked.connect(self.clear)
        self.btn_logs.clicked.connect(self.open_logs_requested)
        self.btn_export.clicked.connect(self.export_requested)
        self.level.currentIndexChanged.connect(lambda _i: self.rebuild())
        self.search.textChanged.connect(lambda _t: self.rebuild())
        self.list.currentItemChanged.connect(self._show_detail)
        self._anim = QPropertyAnimation(self, b"geometry", self)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        unsub = bus.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: unsub())
        self.rebuild()
        self.hide()

    # ---- feed
    def _passes(self, ev: StatusEvent) -> bool:
        if not ev.feed:
            return False
        if LEVELS.get(ev.level, 20) < LEVELS[self.level.currentData()]:
            return False
        if self._stage and ev.stage != self._stage:
            return False
        q = self.search.text().strip().lower()
        return not q or q in ev.message.lower() or q in ev.detail.lower() or q in ev.stage

    def _item(self, ev: StatusEvent) -> QListWidgetItem:
        it = QListWidgetItem(ev.line())
        it.setIcon(icons.icon(LEVEL_ICON.get(ev.level, "info"), theme.current()[LEVEL_TOKEN.get(ev.level, "info")], 16))
        it.setData(Qt.UserRole, ev)
        it.setToolTip(ev.detail or ev.message)
        return it

    def _on_event(self, ev: StatusEvent) -> None:
        assert_gui_thread("ActivityDrawer")
        if not ev.feed:
            return
        self._events.append(ev)
        if len(self._events) > MAX_ROWS:
            del self._events[: len(self._events) - MAX_ROWS]
        if self._passes(ev):
            bottom = self.list.verticalScrollBar().value() >= self.list.verticalScrollBar().maximum() - 4
            self.list.addItem(self._item(ev))
            while self.list.count() > MAX_ROWS:
                self.list.takeItem(0)
            if bottom:
                self.list.scrollToBottom()

    def rebuild(self) -> None:
        self.list.clear()
        if not self._events:
            self._events = [e for e in self.bus.events("DEBUG")]
        for ev in self._events:
            if self._passes(ev):
                self.list.addItem(self._item(ev))
        self.list.scrollToBottom()

    def set_stage_filter(self, stage: str) -> None:
        self._stage = stage
        self.stage_label.setText(f"stage: {stage}" if stage else "")
        self.rebuild()

    def clear(self) -> None:
        self._events.clear()
        self.bus._events.clear()
        self.list.clear()
        self.detail.clear()

    def _show_detail(self, cur, _prev=None) -> None:
        ev = cur.data(Qt.UserRole) if cur else None
        self.detail.setPlainText("" if ev is None else f"{ev.line()}\n\n{ev.detail or ev.message}")

    def visible_text(self) -> str:
        return "\n".join(self.list.item(i).text() for i in range(self.list.count()))

    # ---- slide
    def is_open(self) -> bool:
        return self._open

    def _rect(self, shown: bool) -> QRect:
        h = self.host.height()
        top = 56
        w = min(self.WIDTH, max(300, self.host.width() - 40))
        x = self.host.width() - w - 12 if shown else self.host.width() + 4
        return QRect(x, top, w, max(200, h - top - 12))

    def open_drawer(self) -> None:
        assert_gui_thread("ActivityDrawer.open")
        if self._open:
            return
        self._open = True
        self.rebuild()
        self.show()
        self.raise_()
        self._slide(self._rect(False), self._rect(True))
        self.search.setFocus()

    def close_drawer(self) -> None:
        if not self._open:
            return
        self._open = False
        self._slide(self.geometry(), self._rect(False), hide_after=True)

    def toggle(self) -> None:
        self.close_drawer() if self._open else self.open_drawer()

    def _slide(self, a: QRect, b: QRect, hide_after: bool = False) -> None:
        self._anim.stop()
        if getattr(self, "_hide_conn", False):
            self._anim.finished.disconnect(self.hide)
            self._hide_conn = False
        dur = theme.duration("normal")
        self.setGeometry(a)
        if dur == 0:
            self.setGeometry(b)
            if hide_after:
                self.hide()
            return
        self._anim.setDuration(dur)
        self._anim.setStartValue(a)
        self._anim.setEndValue(b)
        if hide_after:
            self._anim.finished.connect(self.hide)
            self._hide_conn = True
        self._anim.start()

    def relayout(self) -> None:
        if self._open:
            self.setGeometry(self._rect(True))

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.close_drawer()
        else:
            super().keyPressEvent(e)
