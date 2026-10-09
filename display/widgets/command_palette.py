"""Command palette (Ctrl+K): every action in one searchable list, plus the keyboard-shortcuts overlay ('?')."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from app.gui_thread import assert_gui_thread
from display import icons, theme
from display.widgets.common import SearchBox, label


@dataclass
class Command:
    title: str
    run: Callable[[], None]
    shortcut: str = ""
    icon: str = "command"
    keywords: str = ""


class CommandPalette(QDialog):
    def __init__(self, commands: list[Command], parent: QWidget | None = None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setObjectName("palette")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.commands = commands
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        self.search = SearchBox("Type a command…")
        self.list = QListWidget()
        self.list.setMinimumHeight(280)
        lay.addWidget(self.search)
        lay.addWidget(self.list)
        self.search.textChanged.connect(self.filter)
        self.search.returnPressed.connect(self.run_current)
        self.list.itemActivated.connect(lambda _i: self.run_current())
        self.list.itemClicked.connect(lambda _i: self.run_current())
        self.setAccessibleName("Command palette")
        self.filter("")

    def filter(self, text: str) -> None:
        q = text.lower().strip()
        self.list.clear()
        for c in self.commands:
            hay = f"{c.title} {c.keywords}".lower()
            if all(w in hay for w in q.split()):
                it = QListWidgetItem(c.title + (f"        {c.shortcut}" if c.shortcut else ""))
                it.setIcon(icons.icon(c.icon, theme.current()["text_muted"], 16))
                it.setData(Qt.UserRole, c)
                self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def run_current(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        cmd = it.data(Qt.UserRole)
        self.accept()
        cmd.run()

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Down, Qt.Key_Up):
            row = self.list.currentRow() + (1 if e.key() == Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
        else:
            super().keyPressEvent(e)

    def open_centered(self) -> None:
        assert_gui_thread("CommandPalette")
        if self.parent() is not None:
            p = self.parent().window()
            g = p.geometry()
            self.move(g.x() + (g.width() - self.width()) // 2, g.y() + 90)
        self.search.clear()
        self.show()
        self.search.setFocus()


class ShortcutsOverlay(QDialog):
    def __init__(self, shortcuts: list[tuple[str, str]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.addWidget(label("Keyboard shortcuts", "section"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        for r, (keys, what) in enumerate(shortcuts):
            k = QLabel(keys)
            k.setObjectName("kbd")
            grid.addWidget(k, r, 0, Qt.AlignLeft)
            grid.addWidget(QLabel(what), r, 1)
        lay.addLayout(grid)
        lay.addWidget(label("Press Esc to close.", "caption"))
