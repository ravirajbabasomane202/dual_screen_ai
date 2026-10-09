"""Display detection and placement of the AI window. Never touches the meeting window."""
from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QGuiApplication, QScreen


@dataclass
class MonitorInfo:
    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    primary: bool

    @property
    def label(self) -> str:
        return f"Monitor {self.index + 1} — {self.width} × {self.height}" + (" (primary)" if self.primary else "")


class MonitorManager:
    """Monitors are ordered left-to-right, then top-to-bottom, so 'Monitor 2' is stable and predictable."""

    @staticmethod
    def _screens() -> list[QScreen]:
        return sorted(QGuiApplication.screens(), key=lambda s: (s.geometry().x(), s.geometry().y()))

    @classmethod
    def list_monitors(cls) -> list[MonitorInfo]:
        primary = QGuiApplication.primaryScreen()
        return [MonitorInfo(i, s.name(), s.geometry().x(), s.geometry().y(), s.geometry().width(),
                            s.geometry().height(), s is primary) for i, s in enumerate(cls._screens())]

    @classmethod
    def has_second_monitor(cls) -> bool:
        return len(cls._screens()) > 1

    @classmethod
    def screen_for(cls, index: int) -> QScreen:
        screens = cls._screens()
        if 0 <= index < len(screens):
            return screens[index]
        return screens[-1] if screens else QGuiApplication.primaryScreen()   # no such monitor: last one

    @classmethod
    def index_of_window(cls, window) -> int:
        center = window.frameGeometry().center()
        for i, s in enumerate(cls._screens()):
            if s.geometry().contains(center):
                return i
        return 0

    @classmethod
    def is_on_monitor(cls, window, index: int) -> bool:
        screens = cls._screens()
        index = min(max(index, 0), len(screens) - 1)
        return cls.index_of_window(window) == index

    @classmethod
    def place(cls, window, index: int, mode: str = "windowed", always_on_top: bool = True) -> QScreen:
        """mode: windowed | borderless | fullscreen."""
        screen = cls.screen_for(index)
        flags = window.windowFlags() & ~(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        if mode in ("borderless", "fullscreen"):
            flags |= Qt.FramelessWindowHint
        if always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        window.setWindowFlags(flags)
        window.setScreen(screen)
        geo: QRect = screen.geometry()
        if mode in ("borderless", "fullscreen"):
            window.setGeometry(geo)
        else:
            w, h = int(geo.width() * 0.6), int(geo.height() * 0.8)
            window.setGeometry(geo.x() + (geo.width() - w) // 2, geo.y() + (geo.height() - h) // 2, w, h)
        if mode == "fullscreen":
            window.showFullScreen()
        else:
            window.showNormal()
        handle = window.windowHandle()
        if handle is not None:
            handle.setScreen(screen)
            if mode == "fullscreen":
                window.setGeometry(geo)
        window.raise_()
        return screen
