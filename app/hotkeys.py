"""Global, configurable keyboard shortcuts (work while the meeting window has focus)."""
import logging

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class HotkeyManager(QObject):
    triggered = Signal(str)

    def __init__(self):
        super().__init__()
        self._handles = []

    def register(self, mapping: dict[str, str]) -> None:
        self.unregister()
        try:
            import keyboard
        except ImportError:
            log.warning("'keyboard' package missing: global shortcuts disabled")
            return
        for action, combo in mapping.items():
            try:
                self._handles.append(keyboard.add_hotkey(combo.lower().replace(" ", ""),
                                                         lambda a=action: self.triggered.emit(a)))
            except Exception as e:
                log.warning("could not register %s (%s): %s", action, combo, e)

    def unregister(self) -> None:
        try:
            import keyboard
            for h in self._handles:
                keyboard.remove_hotkey(h)
        except Exception:
            pass
        self._handles.clear()
