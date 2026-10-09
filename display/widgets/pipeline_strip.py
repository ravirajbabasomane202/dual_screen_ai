"""Pipeline strip: Audio → VAD → STT → Detect → Context → Retrieve → LLM → Validate → Display.

One chip per stage (icon + label, colour AND icon change with the state so colour is never the only signal), a tooltip
with the last message and timing, and a click that opens the detail of that stage. Driven by the StatusBus."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from app.gui_thread import assert_gui_thread
from app.status import ERROR, IDLE, OK, PIPELINE, RUNNING, STAGE_LABELS, WARN, StatusBus
from display import icons, theme
from display.widgets.common import PulseDot, set_prop

STAGE_ICONS = {"audio": "speaker", "vad": "waveform", "stt": "mic", "detect": "target", "context": "layers",
               "retrieve": "search", "llm": "brain", "validate": "check", "display": "eye"}
STATE_WORD = {IDLE: "idle", RUNNING: "running", OK: "done", WARN: "warning", ERROR: "error"}


class StageChip(QFrame):
    clicked = Signal(str)

    def __init__(self, stage: str, compact: bool = False, parent=None):
        super().__init__(parent)
        self.stage, self.state, self.compact = stage, IDLE, compact
        self.setObjectName("stage")
        self.setProperty("state", IDLE)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 10 if not compact else 8, 4)
        lay.setSpacing(6)
        self.dot = PulseDot(8)
        self.dot.hide()
        self.icon = QLabel()
        self.icon.setFixedSize(16, 16)
        self.name = QLabel(STAGE_LABELS[stage])
        self.name.setProperty("role", "caption")
        self.time = QLabel("")
        self.time.setProperty("role", "mono")
        lay.addWidget(self.dot)
        lay.addWidget(self.icon)
        if not compact:
            lay.addWidget(self.name)
            lay.addWidget(self.time)
        self.apply(IDLE, "", None)

    def apply(self, state: str, message: str, elapsed: float | None) -> None:
        t = theme.current()
        self.state = state
        set_prop(self, "state", state)
        col = {IDLE: t["text_faint"], RUNNING: t["info"], OK: t["success"], WARN: t["warning"], ERROR: t["danger"]}[state]
        glyph = {OK: "check", WARN: "warning", ERROR: "error"}.get(state, STAGE_ICONS[self.stage])
        self.icon.setPixmap(icons.pixmap(glyph, col, 16))
        self.dot.setVisible(state == RUNNING)
        self.dot.set_state(t["info"], state == RUNNING)
        self.time.setText(f"{elapsed:.2f}s" if (elapsed is not None and state in (OK, WARN, ERROR)) else "")
        label = STAGE_LABELS[self.stage]
        tip = f"{label} — {STATE_WORD[state]}"
        if message:
            tip += f"\n{message}"
        if elapsed is not None and state in (OK, WARN, ERROR):
            tip += f"\n{label} {elapsed:.2f} s"
        self.setToolTip(tip + "\n(click for details)")
        self.setAccessibleName(f"{label}: {STATE_WORD[state]}")
        self.setAccessibleDescription(message)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.stage)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.stage)
        else:
            super().keyPressEvent(e)


class PipelineStrip(QWidget):
    stage_clicked = Signal(str)

    def __init__(self, bus: StatusBus, compact: bool = False, parent=None):
        super().__init__(parent)
        self.bus, self.compact = bus, compact
        self.chips: dict[str, StageChip] = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._arrows: list[QLabel] = []
        for i, stage in enumerate(PIPELINE):
            chip = StageChip(stage, compact)
            chip.clicked.connect(self.stage_clicked)
            self.chips[stage] = chip
            lay.addWidget(chip)
            if i < len(PIPELINE) - 1:
                a = QLabel()
                a.setFixedSize(12, 16)
                self._arrows.append(a)
                lay.addWidget(a)
        lay.addStretch(1)
        self._paint_arrows()
        self._unsub = bus.subscribe(self._on_event)
        self.destroyed.connect(lambda *_: self._unsub())
        for stage, ev in bus.snapshot().items():
            if stage in self.chips:
                self.chips[stage].apply(ev.state, ev.message, bus.elapsed_of(stage))
        theme.on_changed(self, self._retheme)
        self.setAccessibleName("Processing pipeline")

    def _paint_arrows(self) -> None:
        pm = icons.pixmap("chevron_right", theme.current()["text_faint"], 12)
        for a in self._arrows:
            a.setPixmap(pm)

    def _retheme(self) -> None:
        self._paint_arrows()
        for stage, chip in self.chips.items():
            ev = self.bus.snapshot().get(stage)
            chip.apply(ev.state if ev else IDLE, ev.message if ev else "", self.bus.elapsed_of(stage))

    def _on_event(self, ev) -> None:
        assert_gui_thread("PipelineStrip")
        chip = self.chips.get(ev.stage)
        if chip is not None:
            chip.apply(ev.state, ev.message, self.bus.elapsed_of(ev.stage) if ev.state != RUNNING else None)

    def set_compact(self, on: bool) -> None:
        self.compact = on
        for c in self.chips.values():
            c.name.setVisible(not on)
            c.time.setVisible(not on)

    def states(self) -> dict[str, str]:
        return {s: c.state for s, c in self.chips.items()}

    def closeEvent(self, e) -> None:
        self._unsub()
        super().closeEvent(e)
