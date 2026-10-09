"""The Monitor-2 window: header (health, pause, settings), pipeline strip, question card, answer card with grounding badge,
sources and toolbar, quick-ask box, transcript with question feedback, collapsible history and debug panel.

UI only: every action is a signal; the logic lives in app/engine.py and friends. All methods run on the GUI thread."""
from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                               QSizePolicy, QTextBrowser, QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from app.status import get_bus
from display import icons, theme
from display.monitor_manager import MonitorManager  # noqa: F401  (kept importable from here)
from display.widgets.activity_drawer import ActivityDrawer
from display.widgets.common import (ConfidenceRing, FlowLayout, Chip, PulseDot, SearchBox, badge, button, card, divider,
                                    fade_in, hbox, icon_button, label, set_badge)
from display.widgets.feedback import Banner, EmptyState, ToastManager
from display.widgets.health_bar import HealthBar
from display.widgets.pipeline_strip import PipelineStrip

STYLE_PRESETS = ("Concise", "STAR", "Technical deep-dive", "Spoken-natural")

# state key: (label, theme token, pulses)
STATES = {
    "loading": ("Loading models…", "warning", True),
    "listening": ("Listening", "success", True),
    "transcribing": ("Transcribing", "info", True),
    "question_detected": ("Question detected", "accent", False),
    "cleaning": ("Cleaning question", "accent", True),
    "generating": ("Generating answer", "accent", True),
    "ready": ("Answer ready", "success", False),
    "paused": ("Paused", "text_faint", False),
    "error": ("Error", "danger", False),
}
GROUND = {"resume": ("Resume", "info"), "knowledge_base": ("Knowledge base", "success"),
          "conversation": ("Conversation", "neutral"), "general": ("General knowledge", "neutral"),
          "unknown": ("Unverified", "warning")}
_GROUND_TEXT = {"resume": "your resume", "knowledge_base": "knowledge base", "conversation": "conversation",
                "general": "general knowledge", "unknown": "unverified"}


class AIWindow(QWidget):
    pause_toggled = Signal()
    process_requested = Signal()
    dismiss_pending = Signal()
    clear_requested = Signal()
    clear_context_requested = Signal()
    clear_history_requested = Signal()
    settings_requested = Signal()
    move_requested = Signal()
    # new
    ask_requested = Signal(str)                       # quick re-ask / typed question
    practice_requested = Signal()
    regenerate_requested = Signal(str)                # "", "shorter", "longer"
    feedback_requested = Signal(str, bool, str, float, str)   # text, is_question, detected type, confidence, action
    answer_rated = Signal(bool, str, str)             # good?, question, answer
    style_changed = Signal(str)
    page_requested = Signal(str)                      # settings page to open
    test_requested = Signal(str)                      # "llm" ...
    open_logs_requested = Signal()
    export_requested = Signal()
    mini_toggled = Signal(bool)
    click_through_toggled = Signal(bool)
    palette_requested = Signal()
    debug_toggled = Signal(bool)
    why_requested = Signal()
    pinned = Signal(str, str)

    def __init__(self, settings, bus=None, health_fn=None):
        super().__init__()
        self.s = settings
        self.bus = bus or get_bus()
        self._health_fn = health_fn
        self.setObjectName("window")
        self.setWindowTitle("AI Assistant")
        self.setAccessibleName("AI Assistant window")
        self._answer_md = ""
        self._last_done = ("", "")
        self._last_detection: dict = {}
        self._pinned: list[tuple[str, str]] = []
        self._history_data: list[tuple[str, str]] = []
        self._streaming = False
        self._t0 = 0.0
        theme.on_changed(self, self._retheme)
        self._build()
        self.toasts = ToastManager(self)
        self.drawer = ActivityDrawer(self.bus, self)
        self.drawer.open_logs_requested.connect(self.open_logs_requested)
        self.drawer.export_requested.connect(self.export_requested)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.drawer.toggle)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.palette_requested)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=lambda: self.submit_question())
        self.apply_theme()
        self.apply_display_settings()
        self.set_status("loading")

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        S = theme.SPACE
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setMinimumSize(720, 520)

        # ---- header
        self.header = QWidget(objectName="header")
        hl = QHBoxLayout(self.header)
        hl.setContentsMargins(S["lg"], S["sm"], S["lg"], S["sm"])
        hl.setSpacing(S["sm"])
        self.logo = QFrame()
        self.logo.setFixedSize(26, 26)
        self.logo_icon = label("")
        hl.addWidget(self.logo_icon)
        hl.addWidget(label("AI Assistant", "section"))
        self.dot = PulseDot(10)
        self.status = label("", "strong")
        hl.addSpacing(S["md"])
        hl.addWidget(self.dot)
        hl.addWidget(self.status)
        hl.addSpacing(S["md"])
        self.health = HealthBar(self._health_fn or (lambda: []), self.bus)
        self.health.activated.connect(self._on_health)
        hl.addWidget(self.health)
        hl.addStretch(1)
        self.btn_pause = icon_button("pause", "Pause / resume listening", self.pause_toggled.emit, shortcut="Ctrl+Shift+Space")
        self.btn_answer = button("Answer now", "primary", "bolt", "Answer the pending or latest question", "Ctrl+Shift+A",
                                 on_click=lambda: self.process_requested.emit())
        self.btn_clear = icon_button("x", "Clear the answer", self.clear_requested.emit, shortcut="Ctrl+Shift+C")
        self.btn_practice = icon_button("target", "Practice mode — ask yourself a mock question", self.practice_requested.emit)
        self.btn_activity = icon_button("activity", "Activity log", self.drawer_toggle, shortcut="Ctrl+L")
        self.btn_debug = icon_button("bug", "Debug panel", self._debug_clicked, checkable=True)
        self.btn_mini = icon_button("minimize", "Mini mode", self._mini_clicked, checkable=True)
        self.btn_lock = icon_button("cursor", "Click-through lock (mouse passes through the window)", self._lock_clicked, checkable=True)
        self.btn_palette = icon_button("command", "Command palette", self.palette_requested.emit, shortcut="Ctrl+K")
        self.btn_settings = icon_button("gear", "Settings", self.settings_requested.emit)
        for b in (self.btn_pause, self.btn_answer, self.btn_clear, self.btn_practice, self.btn_activity, self.btn_debug,
                  self.btn_mini, self.btn_lock, self.btn_palette, self.btn_settings):
            hl.addWidget(b)
        root.addWidget(self.header)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(S["lg"], S["md"], S["lg"], S["md"])
        bl.setSpacing(S["md"])
        root.addWidget(body, 1)
        self.body = body

        # ---- banners
        self.banner_move = button("Move AI to configured monitor", "secondary", "external", on_click=lambda: self.move_requested.emit())
        self.banner_move.hide()
        self.banner = Banner()
        self.banner.action_clicked.connect(self._banner_action)
        self.recovery = Banner()
        self.recovery.action_clicked.connect(lambda _k: self.export_requested.emit())
        bl.addWidget(self.banner_move)
        bl.addWidget(self.banner)
        bl.addWidget(self.recovery)

        # ---- pipeline
        self.pipeline = PipelineStrip(self.bus)
        self.pipeline.stage_clicked.connect(self._on_stage)
        bl.addWidget(self.pipeline)

        # ---- pending confirm
        self.pending = QFrame(objectName="card")
        self.pending.setProperty("elevated", True)
        pl = QHBoxLayout(self.pending)
        self.pending_label = label("", "", wrap=True)
        pl.addWidget(self.pending_label, 1)
        pl.addWidget(button("Answer", "primary", on_click=lambda: self.process_requested.emit()))
        dis = button("Dismiss", "ghost")
        dis.clicked.connect(self.dismiss_pending)
        dis.clicked.connect(self.pending.hide)
        pl.addWidget(dis)
        self.pending.hide()
        bl.addWidget(self.pending)

        # ---- question card
        self.qcard, ql = card()
        top = QHBoxLayout()
        top.addWidget(label("DETECTED QUESTION", "overline"))
        self.q_badge = badge("—", "neutral")
        top.addWidget(self.q_badge)
        top.addStretch(1)
        self.ring = ConfidenceRing(40)
        self.ring.setToolTip("Question confidence")
        top.addWidget(self.ring)
        ql.addLayout(top)
        self.question = label("Waiting for a question…", "section", wrap=True, selectable=True)
        ql.addWidget(self.question)
        self.live = label("", "caption", wrap=True)
        ql.addWidget(self.live)
        bl.addWidget(self.qcard)

        # ---- answer card
        self.acard, al = card(elevated=True)
        ah = QHBoxLayout()
        ah.addWidget(label("ANSWER", "overline"))
        self.g_badge = badge("", "neutral")
        self.g_badge.hide()
        ah.addWidget(self.g_badge)
        self.gen_dot = PulseDot(8)
        self.gen_dot.hide()
        ah.addWidget(self.gen_dot)
        ah.addStretch(1)
        self.style_combo = QComboBox()
        self.style_combo.addItems(STYLE_PRESETS)
        self.style_combo.setToolTip("Answer style")
        self.style_combo.setAccessibleName("Answer style")
        self.style_combo.activated.connect(lambda _i: self.style_changed.emit(self.style_combo.currentText()))
        ah.addWidget(self.style_combo)
        al.addLayout(ah)
        self.stack = QWidget()
        sl = QVBoxLayout(self.stack)
        sl.setContentsMargins(0, 0, 0, 0)
        self.answer = QTextBrowser(objectName="answer")
        self.answer.setOpenExternalLinks(False)
        self.answer.setAccessibleName("Answer")
        self.empty = EmptyState("sparkle", "Ready when you are",
                                "The answer appears here as soon as a question is heard. You can also type a question below.")
        sl.addWidget(self.empty)
        sl.addWidget(self.answer)
        self.answer.hide()
        al.addWidget(self.stack, 1)
        self.sources_host = QWidget()
        self.sources = FlowLayout(self.sources_host)
        self.sources_host.hide()
        al.addWidget(self.sources_host)
        tb = QHBoxLayout()
        tb.setSpacing(2)
        self.tool = {
            "copy": icon_button("copy", "Copy answer", self.copy_answer),
            "shorter": icon_button("minus", "Shorter", lambda: self.regenerate_requested.emit("shorter")),
            "longer": icon_button("plus", "Longer", lambda: self.regenerate_requested.emit("longer")),
            "regen": icon_button("refresh", "Regenerate", lambda: self.regenerate_requested.emit("")),
            "pin": icon_button("pin", "Pin this answer", self.pin_answer),
            "why": icon_button("help", "Why this answer?", self.why_requested.emit),
            "up": icon_button("thumbs_up", "Good answer", lambda: self.rate(True)),
            "down": icon_button("thumbs_down", "Bad answer", lambda: self.rate(False)),
        }
        for b in self.tool.values():
            tb.addWidget(b)
        tb.addStretch(1)
        self.footer = label("Confidence: —    Response: —", "caption")
        tb.addWidget(self.footer)
        al.addLayout(tb)
        bl.addWidget(self.acard, 1)

        # ---- quick ask
        self.ask_box = QLineEdit()
        self.ask_box.setPlaceholderText("Missed something? Type a question and press Enter…")
        self.ask_box.setAccessibleName("Ask a question")
        self.ask_box.returnPressed.connect(self.submit_question)
        bl.addLayout(hbox(self.ask_box, button("Ask", "secondary", "message", on_click=lambda: self.submit_question())))

        # ---- transcript / debug / history
        self.transcript_toggle = button("Transcript", "ghost", "chevron_right", checkable=True)
        self.transcript_toggle.toggled.connect(self._toggle_transcript)
        self.transcript = QListWidget()
        self.transcript.setMaximumHeight(130)
        self.transcript.setAccessibleName("Recent transcript")
        self.transcript.hide()
        self.history_toggle = button("History", "ghost", "chevron_right", checkable=True)
        self.history_toggle.toggled.connect(self._toggle_history)
        self.history_search = SearchBox("Search history…")
        self.history_search.textChanged.connect(lambda _t: self._filter_history())
        self.history = QListWidget(objectName="history")
        self.history.setMaximumHeight(110)
        self.history.itemClicked.connect(self._show_history_item)
        self.history_box = QWidget()
        hb = QVBoxLayout(self.history_box)
        hb.setContentsMargins(0, 0, 0, 0)
        hr = QHBoxLayout()
        hr.addWidget(self.history_search, 1)
        hr.addWidget(button("Clear context", "ghost", on_click=lambda: self.clear_context_requested.emit()))
        hr.addWidget(button("Clear history", "ghost", on_click=lambda: self.clear_history_requested.emit()))
        hb.addLayout(hr)
        hb.addWidget(self.history)
        self.history_box.hide()
        self.debug = QTextBrowser(objectName="debug")
        self.debug.setProperty("mono", True)
        self.debug.setMaximumHeight(220)
        self.debug.setLineWrapMode(QTextBrowser.NoWrap)
        self.debug.setAccessibleName("Debug trace")
        self.debug.setVisible(False)
        bl.addLayout(hbox(self.transcript_toggle, self.history_toggle, stretch_end=True))
        bl.addWidget(self.transcript)
        bl.addWidget(self.history_box)
        bl.addWidget(self.debug)

    # ------------------------------------------------------------------ theming
    def _retheme(self) -> None:
        self.logo_icon.setPixmap(icons.pixmap("logo", theme.current()["accent"], 24))
        self._style_answer()
        self.set_status(self._state)

    def _style_answer(self) -> None:
        t = theme.current()
        px = int(self.s.get("display.answer_font_size", 22))
        self.answer.setStyleSheet(f"QTextBrowser#answer {{ font-size:{px}px; }}")
        self.answer.document().setDefaultStyleSheet(
            f"a {{ color:{t['accent']}; }} code, pre {{ color:{t['text']}; background-color:{t['code_bg']}; }}")
        if self._answer_md:
            self.answer.setMarkdown(self._answer_md)

    def apply_theme(self) -> None:
        """Apply display.theme / accent / font size. Called ONLY when a display.* setting changed."""
        assert_gui_thread("AIWindow.apply_theme")
        theme.apply_settings(self.s)
        self._retheme()

    def apply_display_settings(self) -> None:
        assert_gui_thread("AIWindow.apply_display_settings")
        self.setWindowOpacity(float(self.s.get("display.mini_opacity", 0.92)) if self.s.get("display.mini_mode") else 1.0)
        self.set_mini(bool(self.s.get("display.mini_mode")), emit=False)
        self.transcript_toggle.setChecked(bool(self.s.get("display.show_transcript")))
        self.history_toggle.setChecked(bool(self.s.get("display.show_history", True)))
        self.style_combo.setCurrentText(str(self.s.get("ai.style_preset", "Concise")))
        self.btn_lock.setChecked(bool(self.s.get("display.click_through")))
        self._style_answer()

    # ------------------------------------------------------------------ header actions
    _state = "loading"

    def set_status(self, state: str) -> None:
        assert_gui_thread("AIWindow.set_status")
        lab, token, pulses = STATES.get(state, STATES["listening"])
        self._state = state if state in STATES else "listening"
        self.status.setText(lab)
        self.dot.set_state(theme.current()[token], pulses)
        self.status.setAccessibleDescription(lab)
        self.btn_pause.setChecked(state == "paused")

    def _on_health(self, action: str) -> None:
        kind, _, arg = action.partition(":")
        (self.page_requested if kind == "page" else self.test_requested).emit(arg)

    def _on_stage(self, stage: str) -> None:
        self.drawer.set_stage_filter(stage)
        self.drawer.open_drawer()

    def drawer_toggle(self) -> None:
        self.drawer.toggle()

    def _debug_clicked(self, on: bool) -> None:
        self.set_debug_visible(on)
        self.debug_toggled.emit(on)

    def _mini_clicked(self, on: bool) -> None:
        self.set_mini(on)

    def _lock_clicked(self, on: bool) -> None:
        self.click_through_toggled.emit(on)

    def set_mini(self, on: bool, emit: bool = True) -> None:
        """Mini mode: only the answer card, always on top, adjustable opacity. Full mode shows everything."""
        for w in (self.pipeline, self.qcard, self.transcript_toggle, self.history_toggle, self.ask_box, self.health):
            w.setVisible(not on)
        if on:
            self.transcript.hide()
            self.history_box.hide()
        else:
            self.transcript.setVisible(self.transcript_toggle.isChecked())
            self.history_box.setVisible(self.history_toggle.isChecked())
        self.setMinimumSize(420, 220) if on else self.setMinimumSize(720, 520)
        self.setWindowOpacity(float(self.s.get("display.mini_opacity", 0.92)) if on else 1.0)
        self.btn_mini.blockSignals(True)
        self.btn_mini.setChecked(on)
        self.btn_mini.blockSignals(False)
        if emit:
            self.mini_toggled.emit(on)

    def set_click_through(self, on: bool) -> None:
        self.setWindowFlag(Qt.WindowTransparentForInput, bool(on))
        if self.isVisible():
            self.show()
        self.btn_lock.blockSignals(True)
        self.btn_lock.setChecked(bool(on))
        self.btn_lock.blockSignals(False)

    def _toggle_transcript(self, on: bool) -> None:
        self.transcript.setVisible(on and not self.btn_mini.isChecked())
        self.transcript_toggle.setIcon(icons.icon("chevron_down" if on else "chevron_right", size=16))

    def _toggle_history(self, on: bool) -> None:
        self.history_box.setVisible(on and not self.btn_mini.isChecked())
        self.history_toggle.setIcon(icons.icon("chevron_down" if on else "chevron_right", size=16))

    # ------------------------------------------------------------------ pipeline events
    def set_live_partial(self, text: str) -> None:
        self.live.setText(f"hearing: {text}" if text else "")

    def add_transcript(self, text: str, qtype: str = "", conf: float = 0.0, action: str = "") -> None:
        """One recent transcript line with 'Question' / 'Not a question' feedback buttons."""
        assert_gui_thread("AIWindow.add_transcript")
        item = QListWidgetItem()
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(4, 2, 4, 2)
        t = label(text, "", wrap=True)
        tip = f"{qtype or 'unclassified'} · {conf:.0%} · {action}" if qtype else ""
        t.setToolTip(tip)
        rl.addWidget(t, 1)
        yes = icon_button("check", "This was a question", None, size=14)
        no = icon_button("x", "Not a question", None, size=14)
        yes.clicked.connect(lambda: self._feedback(item, text, True, qtype, conf, action))
        no.clicked.connect(lambda: self._feedback(item, text, False, qtype, conf, action))
        rl.addWidget(yes)
        rl.addWidget(no)
        item.setSizeHint(row.sizeHint())
        self.transcript.addItem(item)
        self.transcript.setItemWidget(item, row)
        while self.transcript.count() > 30:
            self.transcript.takeItem(0)
        self.transcript.scrollToBottom()

    def _feedback(self, item, text, is_q, qtype, conf, action) -> None:
        w = self.transcript.itemWidget(item)
        if w:
            for b in w.findChildren(QFrame):
                pass
            for b in w.findChildren(type(self.tool["copy"])):
                b.setEnabled(False)
        self.feedback_requested.emit(text, is_q, qtype, conf, action)
        self.toast("success", "Thanks — marked as a question." if is_q else "Thanks — I'll ignore sentences like this.")

    def show_pending(self, text: str, conf: float) -> None:
        self.pending_label.setText(f"Possible question ({conf:.0%}): “{text}”   — press Answer or Ctrl+Shift+A")
        self.pending.show()

    def begin_question(self, raw: str, conf: float = 0.0, qtype: str = "") -> None:
        assert_gui_thread("AIWindow.begin_question")
        self.pending.hide()
        self.live.setText("")
        self._answer_md = ""
        self.answer.clear()
        self.question.setText(raw)
        self.g_badge.hide()
        self._clear_sources()
        self._show_answer(True)
        self._streaming, self._t0 = True, time.monotonic()
        self.gen_dot.show()
        self.gen_dot.set_state(theme.current()["accent"], True)
        if conf:
            self.ring.set_value(conf)
        if qtype:
            set_badge(self.q_badge, qtype.replace("_", " ").title(), "info")

    def set_question_info(self, qtype: str, conf: float) -> None:
        self.ring.set_value(conf)
        set_badge(self.q_badge, qtype.replace("_", " ").title() if qtype else "—", "info" if qtype else "neutral")

    def set_clean_question(self, text: str) -> None:
        self.question.setText(text)

    def _show_answer(self, on: bool) -> None:
        self.answer.setVisible(on)
        self.empty.setVisible(not on)

    def append_answer(self, delta: str) -> None:
        self._show_answer(True)
        self._answer_md += delta
        self.answer.setMarkdown(self._answer_md)
        self.answer.verticalScrollBar().setValue(self.answer.verticalScrollBar().maximum())

    def reset_answer(self) -> None:
        """A regenerated answer replaces one that was already streaming."""
        self._answer_md = ""
        self.answer.clear()

    def show_debug(self, text: str) -> None:
        self.debug.setPlainText(text)

    def set_debug_visible(self, on: bool) -> None:
        self.debug.setVisible(bool(on))
        self.btn_debug.blockSignals(True)
        self.btn_debug.setChecked(bool(on))
        self.btn_debug.blockSignals(False)

    def _clear_sources(self) -> None:
        self.sources.clear()
        self.sources_host.hide()

    def finish_answer(self, question: str, answer: str, conf: float, seconds: float, needs_clarification: bool,
                      grounding: str = "", sources=None, source_refs=None, qtype: str = "") -> None:
        assert_gui_thread("AIWindow.finish_answer")
        self._answer_md = answer + ("\n\n*The question looks ambiguous — consider asking for clarification.*"
                                    if needs_clarification else "")
        self._show_answer(True)
        self.answer.setMarkdown(self._answer_md)
        self.question.setText(question)
        self._streaming = False
        self.gen_dot.set_state(theme.current()["accent"], False)
        self.gen_dot.hide()
        self.ring.set_value(conf)
        extra = f"    Based on: {_GROUND_TEXT.get(grounding, grounding)}" if grounding else ""
        if sources:
            extra += f" ({'; '.join(sources[:3])})"
        self.footer.setText(f"Confidence: {conf:.0%}    Response: {seconds:.1f} sec{extra}")
        if grounding:
            txt, tone = GROUND.get(grounding, (grounding, "neutral"))
            set_badge(self.g_badge, txt, tone)
            self.g_badge.show()
        self._clear_sources()
        for s in (sources or [])[:8]:
            c = Chip(s, "document", payload=s)
            c.clicked.connect(self._source_clicked)
            self.sources.addWidget(c)
        self.sources_host.setVisible(bool(sources))
        if qtype:
            set_badge(self.q_badge, qtype.replace("_", " ").title(), "info")
        self._last_done = (question, answer)
        self.ready_cue()

    def _source_clicked(self, payload) -> None:
        self.toast("info", f"Source: {payload}")
        self.page_requested.emit("Knowledge Base")

    def ready_cue(self) -> None:
        cue = str(self.s.get("display.ready_cue", "visual"))
        if cue == "sound":
            QApplication.beep()
        elif cue == "visual" and not theme.reduced_motion():
            fade_in(self.acard)

    def restore_previous(self, clear: bool) -> None:
        q, a = ("", "") if clear else self._last_done
        self.question.setText(q or "Waiting for a question…")
        self._answer_md = a
        self._streaming = False
        self.gen_dot.hide()
        if a:
            self.answer.setMarkdown(a)
        else:
            self.answer.clear()
            self._show_answer(False)
            self.g_badge.hide()
            self._clear_sources()

    # ------------------------------------------------------------------ answer toolbar
    def copy_answer(self) -> None:
        from display.widgets.feedback import copy_text
        copy_text(self._last_done[1] or self._answer_md)
        self.toast("success", "Answer copied.")

    def pin_answer(self) -> None:
        if self._last_done[1]:
            self._pinned.append(self._last_done)
            self.pinned.emit(*self._last_done)
            self.toast("success", f"Pinned ({len(self._pinned)}).")

    def rate(self, good: bool) -> None:
        q, a = self._last_done
        if a:
            self.answer_rated.emit(good, q, a)
            self.toast("success", "Thanks for the feedback.")

    def submit_question(self) -> None:
        text = self.ask_box.text().strip()
        if text:
            self.ask_box.clear()
            self.ask_requested.emit(text)

    # ------------------------------------------------------------------ history
    def add_history(self, q: str, a: str) -> None:
        self._history_data.append((q, a))
        self.history.addItem(f"{len(self._history_data)}. {q}")
        self.history.scrollToBottom()
        self._filter_history()

    def clear_history(self) -> None:
        self._history_data.clear()
        self.history.clear()

    def _filter_history(self) -> None:
        q = self.history_search.text().strip().lower()
        for i in range(self.history.count()):
            qa = self._history_data[i] if i < len(self._history_data) else ("", "")
            self.history.item(i).setHidden(bool(q) and q not in (qa[0] + " " + qa[1]).lower())

    def _show_history_item(self, item) -> None:
        q, a = self._history_data[self.history.row(item)]
        self.question.setText(q)
        self._show_answer(True)
        self.answer.setMarkdown(a)

    # ------------------------------------------------------------------ feedback surfaces
    def toast(self, kind: str, message: str, action_text: str = "", on_action=None) -> None:
        self.toasts.show(kind, message, action_text, on_action)

    def show_error(self, message: str, details: str = "", action: str = "") -> None:
        """Non-blocking: inline banner (cause, action, details) + toast. The stack trace is already in errors.log."""
        assert_gui_thread("AIWindow.show_error")
        from app.errorinfo import describe
        info = describe(message if not details else RuntimeError(message))
        self.banner.set_action(f"Open {info.page}" if info.page else "", info.page)
        self.banner.show_message("error", info.cause if info.cause else message, action or info.action, details or info.details,
                                 timeout_ms=0)
        self.toast("error", (info.cause or message).splitlines()[0][:140])

    def _banner_action(self, page: str) -> None:
        if page:
            self.page_requested.emit(page)

    def show_recovery(self, message: str) -> None:
        self.recovery.set_action("Export diagnostics", "diag")
        self.recovery.show_message("warning", message, "A crash report was saved to the logs folder.")

    def set_off_monitor(self, off: bool) -> None:
        self.banner_move.setVisible(off)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if hasattr(self, "toasts"):
            self.toasts.relayout()
            self.drawer.relayout()
