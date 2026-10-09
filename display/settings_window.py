"""Settings: left icon sidebar + search + pages + sticky Save / Reset footer. Inline validation, help text, a 'restart
needed' indicator and live-apply for the look (theme / accent / sizes). UI only: values go through SettingsManager.

Pages that need actions (upload, index, test, diagnostics, privacy) are in display/settings_pages.py and display/pages/."""
from __future__ import annotations

from urllib.parse import urlparse

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFrame, QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from display import icons, theme
from display.i18n import tr
from display.monitor_manager import MonitorManager
from display.widgets.common import SearchBox, badge, button, hbox, label, set_prop

# page -> rows: (settings key, label, kind, extra)
SCHEMA = {
    "General": [
        ("ai.style_preset", "Answer style", "choice", ["Concise", "STAR", "Technical deep-dive", "Spoken-natural"]),
        ("question.auto_threshold", "Auto-process threshold", "float", (0.5, 1.0, 0.01)),
        ("question.confirm_threshold", "Ask-for-confirmation threshold", "float", (0.3, 0.95, 0.01)),
        ("question.duplicate_timeout", "Duplicate timeout (seconds)", "int", (0, 600)),
        ("question.auto_answer", "Answer automatically", "bool", None),
        ("question.min_confidence", "Minimum question confidence", "float", (0.0, 0.9, 0.05)),
        ("question.embedding_model", "Embedding model (optional, e.g. nomic-embed-text)", "str", None),
    ],
    "Audio": [
        ("audio.system_enabled", "Capture system audio (WASAPI loopback)", "bool", None),
        ("audio.system_device", "System audio device", "system_device", None),
        ("audio.mic_enabled", "Also capture microphone", "bool", None),
        ("audio.mic_device", "Microphone", "mic_device", None),
        ("audio.volume", "Audio volume (gain)", "float", (0.1, 4.0, 0.1)),
        ("audio.vad_sensitivity", "VAD sensitivity", "float", (0.0, 1.0, 0.05)),
    ],
    "Speech": [
        ("speech.engine", "Engine", "choice", ["faster-whisper", "whisper.cpp"]),
        ("speech.model", "Whisper model", "choice", ["auto", "tiny", "base", "small", "medium", "large-v3"]),
        ("speech.language", "Language", "editable", ["en", "auto", "hi", "es", "fr", "de", "pt", "zh", "ja", "ar"]),
        ("speech.device", "Compute device", "choice", ["auto", "cuda", "cpu"]),
    ],
    "LLM": [
        ("ai.provider", "Provider (local LLM runtime)", "choice", ["ollama", "llama.cpp"]),
        ("ai.host", "Endpoint (localhost only)", "str", None),
        ("ai.model", "Model", "llm_model", None),
        ("ai.temperature", "Temperature", "float", (0.0, 1.5, 0.05)),
        ("ai.max_tokens", "Max tokens (0 = automatic)", "int", (0, 8192)),
        ("ai.context_window", "Context window (tokens)", "int", (1024, 131072)),
        ("ai.timeout", "Timeout (seconds)", "int", (5, 600)),
        ("ai.answer_length", "Answer length", "choice", ["Very Short", "Short", "Medium", "Detailed"]),
        ("ai.strict_validation", "Reject answers that describe the question and regenerate", "bool", None),
    ],
    "Resume": [
        ("retrieval.use_resume", "Use my resume when answering", "bool", None),
        ("retrieval.resume_top_k", "Resume entries sent to the LLM", "int", (1, 8)),
    ],
    "Knowledge Base": [
        ("retrieval.use_knowledge_base", "Use the knowledge base when answering", "bool", None),
        ("retrieval.top_k", "Retrieved chunks sent to the LLM", "int", (1, 12)),
        ("retrieval.min_score", "Minimum relevance (0-1)", "float", (0.0, 1.0, 0.05)),
        ("knowledge.chunk_tokens", "Chunk size (tokens; applies on re-index)", "int", (80, 800)),
    ],
    "Context & Memory": [
        ("ai.context", "Use conversation context", "bool", None),
        ("context.short_term_turns", "Recent turns remembered", "int", (1, 12)),
        ("context.persistent_facts", "Important facts remembered", "int", (0, 40)),
        ("context.answer_chars", "Remembered answer length (characters)", "int", (100, 3000)),
        ("profile.jd_chars", "Job-description characters sent to the model", "int", (0, 6000)),
    ],
    "Display & Theme": [
        ("display.ai_monitor", "AI monitor", "monitor", None),
        ("display.window_mode", "Window mode", "choice", ["windowed", "borderless", "fullscreen"]),
        ("display.theme", "Theme", "choice", ["dark", "light", "contrast"]),
        ("display.accent", "Accent colour (#rrggbb, empty = theme default)", "color", None),
        ("display.font_size", "Interface text size", "int", (10, 48)),
        ("display.answer_font_size", "Answer text size", "int", (12, 64)),
        ("display.always_on_top", "Always on top", "bool", None),
        ("display.mini_mode", "Mini mode (answer only)", "bool", None),
        ("display.mini_opacity", "Mini-mode opacity", "float", (0.3, 1.0, 0.05)),
        ("display.click_through", "Click-through lock (mouse passes through; Ctrl+Shift+U unlocks)", "bool", None),
        ("display.reduced_motion", "Reduce motion", "bool", None),
        ("display.ready_cue", "Answer-ready cue", "choice", ["none", "visual", "sound"]),
        ("display.show_transcript", "Show the transcript panel", "bool", None),
        ("display.show_history", "Show the history panel", "bool", None),
    ],
    "Shortcuts": [
        ("shortcuts.start_pause", "Start / Pause", "str", None),
        ("shortcuts.process", "Process current question", "str", None),
        ("shortcuts.clear_answer", "Clear answer", "str", None),
        ("shortcuts.clear_context", "Clear context", "str", None),
        ("shortcuts.move_window", "Move AI window to configured monitor", "str", None),
        ("shortcuts.toggle_lock", "Toggle click-through lock", "str", None),
    ],
    "Privacy": [
        ("privacy.store_transcript", "Store transcript", "bool", None),
        ("privacy.store_questions", "Store questions & answers", "bool", None),
        ("privacy.store_audio", "Store audio", "bool", None),
    ],
    "Debug & Logs": [
        ("debug.enabled", "Debug mode (verbose logs incl. question/answer text, trace panel)", "bool", None),
        ("debug.log_level", "Log level", "choice", ["DEBUG", "INFO", "WARNING", "ERROR"]),
        ("debug.show_trace", "Show the answer trace panel in debug mode", "bool", None),
    ],
    "Diagnostics": [],
    "About": [],
}
PAGE_ICONS = {"General": "sliders", "Audio": "speaker", "Speech": "mic", "LLM": "brain", "Resume": "document",
              "Knowledge Base": "database", "Context & Memory": "layers", "Display & Theme": "palette", "Shortcuts": "keyboard",
              "Privacy": "shield", "Debug & Logs": "bug", "Diagnostics": "activity", "About": "info"}
ALIASES = {"Context / Memory": "Context & Memory", "Display": "Display & Theme", "Debug": "Debug & Logs",
           "Question Detection": "General"}                  # old page names keep working (health bar, tests)

HELP = {
    "ai.host": "Only localhost addresses are accepted — nothing leaves this computer.",
    "ai.timeout": "How long to wait for the model before giving up.",
    "ai.context_window": "Larger values remember more but use more memory.",
    "question.auto_threshold": "At or above this confidence a question is answered without asking.",
    "question.confirm_threshold": "Between this and the auto threshold you are asked first.",
    "question.min_confidence": "Below this, sentences are dropped without calling the model.",
    "audio.vad_sensitivity": "Higher picks up quieter speech but may react to noise.",
    "display.click_through": "While on, clicks pass through the window. Use the shortcut to turn it off again.",
    "display.accent": "Example: #6c9cff. The High-contrast theme ignores the accent.",
    "privacy.store_audio": "Audio is never written to disk unless this is on.",
    "debug.enabled": "Question and answer text appears in logs only while this is on.",
    "knowledge.chunk_tokens": "Smaller chunks are more precise; re-index to apply.",
    "ai.style_preset": "Also changeable from the answer card.",
}
RESTART = {"audio.system_enabled", "audio.system_device", "audio.mic_enabled", "audio.mic_device", "speech.engine",
           "speech.model", "speech.language", "speech.device", "ai.provider", "ai.host", "ai.model"}
LIVE_PREFIX = "display."


def validate(values: dict) -> dict[str, str]:
    """key -> problem. Pure function (testable without widgets)."""
    bad = {}
    host = str(values.get("ai.host", "")).strip()
    if "ai.host" in values:
        u = urlparse(host if "://" in host else "http://" + host)
        if not host or u.hostname not in ("localhost", "127.0.0.1", "::1"):
            bad["ai.host"] = "Use a localhost address such as http://localhost:11434."
    if "question.auto_threshold" in values and "question.confirm_threshold" in values \
            and float(values["question.confirm_threshold"]) > float(values["question.auto_threshold"]):
        bad["question.confirm_threshold"] = "Must not be higher than the auto-process threshold."
    acc = str(values.get("display.accent", "")).strip()
    if "display.accent" in values and acc and not theme.valid_hex(acc):
        bad["display.accent"] = "Use a colour like #6c9cff."
    for k in ("shortcuts.start_pause", "shortcuts.process", "shortcuts.clear_answer", "shortcuts.clear_context",
              "shortcuts.move_window", "shortcuts.toggle_lock"):
        if k in values and not str(values[k]).strip():
            bad[k] = "A shortcut cannot be empty."
    return bad


class SettingsWindow(QDialog):
    def __init__(self, settings, parent=None, services=None, db=None, initial_page: str = ""):
        """`services`: the MeetingEngine (gives access to resume, kb, test_llm, run_diagnostics, context)."""
        super().__init__(parent)
        self.s = settings
        self.services = services
        self.db = db or getattr(services, "db", None)
        self.setWindowTitle(tr("Settings"))
        self.setObjectName("window")
        self.setMinimumSize(860, 560)
        self.resize(1020, 700)
        self._widgets: dict[str, tuple[str, QWidget]] = {}
        self._rows: dict[str, tuple[str, QWidget, str, QLabel, QLabel]] = {}
        self._initial = {k: settings.get(k) for rows in SCHEMA.values() for k, *_ in rows}
        self._live_original = {k: v for k, v in self._initial.items() if k.startswith(LIVE_PREFIX)}
        self._pages: dict[str, QWidget] = {}
        self._extras: dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        main = QHBoxLayout()
        main.setSpacing(0)
        side = QWidget(objectName="sidebar")
        side.setFixedWidth(230)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(8, 12, 8, 8)
        self.search = SearchBox(tr("Search settings…"))
        self.search.textChanged.connect(self.apply_search)
        sl.addWidget(self.search)
        self.nav = QListWidget(objectName="nav")
        self.nav.setAccessibleName("Settings sections")
        sl.addWidget(self.nav, 1)
        main.addWidget(side)
        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        root.addLayout(main, 1)

        for name, rows in SCHEMA.items():
            self._add_page(name, rows)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)

        footer = QWidget(objectName="footer")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(16, 10, 16, 10)
        self.restart_note = badge(tr("Engine restarts when you save"), "warning")
        self.restart_note.hide()
        self.problem = label("", "danger")
        fl.addWidget(self.restart_note)
        fl.addWidget(self.problem, 1)
        self.btn_reset = button(tr("Reset page"), "ghost", "undo", tr("Restore this page's defaults"))
        self.btn_cancel = button(tr("Cancel"), "secondary")
        self.btn_save = button(tr("Save"), "primary", "check", shortcut="Ctrl+S")
        self.btn_reset.clicked.connect(lambda: self.reset_page())
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(lambda: self._save())
        for b in (self.btn_reset, self.btn_cancel, self.btn_save):
            fl.addWidget(b)
        root.addWidget(footer)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=lambda: self.search.setFocus())
        self.nav.setCurrentRow(0)
        if initial_page:
            self.show_page(initial_page)
        self._revalidate()

    # ------------------------------------------------------------------ pages
    def _add_page(self, name: str, rows: list) -> None:
        item = QListWidgetItem(icons.icon(PAGE_ICONS.get(name, "gear"), size=18), tr(name))
        item.setData(Qt.UserRole, name)
        self.nav.addItem(item)
        page = QWidget(objectName="page")
        scroll_host = QVBoxLayout(page)
        scroll_host.setContentsMargins(24, 20, 24, 12)
        scroll_host.addWidget(label(tr(name), "title"))
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 8, 8, 8)
        bl.setSpacing(10)
        for key, text, kind, extra in rows:
            w = self._make(kind, extra, self.s.get(key))
            self._widgets[key] = (kind, w)
            self._connect(key, kind, w)
            lab = label(tr(text), "", wrap=True)
            help_l = label(HELP.get(key, ""), "caption", wrap=True)
            help_l.setVisible(bool(HELP.get(key)))
            err = label("", "danger", wrap=True)
            err.hide()
            row = QWidget()
            rl = QVBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(3)
            top = QHBoxLayout()
            if kind == "bool":
                top.addWidget(w)
                top.addWidget(lab, 1)
            else:
                top.addWidget(lab, 1)
                w.setMinimumWidth(240)
                top.addWidget(w)
            if key in RESTART:
                r = badge(tr("restart"), "neutral")
                r.setToolTip(tr("The engine restarts automatically when this changes."))
                top.addWidget(r)
            rl.addLayout(top)
            rl.addWidget(help_l)
            rl.addWidget(err)
            if kind != "bool":
                lab.setBuddy(w)
            w.setAccessibleName(text)
            self._rows[key] = (name, row, f"{text} {key} {HELP.get(key, '')}".lower(), help_l, err)
            bl.addWidget(row)
        extra = self._extra_page(name)
        if extra is not None:
            self._extras[name] = extra
            bl.addWidget(extra, 1)
        bl.addStretch(0 if extra is not None else 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        scroll_host.addWidget(scroll, 1)
        self.stack.addWidget(page)
        self._pages[name] = page

    def _extra_page(self, section: str):
        from applog import get_log_dir
        from display import settings_pages as sp
        e = self.services
        if section == "Resume":
            return sp.ResumePage(e)
        if section == "Knowledge Base":
            return sp.KnowledgePage(e, self.s)
        if section == "LLM":
            return sp.LLMTestPage(e, self.s)
        if section == "General":
            return sp.InterviewProfilePage(e)
        if section == "Context & Memory":
            return sp.MemoryPage(e)
        if section == "Privacy":
            return sp.PrivacyPage(e, self.s, self.db, get_log_dir())
        if section == "Debug & Logs":
            return sp.DebugPage(e, get_log_dir(), settings=self.s)
        if section == "Diagnostics":
            return sp.DebugPage(e, get_log_dir(), settings=self.s)
        if section == "About":
            return sp.AboutPage(get_log_dir())
        return None

    def show_page(self, name: str) -> bool:
        name = ALIASES.get(name, name)
        for i in range(self.nav.count()):
            if self.nav.item(i).data(Qt.UserRole) == name:
                self.search.setText("")
                self.nav.setCurrentRow(i)
                return True
        return False

    def current_page(self) -> str:
        it = self.nav.currentItem()
        return it.data(Qt.UserRole) if it else ""

    # ------------------------------------------------------------------ search
    def apply_search(self, text: str = "") -> list[str]:
        """Filter rows and sidebar entries; returns the matching setting keys."""
        q = (text or "").strip().lower()
        hits, pages = [], set()
        for key, (page, row, hay, _h, _e) in self._rows.items():
            ok = not q or q in hay or q in page.lower()
            row.setVisible(ok)
            if ok:
                pages.add(page)
                if q:
                    hits.append(key)
        for i in range(self.nav.count()):
            name = self.nav.item(i).data(Qt.UserRole)
            self.nav.item(i).setHidden(bool(q) and name not in pages and q not in name.lower())
        if q:
            for i in range(self.nav.count()):
                if not self.nav.item(i).isHidden():
                    self.nav.setCurrentRow(i)
                    break
        return hits

    # ------------------------------------------------------------------ widgets
    def _make(self, kind, extra, value) -> QWidget:
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
        elif kind in ("choice", "editable"):
            w = QComboBox()
            w.addItems(extra)
            w.setEditable(kind == "editable")
            w.setCurrentText(str(value))
        elif kind == "float":
            lo, hi, step = extra
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(2)
            w.setValue(float(value))
        elif kind == "int":
            w = QSpinBox()
            w.setRange(*extra)
            w.setValue(int(value))
        elif kind == "monitor":
            w = QComboBox()
            for m in MonitorManager.list_monitors():
                w.addItem(m.label, m.index)
            w.setCurrentIndex(max(0, w.findData(int(value))))
        elif kind in ("system_device", "mic_device", "llm_model"):
            w = QComboBox()
            w.setEditable(True)
            w.addItem("")
            for name in self._devices(kind):
                w.addItem(name)
            w.setCurrentText(str(value))
            if kind == "llm_model":
                w.addItem("auto")
        elif kind == "color":
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(str(value or ""))
            edit.setPlaceholderText("#6c9cff")
            pick = button(tr("Pick…"), "secondary", "palette")
            pick.clicked.connect(lambda: self._pick_color(edit))
            h.addWidget(edit, 1)
            h.addWidget(pick)
            w.edit = edit
        else:
            w = QLineEdit(str(value))
        return w

    def _pick_color(self, edit: QLineEdit) -> None:
        c = QColorDialog.getColor(QColor(edit.text() or theme.current()["accent"]), self)
        if c.isValid():
            edit.setText(c.name())

    def _connect(self, key: str, kind: str, w: QWidget) -> None:
        sig = {"bool": "toggled", "choice": "currentTextChanged", "editable": "currentTextChanged", "float": "valueChanged",
               "int": "valueChanged", "monitor": "currentIndexChanged", "system_device": "currentTextChanged",
               "mic_device": "currentTextChanged", "llm_model": "currentTextChanged", "str": "textChanged"}.get(kind)
        target = w.edit if kind == "color" else w
        getattr(target, "textChanged" if kind == "color" else sig).connect(lambda *_a, k=key: self._changed(k))

    def _devices(self, kind) -> list[str]:
        try:
            if kind == "system_device":
                from audio.wasapi_capture import WasapiLoopbackCapture
                return [d["name"] for d in WasapiLoopbackCapture.list_devices()]
            if kind == "mic_device":
                from audio.microphone import MicrophoneCapture
                return [d["name"] for d in MicrophoneCapture.list_devices()]
            from llm.model_manager import create_provider
            return create_provider(self.s, "").list_models()
        except Exception:
            return []

    def _value(self, key: str):
        kind, w = self._widgets[key]
        if kind == "bool":
            return w.isChecked()
        if kind in ("choice", "editable", "system_device", "mic_device", "llm_model"):
            return w.currentText().strip()
        if kind == "monitor":
            return int(w.currentData())
        if kind in ("float", "int"):
            return w.value()
        if kind == "color":
            return w.edit.text().strip()
        return w.text().strip()

    def values(self) -> dict:
        return {k: self._value(k) for k in self._widgets}

    # ------------------------------------------------------------------ change handling
    def _changed(self, key: str) -> None:
        assert_gui_thread("SettingsWindow._changed")
        self._revalidate()
        if key.startswith(LIVE_PREFIX) and not self.problem.text():
            # live preview of the look; Cancel restores the original values
            if key in ("display.theme", "display.accent", "display.font_size", "display.answer_font_size",
                       "display.reduced_motion", "display.mini_opacity"):
                self.s.set(key, self._value(key))
                theme.apply_settings(self.s)

    def _revalidate(self) -> None:
        vals = self.values()
        bad = validate(vals)
        for key, (_p, _row, _h, _help, err) in self._rows.items():
            w = self._widgets[key][1]
            target = w.edit if self._widgets[key][0] == "color" else w
            set_prop(target, "invalid", key in bad)
            err.setText(bad.get(key, ""))
            err.setVisible(key in bad)
        self.problem.setText(next(iter(bad.values())) if bad else "")
        self.btn_save.setEnabled(not bad)
        self.restart_note.setVisible(any(k in RESTART and vals[k] != self._initial.get(k) for k in vals))

    def reset_page(self) -> None:
        from settings.settings_manager import DEFAULTS
        page = self.current_page()
        for key, *_ in SCHEMA.get(page, []):
            sec, _, name = key.partition(".")
            kind, w = self._widgets[key]
            dv = DEFAULTS[sec][name]
            if kind == "bool":
                w.setChecked(bool(dv))
            elif kind in ("float", "int"):
                w.setValue(dv)
            elif kind == "monitor":
                w.setCurrentIndex(max(0, w.findData(int(dv))))
            elif kind == "color":
                w.edit.setText(str(dv or ""))
            elif kind == "str":
                w.setText(str(dv))
            else:
                w.setCurrentText(str(dv))
        self._revalidate()

    def _save(self) -> None:
        if validate(self.values()):
            return
        self.s.set_many(self.values())
        self._live_original = {}
        self.accept()

    def reject(self) -> None:
        if self._live_original:                                   # undo the live preview
            self.s.set_many(self._live_original)
            theme.apply_settings(self.s)
        super().reject()
