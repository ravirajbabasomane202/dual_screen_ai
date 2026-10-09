"""Settings with dotted keys, persisted in SQLite. Privacy options default to OFF."""
from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any, Callable

from app.gui_thread import call_on_gui

DEFAULTS: dict[str, dict[str, Any]] = {
    "app": {"first_run_done": False},
    "audio": {
        "system_enabled": True, "system_device": "",       # "" = Windows default output
        "mic_enabled": False, "mic_device": "",
        "volume": 1.0, "vad_sensitivity": 0.5,
    },
    "speech": {"engine": "faster-whisper", "model": "auto", "language": "en", "device": "auto"},
    "ai": {
        "provider": "ollama", "host": "http://localhost:11434", "model": "auto",
        "temperature": 0.2, "answer_length": "Short", "context": True, "timeout": 60,
        "max_tokens": 0,               # 0 = automatic (answer length + JSON overhead + the echoed question)
        "context_window": 8192,        # tokens the prompt may use (sent to Ollama as num_ctx)
        "strict_validation": True,     # reject answers that restate/summarise the question and regenerate
        "max_retries": 2,
        "style_preset": "Concise",     # Concise | STAR | Technical deep-dive | Spoken-natural
    },
    "question": {
        "auto_threshold": 0.85, "confirm_threshold": 0.60, "duplicate_timeout": 30,
        "auto_answer": True, "embedding_model": "",
        "min_confidence": 0.30,        # below this the transcript is dropped without calling the LLM
    },
    "retrieval": {
        "top_k": 4,                    # knowledge-base chunks sent to the LLM
        "resume_top_k": 3,             # resume entries sent to the LLM
        "min_score": 0.25,             # 0..1 relevance floor for a retrieved chunk
        "use_resume": True, "use_knowledge_base": True,
    },
    "knowledge": {"folder": "", "chunk_tokens": 220, "chunk_overlap": 40,
                  "auto_sync": True,       # re-check the folder when the app starts
                  "extra_folders": [],     # further folders added with "Add folder…"
                  "watch": False,          # keep watching the folders while the app runs
                  "watch_interval": 30},
    "resume": {"path": ""},            # last resume file chosen (the parsed profile itself lives in the database)
    "context": {"short_term_turns": 4, "persistent_facts": 12, "answer_chars": 600},
    "debug": {"enabled": False, "log_level": "INFO", "show_trace": True},
    "display": {
        "meeting_monitor": 0, "ai_monitor": 1, "window_mode": "windowed",
        "font_size": 18, "theme": "dark", "always_on_top": True,
        "accent": "",                  # "" = the theme's own accent colour; otherwise "#rrggbb"
        "answer_font_size": 22,        # the answer text is the most important thing on screen
        "reduced_motion": False,
        "mini_mode": False, "mini_opacity": 0.92, "click_through": False,
        "ready_cue": "visual",         # none | visual | sound
        "show_transcript": False, "show_history": True,
        "language": "en",
    },
    "profile": {"jd_chars": 1200},      # the interview target itself (company / role / JD) lives in its own DB table
    "privacy": {"store_transcript": False, "store_questions": False, "store_audio": False},
    "shortcuts": {
        "start_pause": "Ctrl+Shift+Space", "process": "Ctrl+Shift+A", "clear_answer": "Ctrl+Shift+C",
        "clear_context": "Ctrl+Shift+X", "move_window": "Ctrl+Shift+M", "toggle_lock": "Ctrl+Shift+U",
    },
}


class SettingsManager:
    """Thread-safe settings store.

    `on_change` callbacks are UI code (they restyle windows, move them between monitors, ...), so they are ALWAYS
    delivered on the GUI thread, however and from wherever `set()` was called (a worker thread that finishes a resume
    upload, for example).  Delivery is synchronous when the caller already is on the GUI thread or when no
    QApplication exists (headless / tests).  Callbacks registered with `thread="inline"` run on the calling thread and
    must not touch widgets.
    """

    def __init__(self, db):
        self.db = db
        self._data = deepcopy(DEFAULTS)
        self._lock = threading.RLock()
        self._callbacks: list[tuple[Callable[[set[str]], None], str]] = []
        for key, value in db.all_kv().items():
            self._assign(key, value, only_known=True)

    def _assign(self, key: str, value, only_known=False) -> bool:
        section, _, name = key.partition(".")
        with self._lock:
            if section not in self._data or (only_known and name not in self._data[section]):
                return False
            self._data[section][name] = value
            return True

    def get(self, key: str, default=None):
        section, _, name = key.partition(".")
        with self._lock:
            return self._data.get(section, {}).get(name, default)

    def set_many(self, mapping: dict[str, Any], save: bool = True) -> None:
        changed = set()
        with self._lock:
            for key, value in mapping.items():
                if self.get(key, object()) != value and self._assign(key, value):
                    changed.add(key)
                    if save:
                        self.db.set_kv(key, value)
            callbacks = list(self._callbacks)
        if not changed:
            return
        for cb, thread in callbacks:
            if thread == "inline":
                self._safe(cb, changed)
            else:
                call_on_gui(self._safe, cb, changed)

    @staticmethod
    def _safe(cb, changed) -> None:
        try:
            cb(changed)
        except Exception:
            import logging
            logging.getLogger("app.settings").exception("settings listener failed")
            if _reraise_in_tests[0]:
                raise

    def set(self, key: str, value, save: bool = True) -> None:
        self.set_many({key: value}, save)

    def on_change(self, cb: Callable[[set[str]], None], thread: str = "gui") -> None:
        self._callbacks.append((cb, thread))

    def section(self, name: str) -> dict:
        with self._lock:
            return dict(self._data[name])

    def reset_section(self, name: str) -> None:
        """Restore one section to its defaults (used by the settings dialog's Reset button)."""
        self.set_many({f"{name}.{k}": deepcopy(v) for k, v in DEFAULTS[name].items()})

    def reset_keys(self, keys) -> None:
        self.set_many({k: deepcopy(DEFAULTS[k.partition('.')[0]][k.partition('.')[2]]) for k in keys
                       if k.partition('.')[0] in DEFAULTS and k.partition('.')[2] in DEFAULTS[k.partition('.')[0]]})


_reraise_in_tests = [False]        # tests flip this so a failing listener fails the test instead of only being logged
