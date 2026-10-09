"""StatusBus — one thread-safe place where every component says what it is doing.

Publishers (engine, resume/KB managers, background tasks) call `bus.publish(stage, state, message, ...)` from any
thread. Subscribers (pipeline strip, status bar, activity drawer, health indicators) are called on the GUI thread, in
publication order. Nothing here imports Qt widgets; delivery uses `app.gui_thread.call_on_gui`.

    stage    pipeline stage or subsystem ("stt", "llm", "resume", "kb", ...)
    state    idle | running | ok | warn | error
    progress 0..1, or None for "indeterminate"
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from app.gui_thread import call_on_gui

PIPELINE = ("audio", "vad", "stt", "detect", "context", "retrieve", "llm", "validate", "display")
STAGE_LABELS = {"audio": "Audio", "vad": "VAD", "stt": "STT", "detect": "Detect", "context": "Context",
                "retrieve": "Retrieve", "llm": "LLM", "validate": "Validate", "display": "Display"}
IDLE, RUNNING, OK, WARN, ERROR = "idle", "running", "ok", "warn", "error"
STATES = (IDLE, RUNNING, OK, WARN, ERROR)
LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


@dataclass
class StatusEvent:
    seq: int
    ts: float
    stage: str
    state: str
    message: str = ""
    detail: str = ""
    progress: float | None = None
    elapsed: float | None = None
    level: str = "INFO"
    feed: bool = True                   # shown in the activity feed (False: only updates the stage chip)

    def line(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.ts))
        el = f" ({self.elapsed:.2f}s)" if self.elapsed is not None else ""
        return f"{t} {self.level:<7} [{self.stage}] {self.message}{el}"


class StageTimer:
    """`with bus.track("resume", "Parsing resume…") as t:`  ->  running / ok (with elapsed) / error automatically."""

    def __init__(self, bus: "StatusBus", stage: str, message: str, detail: str = ""):
        self.bus, self.stage, self.message = bus, stage, message
        self.t0 = time.perf_counter()
        self._done = False
        bus.publish(stage, RUNNING, message, detail, progress=None)

    def progress(self, frac: float | None, message: str = "", detail: str = "") -> None:
        self.bus.publish(self.stage, RUNNING, message or self.message, detail, progress=frac, quiet=True)

    def _end(self, state: str, message: str, detail: str = "") -> None:
        if self._done:
            return
        self._done = True
        self.bus.publish(self.stage, state, message or self.message, detail, progress=1.0 if state == OK else None,
                         elapsed=time.perf_counter() - self.t0)

    def ok(self, message: str = "", detail: str = "") -> None:
        self._end(OK, message, detail)

    def warn(self, message: str, detail: str = "") -> None:
        self._end(WARN, message, detail)

    def fail(self, error, detail: str = "") -> None:
        msg = getattr(error, "user_message", None) or (f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error))
        self._end(ERROR, msg, detail)

    def __enter__(self) -> "StageTimer":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if et is not None:
            self.fail(ev)
        elif not self._done:
            self.ok()
        return False


class StatusBus:
    def __init__(self, capacity: int = 1500):
        self._lock = threading.RLock()
        self._seq = itertools.count(1)
        self._events: deque[StatusEvent] = deque(maxlen=capacity)
        self._stage: dict[str, StatusEvent] = {}
        self._started: dict[str, float] = {}
        self._last_elapsed: dict[str, float] = {}
        self._subs: list[tuple[Callable[[StatusEvent], None], str]] = []

    # ------------------------------------------------------------------ publishing
    def publish(self, stage: str, state: str = RUNNING, message: str = "", detail: str = "", progress: float | None = None,
                elapsed: float | None = None, level: str | None = None, quiet: bool = False) -> StatusEvent:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        now = time.time()
        with self._lock:
            if state == RUNNING and (stage not in self._started or self._stage.get(stage) is None
                                     or self._stage[stage].state != RUNNING):
                self._started[stage] = time.perf_counter()
            if elapsed is None and state in (OK, WARN, ERROR) and stage in self._started:
                elapsed = time.perf_counter() - self._started[stage]
            if elapsed is not None and state in (OK, WARN, ERROR):
                self._last_elapsed[stage] = elapsed
            lvl = level or {ERROR: "ERROR", WARN: "WARNING"}.get(state, "DEBUG" if quiet else "INFO")
            ev = StatusEvent(next(self._seq), now, stage, state, message, detail, progress, elapsed, lvl, feed=not quiet)
            self._stage[stage] = ev
            if ev.feed:
                self._events.append(ev)
            subs = list(self._subs)
        self._deliver(ev, subs)
        return ev

    def log(self, level: str, message: str, stage: str = "app", detail: str = "") -> StatusEvent:
        """Activity-feed entry that does not change any stage chip."""
        with self._lock:
            ev = StatusEvent(next(self._seq), time.time(), stage, IDLE, message, detail, None, None, level.upper())
            self._events.append(ev)
            subs = list(self._subs)
        self._deliver(ev, subs, feed_only=True)
        return ev

    def track(self, stage: str, message: str, detail: str = "") -> StageTimer:
        return StageTimer(self, stage, message, detail)

    def _deliver(self, ev: StatusEvent, subs, feed_only: bool = False) -> None:
        for cb, thread in subs:
            if thread == "inline":
                self._call(cb, ev)
            else:
                call_on_gui(self._call, cb, ev)

    @staticmethod
    def _call(cb, ev) -> None:
        try:
            cb(ev)
        except Exception:
            logging.getLogger("app.status").exception("status subscriber failed")

    # ------------------------------------------------------------------ reading
    def subscribe(self, cb: Callable[[StatusEvent], None], thread: str = "gui") -> Callable[[], None]:
        entry = (cb, thread)
        with self._lock:
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)
        return unsubscribe

    def snapshot(self) -> dict[str, StatusEvent]:
        with self._lock:
            return dict(self._stage)

    def state_of(self, stage: str) -> str:
        with self._lock:
            ev = self._stage.get(stage)
            return ev.state if ev else IDLE

    def elapsed_of(self, stage: str) -> float | None:
        with self._lock:
            return self._last_elapsed.get(stage)

    def events(self, min_level: str = "DEBUG", text: str = "", stage: str = "") -> list[StatusEvent]:
        floor = LEVELS.get(min_level.upper(), 10)
        text = text.lower()
        with self._lock:
            evs = list(self._events)
        return [e for e in evs if LEVELS.get(e.level, 20) >= floor and (not stage or e.stage == stage)
                and (not text or text in e.message.lower() or text in e.detail.lower() or text in e.stage)]

    def reset_pipeline(self) -> None:
        """Back to grey at the start of a new question (keeps non-pipeline stages and the feed)."""
        with self._lock:
            for st in PIPELINE:
                if st in self._stage and st not in ("audio", "vad"):
                    del self._stage[st]
        for st in PIPELINE:
            if st not in ("audio", "vad"):
                ev = StatusEvent(next(self._seq), time.time(), st, IDLE, "", feed=False)
                self._deliver(ev, list(self._subs))

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._stage.clear()
            self._started.clear()
            self._last_elapsed.clear()


class BusLogHandler(logging.Handler):
    """Mirrors WARNING+ python log records of the application into the activity feed."""

    def __init__(self, bus: StatusBus, level=logging.WARNING):
        super().__init__(level)
        self.bus = bus
        self._busy = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("app.status") or getattr(self._busy, "on", False):
            return
        self._busy.on = True
        try:
            msg = record.getMessage()
            detail = ""
            if record.exc_info:
                import traceback
                detail = "".join(traceback.format_exception(*record.exc_info))
            self.bus.log(record.levelname, msg.splitlines()[0][:300] if msg else record.name,
                         stage=record.name.split(".")[-1][:12], detail=detail or msg)
        except Exception:
            pass
        finally:
            self._busy.on = False


_default: StatusBus | None = None
_default_lock = threading.Lock()


def get_bus() -> StatusBus:
    global _default
    with _default_lock:
        if _default is None:
            _default = StatusBus()
        return _default


def set_bus(bus: StatusBus | None) -> None:
    global _default
    with _default_lock:
        _default = bus
