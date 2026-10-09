"""Background tasks with progress, cooperative cancellation and GUI-thread delivery.

    task = run_task(parent, lambda ctx: kb.sync_folder(folder, progress_cb=ctx.progress_cb, cancel=ctx),
                    on_done, stage="kb", title="Indexing PDFs")
    task.cancel()      # sets ctx.cancelled; the worker stops at its next ctx.check() (between files / sections)

`on_done(value, error)` and `on_progress(fraction, message)` always run on the GUI thread (they are slots of a QObject
that lives there). Exceptions inside the worker are returned, never swallowed; a `Cancelled` error is reported with
`error.cancelled == True`. Every task publishes to the StatusBus and leaves a crash breadcrumb.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from app.crashguard import breadcrumb
from app.errors import Cancelled
from app.gui_thread import call_on_gui
from app.status import ERROR, OK, RUNNING, WARN, StatusBus, get_bus

log = logging.getLogger("app.tasks")


class TaskContext:
    """Passed to the worker function: progress reporting + cooperative cancellation."""

    def __init__(self, task: "Task"):
        self._task = task
        self._event = threading.Event()

    # cancellation ----------------------------------------------------------------------------------------
    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:             # threading.Event-compatible, so a ctx can be handed to `cancel=` parameters
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled()

    # progress --------------------------------------------------------------------------------------------
    def progress(self, fraction: float | None, message: str = "") -> None:
        self._task._progress(fraction, message)

    def progress_cb(self, done: int, total: int, name: str = "") -> None:
        """Adapter for `progress_cb(i, n, name)` style callbacks (KnowledgeIndex.sync_folder). Also checks cancel."""
        self.check()
        self.progress((done / total) if total else None, f"{done} of {total}" + (f" — {name}" if name else ""))


class Task:
    def __init__(self, fn: Callable[[TaskContext], object], on_done, on_progress, stage: str, title: str,
                 bus: StatusBus | None):
        self.fn, self.stage, self.title = fn, stage, title
        self._on_done, self._on_progress = on_done, on_progress
        self.bus = bus or get_bus()
        self.ctx = TaskContext(self)
        self.finished = False
        self.started = time.perf_counter()
        self._thread: threading.Thread | None = None
        self._t = None                    # StageTimer

    def cancel(self) -> None:
        self.ctx.cancel()
        self.bus.log("INFO", f"{self.title}: cancelling…", stage=self.stage)

    def start(self) -> "Task":
        self._t = self.bus.track(self.stage, self.title)
        breadcrumb(f"{self.stage}: {self.title}")
        self._thread = threading.Thread(target=self._work, daemon=True, name=f"task-{self.stage}")
        self._thread.start()
        return self

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _progress(self, frac, message) -> None:
        if self._t:
            self._t.progress(frac, f"{self.title} — {message}" if message else self.title)
        if self._on_progress:
            call_on_gui(self._safe_progress, frac, message)

    def _safe_progress(self, frac, message) -> None:
        try:
            self._on_progress(frac, message)
        except Exception:
            log.exception("progress callback failed")

    def _work(self) -> None:
        value, err = None, None
        try:
            value = self.fn(self.ctx)
        except Cancelled as e:
            err = e
        except BaseException as e:                       # surfaced to the user; never swallowed
            err = e
            log.error("task %s failed: %s", self.title, e, exc_info=not hasattr(e, "user_message"))
        self.finished = True
        if err is None:
            self._t.ok(f"{self.title} — done")
            breadcrumb(f"{self.stage}: idle")
        elif getattr(err, "cancelled", False):
            self._t.warn(f"{self.title} — cancelled")
            breadcrumb(f"{self.stage}: cancelled")
        else:
            self._t.fail(err)
            breadcrumb(f"{self.stage}: failed ({type(err).__name__})")
        call_on_gui(self._finish, value, err)

    def _finish(self, value, err) -> None:
        try:
            if self._on_done:
                self._on_done(value, err)
        except Exception:
            log.exception("task completion callback failed")


def run_task(parent, fn: Callable[[TaskContext], object], on_done=None, on_progress=None, *, stage: str = "task",
             title: str = "Working…", bus: StatusBus | None = None) -> Task:
    """Start `fn(ctx)` on a worker thread. `parent` is accepted for API symmetry (callbacks are GUI-thread safe)."""
    return Task(fn, on_done, on_progress, stage, title, bus).start()


def run_async(parent, fn: Callable[[], object], on_done, *, stage: str = "task", title: str = "Working…") -> Task:
    """Back-compatible helper for plain no-argument callables (no progress / cancellation)."""
    return run_task(parent, lambda ctx: fn(), on_done, stage=stage, title=title)
