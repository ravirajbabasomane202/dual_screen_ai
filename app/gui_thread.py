"""GUI-thread affinity helpers.

Qt widgets, stylesheets and window flags may only be touched from the GUI thread. Touching them from a worker thread
is undefined behaviour and usually kills the process natively (no Python traceback). Everything that can be reached
from a worker thread and that ends up in UI code therefore goes through `call_on_gui`.

* `call_on_gui(fn, *a, **kw)`  run fn on the GUI thread (immediately if we already are on it, queued otherwise).
* `is_gui_thread()`             True on the GUI thread, and always True when no QApplication exists (headless/tests).
* `assert_gui_thread(what)`     raises ThreadAffinityError in strict mode (debug mode / tests), otherwise logs once.
* `set_strict(True)`            strict mode switch (the app turns it on with debug mode; tests turn it on always).
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("app.threads")

_strict = os.environ.get("DSAI_STRICT_THREADS", "") not in ("", "0", "false")
_warned: set[str] = set()
_dispatcher = None
_dispatcher_lock = threading.Lock()


class ThreadAffinityError(RuntimeError):
    """A GUI object was used from a non-GUI thread."""


def set_strict(on: bool) -> None:
    global _strict
    _strict = bool(on)


def is_strict() -> bool:
    return _strict


def _qt():
    try:
        from PySide6.QtCore import QCoreApplication, QThread
        return QCoreApplication, QThread
    except Exception:          # pragma: no cover - PySide6 missing
        return None, None


def is_gui_thread() -> bool:
    QCoreApplication, QThread = _qt()
    if QCoreApplication is None:
        return True
    app = QCoreApplication.instance()
    if app is None:
        return True            # headless: there is no GUI thread to protect
    return QThread.currentThread() is app.thread()


def assert_gui_thread(what: str = "") -> None:
    """Call at the top of widget-mutating methods."""
    if is_gui_thread():
        return
    QCoreApplication, QThread = _qt()
    name = threading.current_thread().name
    msg = f"{what or 'GUI object'} used from non-GUI thread '{name}'"
    if _strict:
        raise ThreadAffinityError(msg)
    if msg not in _warned:                 # production: leave a breadcrumb, don't spam
        _warned.add(msg)
        log.error(msg, stack_info=True)


def _ensure_dispatcher():
    """Create the dispatcher object (living in the GUI thread) once. Safe to call from any thread."""
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal, Slot

    with _dispatcher_lock:
        if _dispatcher is not None:
            return _dispatcher

        class Dispatcher(QObject):
            call = Signal(object)

            def __init__(self):
                super().__init__()
                self.call.connect(self._run, Qt.QueuedConnection)

            @Slot(object)
            def _run(self, packed):
                fn, args, kwargs = packed
                try:
                    fn(*args, **kwargs)
                except Exception:           # a failing callback must never take the event loop down
                    log.exception("GUI callback %r failed", getattr(fn, "__name__", fn))

        d = Dispatcher()
        app = QCoreApplication.instance()
        if app is not None and d.thread() is not app.thread():
            d.moveToThread(app.thread())          # created from a worker thread: hand it to the GUI thread
        _dispatcher = d
        return d


def reset_dispatcher() -> None:
    """Tests create/destroy QApplications; drop the cached dispatcher so the next call builds a fresh one."""
    global _dispatcher
    _dispatcher = None


def call_on_gui(fn, *args, **kwargs) -> None:
    """Run fn(*args, **kwargs) on the GUI thread. Synchronous when already there (or headless), queued otherwise."""
    if is_gui_thread():
        fn(*args, **kwargs)
        return
    _emit((fn, args, kwargs))


def _emit(packed) -> None:
    """Queue to the dispatcher. If the QApplication was torn down meanwhile (shutdown, tests) the cached dispatcher is
    deleted: rebuild it once, and if there is still no event loop drop the call instead of killing the worker thread."""
    try:
        _ensure_dispatcher().call.emit(packed)
    except RuntimeError:
        reset_dispatcher()
        try:
            _ensure_dispatcher().call.emit(packed)
        except RuntimeError:
            log.debug("GUI callback dropped: no event loop (application is shutting down)")


def post_to_gui(fn, *args, **kwargs) -> None:
    """Like call_on_gui but never runs inline when a GUI thread exists (always deferred to the event loop)."""
    QCoreApplication, _ = _qt()
    if QCoreApplication is None or QCoreApplication.instance() is None:
        fn(*args, **kwargs)
        return
    _emit((fn, args, kwargs))
