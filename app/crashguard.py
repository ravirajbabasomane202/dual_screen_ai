"""Native-crash diagnostics: faulthandler -> logs/crash.log, a 'last operation' breadcrumb, and crash recovery info.

A native crash (segfault from touching Qt off the GUI thread, a DLL fault, ...) leaves no Python traceback. With
`faulthandler` the interpreter writes the stack of every thread to `crash.log`, and `last_operation.json` says what
the application was doing (e.g. "resume.upload: parsing").  A clean shutdown calls `mark_clean_exit()`; if the next
start finds a non-empty crash.log / an unclean marker, the UI offers a recovery report.
"""
from __future__ import annotations

import faulthandler
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("app.crash")

CRASH_LOG, LAST_OP, RUNNING_MARK, PREVIOUS = "crash.log", "last_operation.json", "running.marker", "crash_previous.log"

_state: dict = {"dir": None, "fh": None, "op": "", "ops": []}
_lock = threading.Lock()


@dataclass
class CrashReport:
    happened: bool
    crash_log: str = ""
    last_operation: str = ""
    when: float = 0.0
    path: Path | None = None

    def text(self) -> str:
        return (f"The application closed unexpectedly last time.\nLast operation: {self.last_operation or 'unknown'}\n"
                f"Crash log:\n{self.crash_log or '(no native stack trace was captured)'}")


def breadcrumb(op: str, **detail) -> None:
    """Record what is running right now. Cheap; safe from any thread; never raises."""
    try:
        with _lock:
            _state["op"] = op
            _state["ops"] = (_state["ops"] + [(time.time(), op)])[-30:]
            d = _state["dir"]
            if d is not None:
                (Path(d) / LAST_OP).write_text(json.dumps(
                    {"ts": time.time(), "operation": op, "detail": detail, "thread": threading.current_thread().name,
                     "recent": [o for _, o in _state["ops"][-10:]]}), encoding="utf-8")
    except Exception:
        pass


def last_operation() -> str:
    return _state["op"]


def check_previous_crash(directory: str | Path) -> CrashReport:
    """Inspect what the previous run left behind (call BEFORE install())."""
    d = Path(directory)
    crash = d / CRASH_LOG
    text = ""
    try:
        if crash.exists() and crash.stat().st_size > 0:
            text = crash.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    unclean = (d / RUNNING_MARK).exists()
    if not text and not unclean:
        return CrashReport(False)
    op = ""
    try:
        op = json.loads((d / LAST_OP).read_text(encoding="utf-8")).get("operation", "")
    except Exception:
        pass
    when = 0.0
    try:
        when = (crash if text else d / RUNNING_MARK).stat().st_mtime
    except OSError:
        pass
    if text:
        try:
            shutil.copyfile(crash, d / PREVIOUS)
        except OSError:
            pass
    return CrashReport(True, text, op, when, d / PREVIOUS if text else None)


def install(directory: str | Path) -> Path:
    """Enable faulthandler into <directory>/crash.log, install thread/exception hooks and the running marker."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    with _lock:
        _state["dir"] = d
        old = _state["fh"]
        if old is not None:
            try:
                faulthandler.disable()
                old.close()
            except Exception:
                pass
        fh = open(d / CRASH_LOG, "w", encoding="utf-8")          # truncated: only THIS run's crash is recorded
        _state["fh"] = fh
        faulthandler.enable(file=fh, all_threads=True)
        (d / RUNNING_MARK).write_text(str(os.getpid()), encoding="utf-8")

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        logging.getLogger("app").error("Unhandled exception in thread %s:\n%s", getattr(args.thread, "name", "?"), msg)
    threading.excepthook = thread_hook
    breadcrumb("startup")
    return d / CRASH_LOG


def mark_clean_exit() -> None:
    d = _state["dir"]
    if d is None:
        return
    try:
        faulthandler.disable()
        fh = _state["fh"]
        if fh is not None:
            fh.close()
            _state["fh"] = None
        (Path(d) / CRASH_LOG).write_text("", encoding="utf-8")
        (Path(d) / RUNNING_MARK).unlink(missing_ok=True)
    except Exception:
        pass
    breadcrumb("clean-exit")
