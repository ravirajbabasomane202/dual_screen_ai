"""Regression for the 'app silently closes after a resume upload' bug, plus stress tests.

Root cause: the upload ran on a worker thread, `settings.set("resume.path")` ran the settings listeners ON THAT THREAD,
and `main.on_settings` restyled windows / moved them between monitors / restarted the engine off the GUI thread.
Fix: settings and resume listeners are always delivered on the GUI thread (app/gui_thread.py)."""
import os
import threading
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from app import gui_thread
from app.tasks import run_task
from database.sqlite import Database
from resume.manager import ResumeManager
from settings.settings_manager import SettingsManager
from tests.test_resume import CLASSIC


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    gui_thread.set_strict(True)
    yield a


def _pump(app, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    return cond()


def _upload_from_worker(app, tmp_path, listener_thread="gui"):
    db = Database(":memory:")
    s = SettingsManager(db)
    mgr = ResumeManager(db, s)
    seen = {"settings": [], "resume": []}
    s.on_change(lambda ch: seen["settings"].append(QThread.currentThread()), thread=listener_thread)
    mgr.on_change(lambda cur: seen["resume"].append(QThread.currentThread()), thread=listener_thread)
    f = tmp_path / "cv.txt"
    f.write_text(CLASSIC)
    done = []
    t = threading.Thread(target=lambda: (mgr.upload(str(f)), done.append(1)))
    t.start()
    assert _pump(app, lambda: bool(done) and seen["settings"] and seen["resume"])
    t.join()
    return seen


def test_listeners_run_on_gui_thread_after_worker_upload(app, tmp_path):
    seen = _upload_from_worker(app, tmp_path)
    gui = QApplication.instance().thread()
    assert seen["settings"] and all(t is gui for t in seen["settings"])
    assert seen["resume"] and all(t is gui for t in seen["resume"])


def test_regression_detects_the_old_behaviour(app, tmp_path):
    """Control: with listeners delivered inline (what the old code did) the listener DOES run on the worker thread,
    so the assertion above fails for the old code and passes for the fix."""
    seen = _upload_from_worker(app, tmp_path, listener_thread="inline")
    gui = QApplication.instance().thread()
    assert any(t is not gui for t in seen["settings"] + seen["resume"])


def test_assert_gui_thread_raises_in_strict_mode(app):
    errs = []

    def work():
        try:
            gui_thread.assert_gui_thread("test")
        except gui_thread.ThreadAffinityError as e:
            errs.append(e)
    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert errs
    gui_thread.assert_gui_thread("test")                      # fine on the GUI thread


def test_widget_methods_reject_worker_threads(app):
    from display.ai_window import AIWindow
    w = AIWindow(SettingsManager(Database(":memory:")))
    errs = []

    def work():
        try:
            w.set_status("ready")
        except gui_thread.ThreadAffinityError as e:
            errs.append(e)
    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert errs


def test_breadcrumb_records_last_operation(tmp_path):
    from app import crashguard
    crashguard.install(tmp_path)
    crashguard.breadcrumb("resume.upload: parsing", file="cv.pdf")
    assert "resume.upload: parsing" in crashguard.last_operation()
    assert (tmp_path / crashguard.CRASH_LOG).exists()
    crashguard.mark_clean_exit()
    assert not crashguard.check_previous_crash(tmp_path).happened


def test_unclean_exit_is_reported_next_start(tmp_path):
    from app import crashguard
    crashguard.install(tmp_path)
    crashguard.breadcrumb("kb: indexing")
    rep = crashguard.check_previous_crash(tmp_path)               # marker still present = the app never exited cleanly
    assert rep.happened and "indexing" in rep.last_operation
    crashguard.mark_clean_exit()


# ------------------------------------------------------------------------------------------------ stress
def _run(app, fn):
    out = {}
    t = run_task(None, fn, lambda v, e: out.update(v=v, e=e), stage="resume", title="stress")
    assert _pump(app, lambda: "e" in out, 30)
    return out["v"], out["e"]


def test_rapid_upload_replace_delete_reparse_never_crashes(app, tmp_path):
    db = Database(":memory:")
    s = SettingsManager(db)
    mgr = ResumeManager(db, s)
    s.on_change(lambda ch: None)
    f = tmp_path / "cv.txt"
    f.write_text(CLASSIC)
    errors = []

    def cycle():
        try:
            for _ in range(4):
                mgr.upload(str(f))
                mgr.upload(str(f), new_profile=True)
                mgr.reparse()
                mgr.delete()
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=cycle) for _ in range(3)]
    [t.start() for t in threads]
    assert _pump(app, lambda: not any(t.is_alive() for t in threads), 60)
    assert errors == []


@pytest.mark.parametrize("name,data,needle", [
    ("corrupt.pdf", os.urandom(4096), ""),
    ("empty.txt", b"", "empty"),
    ("junk.docx", b"not a zip", ""),
])
def test_bad_files_give_a_clear_message_and_keep_the_old_resume(app, name, data, needle):
    db = Database(":memory:")
    mgr = ResumeManager(db, SettingsManager(db))
    mgr.upload_bytes("good.txt", CLASSIC.encode())
    _v, err = _run(app, lambda ctx: mgr.upload_bytes(name, data))
    assert err is not None and getattr(err, "user_message", "") and needle in err.user_message.lower()
    assert mgr.get().filename == "good.txt"


def test_20mb_pdf_does_not_crash_or_block(app):
    db = Database(":memory:")
    mgr = ResumeManager(db, SettingsManager(db))
    big = b"%PDF-1.4\n" + (b"0 obj << /Type /Page >> endobj\n" * 700_000)       # ~20 MB of PDF-looking junk
    assert len(big) > 20_000_000
    worst = [0.0]
    last = [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        worst[0] = max(worst[0], now - last[0])
        last[0] = now
    from PySide6.QtCore import QTimer
    timer = QTimer(interval=10, timeout=tick)
    timer.start()
    _v, err = _run(app, lambda ctx: mgr.upload_bytes("big.pdf", big, cancel=ctx))
    timer.stop()
    assert err is not None and getattr(err, "user_message", "")             # a clear message, not a crash
    assert worst[0] < 1.0


def test_upload_while_engine_is_answering(app, tmp_path):
    from app.engine import MeetingEngine
    from tests.test_answer_generation import Scripted, reply
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    eng = MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(),
                        provider=Scripted([reply("What is Python?", "Python is a language used for many things today.")] * 3),
                        hardware=object())
    s.on_change(lambda ch: None)
    errs = []
    answering = threading.Thread(target=lambda: eng._answer("What is Python?", 0.9, time.monotonic(), manual=True))
    answering.start()
    f = tmp_path / "cv.txt"
    f.write_text(CLASSIC)
    up = threading.Thread(target=lambda: (eng.resume.upload(str(f)) if True else None))
    up.start()
    assert _pump(app, lambda: not up.is_alive(), 30)
    answering.join()
    assert eng.resume.get() is not None and errs == []
