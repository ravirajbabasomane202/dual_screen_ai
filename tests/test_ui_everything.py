"""Offscreen smoke test: construct every window and page, in every theme, and click every button. Native dialogs are
replaced by stubs; hardware / network steps of the wizard use fakes. No exception may escape a slot and no widget may be
touched from a worker thread (strict mode)."""
import sys
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox, QPushButton, QToolButton)

from app import gui_thread
from database.sqlite import Database
from display import theme
from settings.settings_manager import SettingsManager
from tests.test_resume import CLASSIC


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    gui_thread.set_strict(True)
    return a


@pytest.fixture
def errors(monkeypatch):
    caught = []
    monkeypatch.setattr(sys, "excepthook", lambda t, v, tb: caught.append((t, v)))
    return caught


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    f = tmp_path / "cv.txt"
    f.write_text(CLASSIC)
    doc = tmp_path / "kb"
    doc.mkdir()
    (doc / "a.txt").write_text("Notes about Kubernetes deployment and Helm charts for the platform team.")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(f), "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([str(doc / "a.txt")], "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(doc)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "out.zip"), "")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed", True)))
    from PySide6.QtGui import QColor, QDesktopServices
    from PySide6.QtWidgets import QColorDialog
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#336699")))
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda *a, **k: True))
    from display.pages import knowledge_page
    monkeypatch.setattr(knowledge_page.EntryDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(knowledge_page.EntryDialog, "values", lambda self: (
        {"title": "T", "question": "Why?", "answer": "Because.", "tags": ""} if self.kind == "qa"
        else {"title": "T", "body": "Body text about wombats.", "tags": ""}))
    # wizard fakes (no network, no real audio)
    import audio.audio_manager as am
    import speech.stt_manager as sm

    class FakeAudio:
        def __init__(self, s, cb): self.cb = cb
        def start(self): self.cb(np.full(16000, 0.1, dtype=np.float32))
        def stop(self): pass
    monkeypatch.setattr(am, "AudioManager", FakeAudio)
    monkeypatch.setattr(sm, "is_whisper_downloaded", lambda m: True)
    monkeypatch.setattr(sm, "create_engine", lambda *a, **k: type("E", (), {"load": lambda s: None, "transcribe": lambda s, a: "hello"})())
    import llm.model_manager as mm
    monkeypatch.setattr(mm, "pull_ollama_model", lambda host, model, cb=None: cb and cb(100, "success"))
    monkeypatch.setattr(mm, "detect_local_server", lambda: None)
    return f


def _pump(app, seconds=1.0, until=None):
    end = time.time() + seconds
    while time.time() < end and not (until and until()):
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def click_all(app, root, skip=()):
    n = 0
    for b in root.findChildren(QPushButton) + root.findChildren(QToolButton):
        if b.text() in skip or not b.isEnabled() or b.objectName() == "qt_tabbar_scrollbutton":
            continue
        b.click()
        n += 1
        _pump(app, 0.02)
    return n


def _engine():
    from app.engine import MeetingEngine
    from tests.test_answer_generation import Scripted, reply
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    eng = MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(), hardware=object(),
                        provider=Scripted(["Python is a language used for scripting and data work today.",
                                           reply("What is Python?", "Python is a high-level language used widely today.")] * 4))
    return db, s, eng


@pytest.mark.parametrize("name", ["dark", "light", "contrast"])
def test_ai_window_every_button_in_every_theme(app, errors, stubs, name):
    from display.ai_window import AIWindow
    db, s, eng = _engine()
    s.set("display.theme", name)
    theme.apply_settings(s)
    w = AIWindow(s, health_fn=eng.health)
    sigs = []
    for sig in ("pause_toggled", "process_requested", "clear_requested", "settings_requested", "practice_requested",
                "regenerate_requested", "palette_requested", "export_requested", "why_requested", "mini_toggled",
                "click_through_toggled", "debug_toggled", "answer_rated", "pinned"):
        getattr(w, sig).connect(lambda *a, n=sig: sigs.append(n))
    w.show()
    w.begin_question("raw words", 0.9, "technical")
    w.append_answer("Python is ")
    w.finish_answer("What is Python?", "Python is a language.", 0.9, 1.2, False, "resume", ["Project: X", "cv.pdf p.2"])
    w.add_transcript("what is python", "technical", 0.9, "answer")
    w.add_history("q1", "a1")
    w.show_error("The language model server is not running")
    w.show_recovery("The app closed unexpectedly last time.")
    w.toast("success", "hello")
    w.transcript_toggle.setChecked(True)
    w.history_toggle.setChecked(True)
    w.drawer.open_drawer()
    assert click_all(app, w) > 15
    w.drawer.close_drawer()
    w.set_mini(True)
    w.set_mini(False)
    _pump(app, 0.3)
    assert errors == []
    assert {"pause_toggled", "process_requested", "regenerate_requested", "palette_requested"} <= set(sigs)
    assert w.pipeline.states().keys() >= {"audio", "llm", "display"}


def test_settings_window_every_page_and_button(app, errors, stubs):
    from display.settings_window import SettingsWindow
    db, s, eng = _engine()
    eng.resume.upload_bytes("cv.txt", CLASSIC.encode())
    dlg = SettingsWindow(s, services=eng, db=db)
    dlg.show()
    clicked = 0
    for i in range(dlg.nav.count()):
        dlg.nav.setCurrentRow(i)
        _pump(app, 0.05)
        page = dlg.stack.currentWidget()
        clicked += click_all(app, page, skip=("Pull model",))
        _pump(app, 1.5, until=lambda: not any(getattr(p, "busy", lambda: False)() for p in dlg._extras.values()))
    assert clicked > 40
    _pump(app, 0.5)
    assert errors == []
    dlg.btn_save.click() if dlg.btn_save.isEnabled() else None


def test_wizard_every_step_and_button(app, errors, stubs):
    from display.wizard import SetupWizard
    db, s, eng = _engine()
    wiz = SetupWizard(s, db=db)
    wiz.show()
    for i in range(len(wiz.steps)):
        wiz._show(i)
        _pump(app, 0.05)
        click_all(app, wiz.steps[i], skip=("Cancel",))
        _pump(app, 3.0, until=lambda: all(st.task is None for st in wiz.steps))
    assert wiz.steps[-1].state == "ok"
    wiz._show(0)
    assert wiz.btn_back.isEnabled() is False
    wiz.go(1)
    wiz.skip()
    wiz._show(len(wiz.steps) - 1)
    wiz.btn_next.click()
    assert s.get("app.first_run_done") is True and errors == []


def test_command_palette_shortcuts_and_drawer(app, errors):
    from app.status import StatusBus
    from display.widgets.activity_drawer import ActivityDrawer
    from display.widgets.command_palette import Command, CommandPalette, ShortcutsOverlay
    from PySide6.QtWidgets import QWidget
    host = QWidget()
    host.resize(900, 600)
    host.show()
    bus = StatusBus()
    d = ActivityDrawer(bus, host)
    bus.log("ERROR", "something failed", stage="kb", detail="stack")
    bus.log("INFO", "all good", stage="app")
    _pump(app, 0.1)
    d.open_drawer()
    assert "something failed" in d.visible_text()
    d.set_stage_filter("kb")
    assert "all good" not in d.visible_text()
    hit = []
    p = CommandPalette([Command("Open logs", lambda: hit.append(1)), Command("Pause", lambda: hit.append(2))], host)
    p.open_centered()
    p.filter("logs")
    p.run_current()
    assert hit == [1]
    ShortcutsOverlay([("Ctrl+K", "palette")], host)
    assert errors == []


def test_pipeline_strip_follows_the_bus(app, errors):
    from app.status import ERROR, OK, RUNNING, StatusBus
    from display.widgets.pipeline_strip import PipelineStrip
    bus = StatusBus()
    strip = PipelineStrip(bus)
    bus.publish("stt", RUNNING, "Transcribing")
    bus.publish("llm", ERROR, "down")
    bus.publish("display", OK, "shown")
    _pump(app, 0.1)
    assert strip.states()["stt"] == RUNNING and strip.states()["llm"] == ERROR and strip.states()["display"] == OK
    assert "down" in strip.chips["llm"].toolTip()
    bus.reset_pipeline()
    _pump(app, 0.1)
    assert strip.states()["stt"] == "idle" and strip.states()["llm"] == "idle"
