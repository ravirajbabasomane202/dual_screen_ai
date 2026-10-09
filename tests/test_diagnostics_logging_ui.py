import json
import os

import pytest

from applog import configure, is_debug, redact, scrub_pii
from database.sqlite import Database
from diagnostics.runner import FAIL, PASS, SKIP, run_full_diagnostics
from llm.diagnostics import recommend_action, test_llm as run_test_llm
from settings.settings_manager import SettingsManager
from tests.test_answer_generation import Scripted, reply


def test_migrations_create_all_tables_and_adopt_old_databases(tmp_path):
    import sqlite3
    old = tmp_path / "old.db"
    c = sqlite3.connect(old)
    c.executescript("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL); INSERT INTO settings VALUES('ai.model','\"x\"');"
                    "CREATE TABLE questions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, question TEXT, answer TEXT);"
                    "CREATE TABLE context (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, question TEXT, answer TEXT);")
    c.commit(); c.close()
    db = Database(old)
    from database.migrations import latest_version
    assert db.schema_version == latest_version() and db.all_kv() == {"ai.model": "x"}
    tables = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"resume_profiles", "kb_documents", "kb_chunks", "interview_context", "schema_version"} <= tables
    assert Database(old).schema_version == latest_version()                       # idempotent


def test_log_files_and_levels(tmp_path):
    import logging
    configure(tmp_path, debug=False)
    from applog import log_event
    logging.getLogger("app.x").warning("hello app")
    log_event("retrieval", "r", query=redact("secret question"))
    log_event("question", "q")
    logging.getLogger("app.x").error("boom")
    for name in ("application.log", "llm.log", "question_detection.log", "retrieval.log", "errors.log"):
        configure(tmp_path, debug=False)
    assert "hello app" in (tmp_path / "application.log").read_text()
    assert "secret question" not in (tmp_path / "retrieval.log").read_text()
    assert "boom" in (tmp_path / "errors.log").read_text()
    assert not is_debug()
    configure(tmp_path, debug=True)
    assert is_debug() and "secret" in redact("secret question")
    configure(tmp_path, debug=False)


def test_scrubbing():
    t = scrub_pii("mail me at a.b@c.com or +1 (555) 123-4567, id 1234567890123")
    assert "@" not in t and "555" not in t and "1234567890123" not in t


def test_test_llm_success_and_formatting():
    prov = Scripted(["Python is a high-level programming language used for scripting and data analysis.",
                     reply("What is Python?", "Python is a high-level programming language used widely today.")])
    r = run_test_llm(prov)
    out = r.format()
    assert r.ok and "Connection: OK" in out and "Generation: OK" in out and "Parsing: OK" in out and "Latency:" in out and "Python is" in out


def test_test_llm_reports_each_failure_stage():
    from app.errors import LLMNotRunning

    class Down(Scripted):
        def is_available(self): return False
    r = run_test_llm(Down([]))
    assert not r.connection and "FAILED" in r.format() and "Recommended action" in r.format() and "Ollama" in r.recommendation
    r = run_test_llm(Scripted([""]))
    assert r.connection and not r.generation and "empty" in r.error.lower()
    r = run_test_llm(Scripted(["Fine.", "not json"] * 3))
    assert r.generation and not r.ok
    r = run_test_llm(Scripted(["Fine reply here for plain.", reply("q", "The speaker is asking what Python is.")] + [reply("q", "The speaker is asking what Python is.")] * 3))
    assert r.parsing and not r.answer_valid and "validation" in r.error.lower()


def test_recommendations():
    from app.errors import LLMContextOverflow, LLMTimeout
    s = SettingsManager(Database(":memory:"))
    assert "from 4 to 2" in recommend_action(LLMContextOverflow("x"), s)
    assert "timeout" in recommend_action(LLMTimeout(), s).lower()


def test_full_diagnostics_report(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    s = SettingsManager(Database(":memory:"))
    rep = run_full_diagnostics(s, skip_hardware=True)
    for name in ("Database", "Question Detector", "Context Manager", "Resume Parser", "PDF Parser", "PDF Retrieval",
                 "Answer Parser", "Display"):
        assert rep.get(name).status == PASS, (name, rep.get(name))
    assert rep.get("Microphone").status == SKIP
    assert rep.get("LLM Connection").status == FAIL and rep.get("LLM Generation").status == SKIP   # nothing listens on :11434
    text = rep.format()
    assert "SYSTEM DIAGNOSTICS" in text and "Resume Parser ........" in text and "Recommended action" in text and "Ollama" in text
    assert not rep.ok


def test_diagnostics_use_injected_working_llm():
    from app.engine import MeetingEngine
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    prov = Scripted(["Python is a high-level programming language used for scripting and data analysis.",
                     reply("What is Python?", "Python is a high-level programming language used widely today.")])
    eng = MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(), provider=prov, hardware=object())
    rep = run_full_diagnostics(s, db, engine=eng, skip_hardware=True)
    assert rep.get("LLM Connection").status == PASS and rep.get("LLM Generation").status == PASS


# ------------------------------------------------------------------------------------------------------- UI
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from display.ai_window import AIWindow
from display.settings_window import SettingsWindow


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _engine():
    from app.engine import MeetingEngine
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    return s, MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(), provider=Scripted([]), hardware=object())


def test_settings_window_has_all_new_tabs_and_saves_values(app):
    s, eng = _engine()
    dlg = SettingsWindow(s, services=eng)
    names = [dlg.nav.item(i).data(0x100) for i in range(dlg.nav.count())]       # sidebar replaced the old tab bar
    assert {"Resume", "Knowledge Base", "LLM", "Debug & Logs", "Context & Memory"} <= set(names)
    dlg._widgets["debug.enabled"][1].setChecked(True)
    dlg._widgets["retrieval.top_k"][1].setValue(2)
    dlg._widgets["ai.context_window"][1].setValue(4096)
    dlg._save()
    assert s.get("debug.enabled") is True and s.get("retrieval.top_k") == 2 and s.get("ai.context_window") == 4096


def test_settings_window_without_services_still_opens(app):
    SettingsWindow(SettingsManager(Database(":memory:")))


def test_resume_page_upload_view_delete(app, tmp_path):
    from display.settings_pages import ResumePage
    from tests.test_resume import CLASSIC
    s, eng = _engine()
    page = ResumePage(eng)
    assert "No resume" in page.view.toPlainText()
    eng.resume.upload_bytes("cv.txt", CLASSIC.encode())
    page.refresh()
    assert "Acme Corp" in page.view.toPlainText() and page.buttons["del"].isEnabled()
    eng.resume.delete(); page.refresh()
    assert not page.buttons["re"].isEnabled()


def test_knowledge_page_indexes_in_background(app, tmp_path):
    import time
    from display.settings_pages import KnowledgePage
    from tests.pdfgen import make_pdf
    from tests.test_knowledge import PY_QA
    (tmp_path / "a.pdf").write_bytes(make_pdf(PY_QA))
    s, eng = _engine()
    page = KnowledgePage(eng, s)
    page.folder.setText(str(tmp_path))
    page.index(False)
    for _ in range(100):
        app.processEvents(); time.sleep(0.05)
        if "added" in page.status.text():
            break
    assert "1 added" in page.status.text() and "a.pdf" in page.view.toPlainText() and s.get("knowledge.folder") == str(tmp_path)


def test_llm_and_debug_pages_run_checks(app):
    import time
    from display.settings_pages import DebugPage, LLMTestPage
    s, eng = _engine()
    eng._provider = Scripted(["Python is a high-level programming language used for scripting and data work.",
                              reply("What is Python?", "Python is a high-level programming language used widely today.")])
    p = LLMTestPage(eng)
    p.run()
    for _ in range(100):
        app.processEvents(); time.sleep(0.05)
        if "Connection" in p.out.toPlainText():
            break
    assert "Connection: OK" in p.out.toPlainText()


def test_ai_window_debug_panel_and_regeneration(app):
    s = SettingsManager(Database(":memory:"))
    w = AIWindow(s)
    assert not w.debug.isVisibleTo(w)
    w.set_debug_visible(True); assert w.debug.isVisibleTo(w)
    w.show_debug("TRACE"); assert "TRACE" in w.debug.toPlainText()
    w.begin_question("q"); w.append_answer("bad start"); w.reset_answer()
    assert w.answer.toPlainText().strip() == ""
    w.finish_answer("Q?", "Real answer.", 0.9, 1.0, False, "resume", ["Project: X"])
    assert "your resume" in w.footer.text() and "Project: X" in w.footer.text()
