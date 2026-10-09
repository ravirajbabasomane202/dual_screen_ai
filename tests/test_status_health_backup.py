"""StatusBus delivery / ordering, health, error descriptions, diagnostics bundle redaction, feedback, interview profile,
backup / restore and 'wipe all data'."""
import json
import threading
import time
import zipfile

import pytest

from app.backup import export_backup, import_backup, stored_data_report, wipe_all_data
from app.errorinfo import describe
from app.errors import LLMNotRunning, ResumeParseError
from app.health import compute_health
from app.status import ERROR, IDLE, OK, RUNNING, WARN, StatusBus
from database.sqlite import Database
from settings.settings_manager import SettingsManager
from tests.test_resume import CLASSIC


# ------------------------------------------------------------------------------------------------ StatusBus
def test_bus_events_are_ordered_and_state_tracked():
    bus = StatusBus()
    seen = []
    bus.subscribe(lambda ev: seen.append((ev.seq, ev.stage, ev.state)), thread="inline")
    bus.publish("stt", RUNNING, "Transcribing")
    bus.publish("stt", OK, "done")
    bus.publish("llm", ERROR, "boom")
    assert [s[0] for s in seen] == sorted(s[0] for s in seen)
    assert bus.state_of("stt") == OK and bus.state_of("llm") == ERROR and bus.state_of("vad") == IDLE
    assert bus.elapsed_of("stt") is not None


def test_bus_is_thread_safe_and_keeps_every_event_in_order():
    bus = StatusBus(capacity=5000)
    got = []
    bus.subscribe(lambda ev: got.append(ev.seq), thread="inline")

    def spam(stage):
        for i in range(200):
            bus.publish(stage, RUNNING, f"{i}")
    ts = [threading.Thread(target=spam, args=(s,)) for s in ("a", "b", "c", "d")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(got)) == 800
    assert len(bus.events()) == 800


def test_bus_filters_ring_buffer_and_bad_state():
    bus = StatusBus(capacity=10)
    for i in range(30):
        bus.log("INFO" if i % 2 else "ERROR", f"msg {i}", stage="kb")
    assert len(bus.events()) == 10
    assert all(e.level == "ERROR" for e in bus.events(min_level="ERROR"))
    assert bus.events(text="msg 29")
    with pytest.raises(ValueError):
        bus.publish("x", "nonsense")


def test_stage_timer_reports_failure_and_success():
    bus = StatusBus()
    with bus.track("retrieve", "Searching…") as t:
        t.ok("2 chunks")
    assert bus.state_of("retrieve") == OK
    with pytest.raises(RuntimeError):
        with bus.track("llm", "Waiting…"):
            raise RuntimeError("down")
    assert bus.state_of("llm") == ERROR


def test_subscriber_exception_does_not_break_the_bus():
    bus = StatusBus()
    bus.subscribe(lambda ev: 1 / 0, thread="inline")
    ok = []
    bus.subscribe(lambda ev: ok.append(ev), thread="inline")
    bus.publish("stt", OK, "x")
    assert ok


def test_gui_subscribers_get_events_on_the_gui_thread(qapp_fixture=None):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    bus = StatusBus()
    threads = []
    bus.subscribe(lambda ev: threads.append(QThread.currentThread()))
    t = threading.Thread(target=lambda: bus.publish("stt", OK, "x"))
    t.start()
    t.join()
    end = time.time() + 5
    while not threads and time.time() < end:
        app.processEvents()
    assert threads and threads[0] is app.thread()


# ------------------------------------------------------------------------------------------------ health / errors
class _Eng:
    resume = None
    kb = None


def test_health_reflects_bus_and_resume():
    db = Database(":memory:")
    s = SettingsManager(db)
    from app.services import Services
    from context.context_manager import ContextManager
    svc = Services(db, s, ContextManager())
    eng = type("E", (), {"resume": svc.resume, "kb": svc.kb})()
    bus = StatusBus()
    items = {i.key: i for i in compute_health(eng, s, bus)}
    assert {"system_audio", "mic", "stt", "llm", "resume", "kb"} <= set(items)
    assert items["resume"].state == IDLE and items["resume"].action == "page:Resume"
    svc.resume.upload_bytes("cv.txt", CLASSIC.encode())
    bus.publish("llm_model", OK, "qwen ready")
    items = {i.key: i for i in compute_health(eng, s, bus)}
    assert items["resume"].state == OK and items["llm"].state == OK and items["llm"].action == "test:llm"
    bus.publish("llm_model", ERROR, "not running")
    assert {i.key: i for i in compute_health(eng, s, bus)}["llm"].state == ERROR


def test_error_descriptions_are_human_and_actionable():
    info = describe(LLMNotRunning())
    assert "not running" in info.cause.lower() and "ollama" in info.action.lower() and info.page == "LLM"
    info = describe(ResumeParseError("cv.pdf: password-protected"))
    assert "password" in info.cause and info.page == "Resume"
    info = describe(ValueError("weird"))
    assert info.cause == "weird" and "ValueError" in info.details
    assert describe("connection refused by host").page == "LLM"


# ------------------------------------------------------------------------------------------------ bundle / redaction
def test_diagnostics_bundle_is_redacted(tmp_path):
    from app.bundle import export_diagnostics_bundle
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "application.log").write_text("user jane.doe@example.com called +1 555 123 4567\n")
    bus = StatusBus()
    bus.log("INFO", "mail me at secret.person@corp.com", stage="app")
    dest = tmp_path / "d.zip"
    names = export_diagnostics_bundle(dest, logs, bus, SettingsManager(Database(":memory:")))
    with zipfile.ZipFile(dest) as z:
        blob = "".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())
    assert "jane.doe@example.com" not in blob and "secret.person@corp.com" not in blob and "555 123 4567" not in blob
    assert "settings.json" in names and "activity.txt" in names


def test_new_log_lines_do_not_leak_question_text_when_debug_is_off(tmp_path):
    import logging
    from applog import configure, redact
    from knowledge.index import KnowledgeIndex
    configure(tmp_path, debug=False)
    db = Database(":memory:")
    kb = KnowledgeIndex(db)
    kb.add_note("Team", "We use Kubernetes for deployment.", "")
    kb.search("TOPSECRETQUERY kubernetes deployment")
    kb.add_qa("TOPSECRETQUESTION?", "TOPSECRETANSWER text here")
    for h in logging.getLogger().handlers + logging.getLogger("retrieval").handlers:
        h.flush()
    text = "".join(f.read_text(errors="replace") for f in tmp_path.glob("*.log*"))
    assert "kb search" in text
    for secret in ("TOPSECRETQUERY", "TOPSECRETQUESTION", "TOPSECRETANSWER"):
        assert secret not in text
    assert "TOPSECRETQUERY" not in redact("TOPSECRETQUERY")
    configure(tmp_path, debug=True)
    assert "TOPSECRETQUERY" in redact("TOPSECRETQUERY")
    configure(tmp_path, debug=False)


# ------------------------------------------------------------------------------------------------ feedback
def test_question_feedback_override_and_threshold_tuning():
    from question.feedback import NOT_QUESTION, QUESTION, FeedbackStore
    db = Database(":memory:")
    fb = FeedbackStore(db)
    fb.record("so how would you scale it", QUESTION, "STATEMENT", 0.35, "ignored:low_confidence")
    assert fb.override_for("So how would you scale it?") == QUESTION
    fb.record("so how would you scale it", NOT_QUESTION)                    # latest opinion wins
    assert fb.override_for("so how would you scale it") == NOT_QUESTION
    assert fb.suggest_min_confidence(0.30) is None                          # not enough evidence yet
    for i in range(4):
        fb.record(f"missed question number {i} about systems", QUESTION, "STATEMENT", 0.28, "ignored:below_min_confidence")
    new = fb.suggest_min_confidence(0.30)
    assert new is not None and new < 0.30
    fb.mark_tuned()
    assert fb.suggest_min_confidence(new) is None


def test_answer_rating_and_pin():
    from question.feedback import FeedbackStore
    fb = FeedbackStore(Database(":memory:"))
    fb.rate_answer("q", "a", -1)
    fb.pin("q2", "a2")
    assert fb.needs_improvement()[0][1] == "q" and fb.pinned()[0][2] == "q2"


def test_engine_mark_question_tunes_setting_and_answers():
    from app.engine import MeetingEngine
    from tests.test_answer_generation import Scripted, reply
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    eng = MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(), provider=Scripted([]), hardware=object())
    for i in range(3):
        r = eng.mark_question(f"missed sentence {i} about databases here", False if False else True, "STATEMENT", 0.28,
                              "ignored:below_min_confidence")
    assert s.get("question.min_confidence") < 0.30


# ------------------------------------------------------------------------------------------------ interview profile
def test_interview_profile_roundtrip_and_prompt_block():
    from context.interview_profile import InterviewProfile, InterviewProfileStore
    st = InterviewProfileStore(Database(":memory:"))
    assert st.load().empty
    st.save(InterviewProfile("Acme", "ML Engineer", "Build models. " * 500, ""))
    p = st.load()
    assert p.company == "Acme"
    text = p.to_text(300)
    assert "Acme" in text and "ML Engineer" in text and len(text) < 900
    st.delete()
    assert st.load().empty


# ------------------------------------------------------------------------------------------------ backup / wipe
def _services(db, s):
    from app.services import Services
    from context.context_manager import ContextManager
    return Services(db, s, ContextManager())


def test_backup_roundtrip_between_two_databases(tmp_path):
    from context.interview_profile import InterviewProfile, InterviewProfileStore
    from question.feedback import FeedbackStore
    db1 = Database(":memory:")
    s1 = SettingsManager(db1)
    s1.set("ai.temperature", 0.7)
    sv1 = _services(db1, s1)
    sv1.resume.upload_bytes("cv.txt", CLASSIC.encode(), label="Main")
    sv1.resume.edit_fields({"name": "Jane Q. Doe"})
    sv1.kb.add_note("Team", "We use Kubernetes.", "infra")
    sv1.kb.add_qa("Why us?", "Because of the mission.")
    InterviewProfileStore(db1).save(InterviewProfile("Acme", "SRE", "jd", ""))
    FeedbackStore(db1).record("is this a question", "question")
    z = tmp_path / "b.zip"
    counts = export_backup(z, db1, s1, sv1)
    assert counts["resume_profiles"] == 1 and counts["kb_entries"] == 2

    db2 = Database(":memory:")
    s2 = SettingsManager(db2)
    sv2 = _services(db2, s2)
    done = import_backup(z, db2, s2, sv2)
    assert done["resume_profiles"] == 1 and done["kb_entries"] == 2 and done["feedback"] == 1
    assert s2.get("ai.temperature") == 0.7
    assert sv2.resume.get().parsed.name == "Jane Q. Doe"
    assert InterviewProfileStore(db2).load().company == "Acme"
    again = import_backup(z, db2, s2, sv2)                               # importing twice does not duplicate
    assert again["resume_profiles"] == 0 and again["kb_entries"] == 0


def test_backup_rejects_foreign_files(tmp_path):
    from app.backup import BackupError
    f = tmp_path / "x.zip"
    f.write_bytes(b"nope")
    db = Database(":memory:")
    with pytest.raises(BackupError):
        import_backup(f, db, SettingsManager(db))


def test_wipe_all_data_removes_everything_but_keeps_settings(tmp_path):
    db = Database(":memory:")
    s = SettingsManager(db)
    s.set("ai.temperature", 0.9)
    sv = _services(db, s)
    sv.resume.upload_bytes("cv.txt", CLASSIC.encode())
    sv.kb.add_note("n", "text body here", "")
    db.add_question("q", "a")
    report = stored_data_report(db, s, tmp_path, sv)
    assert any("Resume" in r[0] and "1 profile" in r[1] for r in report)
    wipe_all_data(db, s, tmp_path, sv)
    assert sv.resume.get() is None and sv.kb.stats()["documents"] == 0 and db.recent_questions() == []
    assert s.get("ai.temperature") == 0.9


def test_automatic_backup_before_schema_migration(tmp_path):
    import sqlite3
    old = tmp_path / "old.db"
    c = sqlite3.connect(old)
    c.executescript("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL); INSERT INTO settings VALUES('ai.model','\"x\"');")
    c.commit()
    c.close()
    Database(old)
    assert list((tmp_path / "backups").glob("old-v*.db"))
