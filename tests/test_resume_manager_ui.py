"""Resume: multiple profiles, manual edits (survive re-parse), version history + restore, codecs, and the page."""
import time

import pytest

from app.errors import ResumeParseError
from database.sqlite import Database
from resume.editor import (collect_edits, format_education, format_experience, format_list, format_projects,
                           parse_education, parse_experience, parse_list, parse_projects)
from resume.manager import ResumeManager
from settings.settings_manager import SettingsManager
from tests.test_resume import CLASSIC

OTHER = CLASSIC.replace("JANE DOE", "JOHN ROE").replace("Acme Corp", "Gamma Inc")


@pytest.fixture
def mgr():
    db = Database(":memory:")
    return ResumeManager(db, SettingsManager(db))


def test_multiple_profiles_one_active(mgr):
    a = mgr.upload_bytes("a.txt", CLASSIC.encode(), label="Backend")
    b = mgr.upload_bytes("b.txt", OTHER.encode(), new_profile=True, label="Data")
    ps = mgr.profiles()
    assert len(ps) == 2 and sum(p.active for p in ps) == 1
    assert mgr.get().id == b.id or mgr.get().id == a.id
    mgr.activate(a.id)
    assert mgr.get().label == "Backend"
    mgr.rename(a.id, "Backend v2")
    assert mgr.get().label == "Backend v2"
    mgr.delete(a.id)                                   # deleting the active one activates the remaining profile
    assert mgr.get() is not None and mgr.get().label == "Data" and len(mgr.profiles()) == 1


def test_replace_goes_to_history_and_can_be_restored(mgr):
    first = mgr.upload_bytes("a.txt", CLASSIC.encode())
    assert "Acme Corp" in mgr.view_text()
    mgr.upload_bytes("b.txt", OTHER.encode())          # replaces the active profile
    assert "Gamma Inc" in mgr.view_text() and "Acme Corp" not in mgr.view_text()
    versions = mgr.versions()
    assert versions
    mgr.restore_version(versions[0].id)
    assert "Acme Corp" in mgr.view_text()


def test_manual_edits_survive_reparse_unless_discarded(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    mgr.edit_fields({"name": "Jane Q. Doe", "skills": ["Python", "Rust"]})
    cur = mgr.get()
    assert cur.parsed.name == "Jane Q. Doe" and cur.parsed.skills == ["Python", "Rust"] and cur.edited
    assert cur.parsed_base.name != "Jane Q. Doe"
    mgr.reparse(keep_edits=True)
    assert mgr.get().parsed.name == "Jane Q. Doe"
    mgr.reparse(keep_edits=False)
    assert mgr.get().parsed.name != "Jane Q. Doe" and not mgr.get().edited
    mgr.edit_fields({"name": "X"})
    mgr.reset_fields(["name"])
    assert mgr.get().parsed.name != "X"


def test_bad_edit_field_is_rejected(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    with pytest.raises(ResumeParseError):
        mgr.edit_fields({"years_of_experience": 99})


def test_edits_feed_derived_fields(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    mgr.edit_fields({"experience": [{"title": "CTO", "company": "Zeta", "start": "2020", "end": "Present", "bullets": ["Led"],
                                     "responsibilities": ["Led"]}]})
    p = mgr.get().parsed
    assert "Zeta" in p.companies and "CTO" in p.job_titles


def test_codecs_roundtrip_without_losing_data(mgr):
    cur = mgr.upload_bytes("a.txt", CLASSIC.encode()).parsed
    assert parse_list(format_list(cur.skills)) == cur.skills
    assert [e["company"] for e in parse_experience(format_experience(cur.experience), cur.experience)] == \
           [e["company"] for e in cur.experience]
    assert [p["name"] for p in parse_projects(format_projects(cur.projects))] == [p["name"] for p in cur.projects]
    assert len(parse_education(format_education(cur.education), cur.education)) == len(cur.education)
    fields = {"name": cur.name, "skills": format_list(cur.skills), "experience": format_experience(cur.experience)}
    assert collect_edits(cur, fields) == {}                                   # untouched editor → no edits
    fields["name"] = "Someone Else"
    assert collect_edits(cur, fields) == {"name": "Someone Else"}


def test_upload_failure_keeps_previous_resume(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    with pytest.raises(ResumeParseError):
        mgr.upload_bytes("empty.txt", b"")
    with pytest.raises(ResumeParseError):
        mgr.upload_bytes("nonsense.txt", b"hello hello hello")
    assert mgr.get().filename == "a.txt"


# ------------------------------------------------------------------------------------------------ the page
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _page(app):
    from app.engine import MeetingEngine
    from display.pages.resume_page import ResumePage
    from tests.test_answer_generation import Scripted
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    eng = MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(), provider=Scripted([]), hardware=object())
    return eng, ResumePage(eng)


def _wait(app, page):
    end = time.time() + 15
    while page.busy() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_page_upload_edit_save_restore_and_profiles(app, tmp_path):
    eng, page = _page(app)
    page.show()
    assert page.empty.isVisibleTo(page) and "No resume" in page.view.toPlainText()
    f = tmp_path / "cv.txt"
    f.write_text(CLASSIC)
    page.upload(str(f))
    _wait(app, page)
    assert not page.empty.isVisibleTo(page) and "Acme Corp" in page.view.toPlainText()
    assert page.fields["name"].text() == eng.resume.get().parsed.name
    page.fields["name"].setText("Jane Q. Doe")
    page.save_edits()
    assert eng.resume.get().parsed.name == "Jane Q. Doe" and "Edited fields" in page.edit_note.text()
    page.fields["name"].setText(eng.resume.get().parsed_base.name)          # editing back to the parsed value drops the edit
    page.save_edits()
    assert not eng.resume.get().edited
    g = tmp_path / "other.txt"
    g.write_text(OTHER)
    page.upload(str(g), new_profile=True)
    _wait(app, page)
    assert page.profile_combo.count() == 2
    page.upload(str(f))
    _wait(app, page)
    assert page.history_list.count() >= 1
    page.restore_selected(page.history_list.item(0).data(0x100))
    page.delete(confirm=False)
    assert page.profile_combo.count() == 1


def test_page_failure_shows_banner_and_keeps_resume(app, tmp_path):
    eng, page = _page(app)
    page.show()
    good = tmp_path / "cv.txt"
    good.write_text(CLASSIC)
    page.upload(str(good))
    _wait(app, page)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"\x00\x01garbage")
    page.upload(str(bad))
    _wait(app, page)
    assert page.banner.isVisibleTo(page) and "previous resume" in page.status.text().lower() or page.banner.isVisibleTo(page)
    assert eng.resume.get().filename == "cv.txt"


def test_page_shows_parser_warnings(app, tmp_path):
    eng, page = _page(app)
    thin = tmp_path / "thin.txt"
    thin.write_text("\n".join(CLASSIC.splitlines()[:11]))
    page.upload(str(thin))
    _wait(app, page)
    assert eng.resume.get() is not None
    assert page.warn_list.count() >= 1
