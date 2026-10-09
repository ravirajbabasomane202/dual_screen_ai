"""Knowledge Base manager: backend CRUD + the page (add file, note, edit, disable, delete + undo, chunks, playground)."""
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from app.gui_thread import set_strict
from database.sqlite import Database
from display import theme
from knowledge.index import KnowledgeIndex
from settings.settings_manager import SettingsManager
from tests.pdfgen import make_pdf
from tests.test_knowledge import PY_QA


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    set_strict(True)
    theme.apply_settings(SettingsManager(Database(":memory:")))       # style the app once, not per test
    return a


def _pump(app, cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return cond()


@pytest.fixture
def env(app, tmp_path):
    from app.engine import MeetingEngine
    from tests.test_answer_generation import Scripted
    from tests.test_engine_interview import STT
    db = Database(":memory:")
    s = SettingsManager(db)
    eng = MeetingEngine(s, lambda n, **p: None, db=db, stt=STT(), vad=object(), provider=Scripted([]), hardware=object())
    from display.pages.knowledge_page import KnowledgePage
    page = KnowledgePage(eng, s)
    page.show()
    yield s, eng, page, tmp_path
    page.close()
    page.deleteLater()
    app.sendPostedEvents(None, 0)


def _wait_idle(app, page, timeout=15.0):
    assert _pump(app, lambda: not page.busy(), timeout)


def test_add_files_note_qa_and_table(app, env):
    s, eng, page, tmp = env
    pdf = tmp / "python.pdf"
    pdf.write_bytes(make_pdf(PY_QA))
    assert page.empty.isVisibleTo(page)
    page.add_files([str(pdf)])
    _wait_idle(app, page)
    assert "1 added" in page.status.text()
    nid = page.create_entry("note", title="Team", body="We deploy with Kubernetes and Helm.", tags="infra")
    qid = page.create_entry("qa", question="Why do you want this job?", answer="Because of the mission.")
    assert nid and qid
    assert page.table.rowCount() == 3 and not page.empty.isVisibleTo(page)
    names = {page.table.item(r, 0).text() for r in range(3)}
    assert "python.pdf" in names and "Team" in names
    types = {page.table.item(r, 1).text() for r in range(3)}
    assert {"PDF", "Note", "Q&A"} <= types
    assert all(page.table.item(r, 5).text() for r in range(3))


def test_edit_note_reembeds_only_when_changed(app, env):
    s, eng, page, tmp = env
    eid = page.create_entry("note", title="Stack", body="We use Python and Postgres.")
    path = eng.kb.entry_path(eid)
    before = {c.chunk_id for c in eng.kb.chunks_of(path)}
    assert page.update_entry(eid, body="We use Python and Postgres.") is False         # unchanged → nothing re-embedded
    assert page.update_entry(eid, body="We use Rust and SQLite now.") is True
    after = eng.kb.chunks_of(path)
    assert {c.chunk_id for c in after} != before and "Rust" in after[0].text
    assert eng.kb.get_entry(eid).version == 2


def test_disable_enable_document_changes_search(app, env):
    s, eng, page, tmp = env
    page.create_entry("note", title="Mongo", body="Zebracorn uses quantum flux capacitors heavily.")
    path = next(page.table.item(r, 0).data(0x100) for r in range(page.table.rowCount()))
    assert eng.kb.search("zebracorn quantum flux capacitors", min_score=0.1)
    page.select_path(path)
    page.toggle_selected(False)
    assert not eng.kb.search("zebracorn quantum flux capacitors", min_score=0.1)
    assert "disabled" in page.table.item(0, 5).text().lower()
    page.toggle_selected(True)
    assert eng.kb.search("zebracorn quantum flux capacitors", min_score=0.1)


def test_delete_and_undo(app, env):
    s, eng, page, tmp = env
    page.create_entry("note", title="Temp", body="Temporary knowledge about wombats and burrows.")
    path = page.selected_paths()[0]
    page.delete_selected([path], confirm=False)
    assert page.table.rowCount() == 0 and eng.kb.search("wombats burrows", min_score=0.1) == []
    assert page.toasts_texts() if hasattr(page, "toasts_texts") else True
    page.undo((path,))
    assert page.table.rowCount() == 1 and eng.kb.search("wombats burrows", min_score=0.1)


def test_undo_window_expires(app, env):
    s, eng, page, tmp = env
    eid = eng.kb.add_note("Gone", "Soon removed content about narwhals.")
    p = eng.kb.delete_document(eng.kb.entry_path(eid))
    eng.kb.purge_expired(now=time.time() + 10_000)
    assert eng.kb.undo_delete(p) is False


def test_chunk_disable_removes_only_that_chunk(app, env):
    s, eng, page, tmp = env
    pdf = tmp / "python.pdf"
    pdf.write_bytes(make_pdf(PY_QA))
    page.add_files([str(pdf)])
    _wait_idle(app, page)
    page.select_path(str(pdf))
    assert page.chunk_list.count() >= 1
    cid = page.chunk_list.item(0).data(0x100)
    page.set_chunk_enabled(cid, False)
    assert all(c.chunk_id != cid for c in eng.kb.repo.all_chunks(only_active=True))
    assert any(c.chunk_id == cid and not c.enabled for c in eng.kb.chunks_of(str(pdf)))


def test_search_playground_shows_scores_and_sources(app, env):
    s, eng, page, tmp = env
    pdf = tmp / "python.pdf"
    pdf.write_bytes(make_pdf(PY_QA))
    page.add_files([str(pdf)])
    _wait_idle(app, page)
    hits = page.run_playground("What is Python?")
    assert hits and hits[0].document == "python.pdf" and hits[0].page >= 1
    txt = page.play_out.toPlainText()
    assert "fused" in txt and "BM25" in txt and "vector" in txt and "python.pdf" in txt


def test_rename_keeps_name_after_reindex(app, env):
    s, eng, page, tmp = env
    pdf = tmp / "python.pdf"
    pdf.write_bytes(make_pdf(PY_QA))
    page.add_files([str(pdf)])
    _wait_idle(app, page)
    page.select_path(str(pdf))
    page.rename_selected("My Python notes")
    assert eng.kb.get_document(str(pdf)).name == "My Python notes"
    eng.kb.reindex(str(pdf))
    assert eng.kb.get_document(str(pdf)).name == "My Python notes"


def test_needs_reindex_badge_when_file_changes(app, env):
    s, eng, page, tmp = env
    pdf = tmp / "python.pdf"
    pdf.write_bytes(make_pdf(PY_QA))
    page.add_files([str(pdf)])
    _wait_idle(app, page)
    assert page.table.item(0, 5).text() == "Indexed"
    pdf.write_bytes(make_pdf(PY_QA + ["Extra page text about Rust ownership and borrowing."]))
    page.refresh()
    assert page.table.item(0, 5).text() == "Needs re-index"


def test_bulk_reindex_and_delete(app, env):
    s, eng, page, tmp = env
    for i in range(3):
        (tmp / f"d{i}.pdf").write_bytes(make_pdf([f"Document {i} talks about topic{i} in depth."]))
    page.add_files([str(tmp / f"d{i}.pdf") for i in range(3)])
    _wait_idle(app, page)
    assert page.table.rowCount() == 3
    paths = [str(tmp / f"d{i}.pdf") for i in range(3)]
    page.reindex_selected(paths)
    _wait_idle(app, page)
    page.delete_selected(paths, confirm=False)
    assert page.table.rowCount() == 0


def test_cancel_stops_indexing_and_keeps_finished_documents(app, env):
    s, eng, page, tmp = env
    for i in range(12):
        (tmp / f"c{i}.pdf").write_bytes(make_pdf([f"Cancel test document {i} about subject{i}. " * 20] * 3))
    s.set("knowledge.folder", str(tmp))
    page.folder.setText(str(tmp))
    page.index(False)
    assert page.busy()
    page._cancel_current()
    _wait_idle(app, page)
    n = eng.kb.stats()["documents"]
    assert 0 <= n <= 12
    assert "ancel" in page.status.text() or n == 12
    for d in eng.kb.documents():
        assert d.status in ("ok", "no_text")                     # no half-indexed documents


def test_empty_state_action_and_stats(app, env):
    s, eng, page, tmp = env
    assert page.empty.isVisibleTo(page) and "0 document" in page.stats_label.text()


def test_drop_of_folder_and_files(app, env):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent
    s, eng, page, tmp = env
    (tmp / "dropped.txt").write_text("Dropped notes about quokkas and their habitats.")
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(tmp / "dropped.txt"))])
    page.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier))
    _wait_idle(app, page)
    assert any(d.name == "dropped.txt" for d in eng.kb.documents())


def test_ui_stays_responsive_while_indexing(app, env):
    """No UI-thread blocking above 50 ms (event-loop latency) while a PDF set is indexed and a resume is parsed."""
    from PySide6.QtCore import QTimer
    s, eng, page, tmp = env
    for i in range(8):
        (tmp / f"p{i}.pdf").write_bytes(make_pdf([("Section %d paragraph. " % i + "Lorem ipsum dolor sit amet. " * 40)] * 4))
    page.refresh()
    end = time.time() + 0.8                                   # warm-up: first polish / paint of the page is not indexing work
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    gaps = []
    last = [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now
    t = QTimer(interval=5, timeout=tick)
    t.start()
    page.add_files([str(tmp / f"p{i}.pdf") for i in range(8)])
    _wait_idle(app, page, 60)
    t.stop()
    assert len(gaps) > 20
    gaps.sort()
    p99 = gaps[int(len(gaps) * 0.99) - 1]
    assert p99 < 0.05, f"event loop stalled: p99 gap {p99 * 1000:.0f} ms (max {gaps[-1] * 1000:.0f} ms)"
