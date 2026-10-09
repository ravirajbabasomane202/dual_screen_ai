"""Knowledge Base manager: document table, per-document actions, notes / Q&A editor, chunk viewer, search playground."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHeaderView,
                               QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from display import theme
from display.pages.base import ActionPage
from display.widgets.common import SearchBox, button, hbox, label
from display.widgets.feedback import EmptyState
from knowledge.index import NOTE_SCHEME, QA_SCHEME

ACCEPT = (".pdf", ".docx", ".txt", ".md")
COLS = ("Name", "Type", "Pages", "Chunks", "Size", "Status", "Last indexed", "Source")
_TYPE = {"note": "Note", "qa": "Q&A"}
_SOURCE = {"folder": "Folder", "file": "File", "note": "Note", "qa": "Q&A"}


def fmt_size(n: int) -> str:
    n = float(n or 0)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return ""


class _Num(QTableWidgetItem):
    def __init__(self, value: int, text: str = ""):
        super().__init__(text or str(value))
        self.setData(Qt.UserRole + 1, value)
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other) -> bool:
        return (self.data(Qt.UserRole + 1) or 0) < (other.data(Qt.UserRole + 1) or 0)


class EntryDialog(QDialog):
    """New / edit a note or a Q&A pair."""

    def __init__(self, kind: str, parent=None, title: str = "", body: str = "", question: str = "", answer: str = "", tags: str = ""):
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle(("Q&A pair" if kind == "qa" else "Note"))
        self.setMinimumWidth(480)
        form = QFormLayout()
        self.title = QLineEdit(title)
        self.title.setAccessibleName("Title")
        self.tags = QLineEdit(tags)
        self.tags.setPlaceholderText("comma separated, optional")
        self.tags.setAccessibleName("Tags")
        if kind == "qa":
            self.question = QLineEdit(question)
            self.question.setAccessibleName("Question")
            self.answer = QPlainTextEdit(answer)
            self.answer.setAccessibleName("Answer")
            self.answer.setMinimumHeight(130)
            form.addRow("Question", self.question)
            form.addRow("Answer you want given", self.answer)
            form.addRow("Title (optional)", self.title)
            form.addRow("", label("Q&A pairs get the highest retrieval priority; answers grounded in them are marked "
                                  "“Knowledge base”.", "caption", wrap=True))
        else:
            self.body = QPlainTextEdit(body)
            self.body.setAccessibleName("Text")
            self.body.setMinimumHeight(180)
            form.addRow("Title", self.title)
            form.addRow("Text", self.body)
        form.addRow("Tags", self.tags)
        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(box)

    def values(self) -> dict:
        if self.kind == "qa":
            return {"title": self.title.text(), "question": self.question.text(), "answer": self.answer.toPlainText(),
                    "tags": self.tags.text()}
        return {"title": self.title.text(), "body": self.body.toPlainText(), "tags": self.tags.text()}


class KnowledgePage(ActionPage):
    def __init__(self, engine, settings, parent=None):
        super().__init__(parent)
        self.engine, self.s = engine, settings
        self.kb = getattr(engine, "kb", None)
        self._indexing: set[str] = set()
        self.setAcceptDrops(True)
        self.setObjectName("KnowledgePage")
        lay = QVBoxLayout(self)
        lay.setSpacing(theme.SPACE["md"])

        # --- folder row
        self.folder = QLineEdit(str(settings.get("knowledge.folder", "")))
        self.folder.setPlaceholderText("Folder with your PDFs, DOCX, TXT, MD (optional)")
        self.folder.setAccessibleName("Knowledge folder")
        self.btn_browse = button("Select folder…", "secondary", "folder")
        self.btn_browse.clicked.connect(lambda: self.pick())
        self.chk_sync = QCheckBox("Check for new files at start")
        self.chk_sync.setChecked(bool(settings.get("knowledge.auto_sync", True)))
        self.chk_sync.toggled.connect(lambda v: self.s.set("knowledge.auto_sync", bool(v)))
        self.chk_watch = QCheckBox("Watch folders while running")
        self.chk_watch.setChecked(bool(settings.get("knowledge.watch", False)))
        self.chk_watch.toggled.connect(lambda v: self.s.set("knowledge.watch", bool(v)))
        lay.addLayout(hbox(self.folder, self.btn_browse, stretch_end=False))
        lay.addLayout(hbox(self.chk_sync, self.chk_watch, stretch_end=True))

        # --- actions
        self.btn_add = button("Add files…", "primary", "plus")
        self.btn_add_folder = button("Add folder…", "secondary", "folder_plus")
        self.btn_note = button("New note", "secondary", "note")
        self.btn_qa = button("New Q&A", "secondary", "message")
        self.btn_index = button("Index", "secondary", "refresh", "Index new and changed files in the folders")
        self.btn_re = button("Re-index all", "ghost", "refresh")
        self.btn_clear = button("Clear index", "danger", "trash")
        self.btn_add.clicked.connect(lambda: self.add_files())
        self.btn_add_folder.clicked.connect(lambda: self.add_folder())
        self.btn_note.clicked.connect(lambda: self.new_entry("note"))
        self.btn_qa.clicked.connect(lambda: self.new_entry("qa"))
        self.btn_index.clicked.connect(lambda: self.index(False))
        self.btn_re.clicked.connect(lambda: self.index(True))
        self.btn_clear.clicked.connect(lambda: self.clear())
        lay.addLayout(hbox(self.btn_add, self.btn_add_folder, self.btn_note, self.btn_qa, self.btn_index, self.btn_re,
                           self.btn_clear, stretch_end=True))
        self.layout_head(lay)
        self.stats_label = label("", "caption", wrap=True)
        lay.addWidget(self.stats_label)
        lay.addWidget(self.status)

        # --- table + detail
        self.search = SearchBox("Search documents…")
        self.search.textChanged.connect(lambda _t: self._apply_filter())
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.setAccessibleName("Documents")
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _i: self.edit_or_preview())
        self.row_buttons = {
            "preview": button("Preview", "ghost", "eye"), "reindex": button("Re-index", "ghost", "refresh"),
            "rename": button("Rename", "ghost", "edit"), "toggle": button("Disable", "ghost", "pause"),
            "open": button("Open location", "ghost", "external"), "delete": button("Delete", "danger", "trash"),
        }
        self.row_buttons["preview"].clicked.connect(lambda: self.preview_selected())
        self.row_buttons["reindex"].clicked.connect(lambda: self.reindex_selected())
        self.row_buttons["rename"].clicked.connect(lambda: self.rename_selected())
        self.row_buttons["toggle"].clicked.connect(lambda: self.toggle_selected())
        self.row_buttons["open"].clicked.connect(lambda: self.open_location())
        self.row_buttons["delete"].clicked.connect(lambda: self.delete_selected())
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.search)
        self.empty = EmptyState("database", "Your knowledge base is empty",
                                "Add PDFs, notes or Q&A pairs. Answers about them will cite the source.", "Add your first document")
        self.empty.action.connect(lambda: self.add_files())
        ll.addWidget(self.empty)
        ll.addWidget(self.table, 1)
        ll.addLayout(hbox(*self.row_buttons.values(), stretch_end=True))

        self.detail = QTabWidget()
        self.preview = QTextBrowser()
        self.preview.setAccessibleName("Document preview")
        self.chunk_list = QListWidget()
        self.chunk_list.setAccessibleName("Chunks")
        self.chunk_list.itemChanged.connect(self._chunk_toggled)
        chunk_tab = QWidget()
        cl = QVBoxLayout(chunk_tab)
        self.chunk_search = SearchBox("Search chunks…")
        self.chunk_search.textChanged.connect(lambda _t: self._filter_chunks())
        cl.addWidget(self.chunk_search)
        cl.addWidget(label("Untick a chunk to exclude it from answers.", "caption"))
        cl.addWidget(self.chunk_list, 1)
        play = QWidget()
        pl = QVBoxLayout(play)
        self.play_query = QLineEdit()
        self.play_query.setPlaceholderText("Type a question to see what would be retrieved…")
        self.play_query.setAccessibleName("Search playground query")
        self.play_query.returnPressed.connect(lambda: self.run_playground())
        self.btn_play = button("Search", "primary", "search")
        self.btn_play.clicked.connect(lambda: self.run_playground())
        self.play_out = QTextBrowser()
        self.play_out.setAccessibleName("Search results")
        pl.addLayout(hbox(self.play_query, self.btn_play))
        pl.addWidget(self.play_out, 1)
        self.view = QTextBrowser()
        self.view.setAccessibleName("Index summary")
        self.detail.addTab(self.preview, "Preview")
        self.detail.addTab(chunk_tab, "Chunks")
        self.detail.addTab(play, "Search playground")
        self.detail.addTab(self.view, "Summary")
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.detail)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)

        if self.kb is None:
            self.status.setText("The knowledge base is unavailable (no database).")
            for b in (self.btn_add, self.btn_add_folder, self.btn_note, self.btn_qa, self.btn_index, self.btn_re,
                      self.btn_clear, self.btn_browse):
                b.setEnabled(False)
        self.refresh()

    # ------------------------------------------------------------------------------------------ data → view
    def _needs_reindex(self, d) -> bool:
        if d.source in ("note", "qa") or d.status == "error":
            return False
        try:
            st = Path(d.path).stat()
        except OSError:
            return False
        return st.st_size != d.size or st.st_mtime > d.indexed_at + 1

    def _status_of(self, d) -> tuple[str, str]:
        if d.path in self._indexing:
            return "Indexing…", "info"
        if d.status == "error":
            return "Error", "danger"
        if d.status == "no_text":
            return "No text found", "warning"
        if self._needs_reindex(d):
            return "Needs re-index", "warning"
        return ("Indexed" if d.enabled else "Indexed · disabled"), ("success" if d.enabled else "text_faint")

    def refresh(self) -> None:
        assert_gui_thread("KnowledgePage.refresh")
        if self.kb is None:
            return
        sel = set(self.selected_paths())
        docs = self.kb.documents()
        st = self.kb.stats()
        t = theme.current()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(docs))
        for r, d in enumerate(docs):
            kind = _TYPE.get(d.source) or (Path(d.path).suffix.lstrip(".").upper() or "File")
            status, tone = self._status_of(d)
            cells = [QTableWidgetItem(d.name), QTableWidgetItem(kind), _Num(d.pages), _Num(d.chunk_count),
                     _Num(d.size, fmt_size(d.size)), QTableWidgetItem(status),
                     QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M", time.localtime(d.indexed_at)) if d.indexed_at else "—"),
                     QTableWidgetItem(_SOURCE.get(d.source, d.source))]
            cells[5].setForeground(QColor(t[tone]))
            if d.error:
                cells[5].setToolTip(d.error)
            if not d.enabled:
                for c in cells:
                    c.setForeground(QColor(t["text_faint"]))
            cells[0].setData(Qt.UserRole, d.path)
            cells[0].setToolTip(d.path)
            for c, it in enumerate(cells):
                self.table.setItem(r, c, it)
            if d.path in sel:
                self.table.selectRow(r)
        self.table.setSortingEnabled(True)
        self.empty.setVisible(not docs)
        self.table.setVisible(bool(docs))
        self.stats_label.setText(
            f"{st['documents']} document(s) · {st['active_chunks']}/{st['chunks']} active chunks · "
            f"{fmt_size(st['index_bytes'])} index · embeddings: {st['embedder']}"
            + (f" · last sync {st['last_sync']}" if st.get("last_sync") else ""))
        lines = [f"{st['documents']} document(s), {st['chunks']} chunk(s) — embeddings: {st['embedder']}"]
        lines += [f"  • {d['name']}: {d['pages']} p, {d['chunks']} chunks [{d['status']}]"
                  + (" (disabled)" if not d["enabled"] else "") + (f" {d['error']}" if d["error"] else "")
                  for d in st["per_document"]]
        self.view.setPlainText("\n".join(lines))
        self._apply_filter()
        self._selection_changed()

    def _apply_filter(self) -> None:
        q = self.search.text().strip().lower()
        for r in range(self.table.rowCount()):
            hay = " ".join((self.table.item(r, c).text() if self.table.item(r, c) else "") for c in (0, 1, 5, 7)).lower()
            self.table.setRowHidden(r, bool(q) and q not in hay)

    def selected_paths(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.table.item(r, 0).data(Qt.UserRole) for r in rows if self.table.item(r, 0)]

    def _row_of(self, path: str) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).data(Qt.UserRole) == path:
                return r
        return -1

    def select_path(self, path: str) -> None:
        r = self._row_of(path)
        if r >= 0:
            self.table.selectRow(r)

    def _selection_changed(self) -> None:
        paths = self.selected_paths()
        one = len(paths) == 1
        for k, b in self.row_buttons.items():
            b.setEnabled(bool(paths) and not self.busy() if k in ("delete", "reindex", "toggle") else one)
        if one and self.kb:
            d = self.kb.get_document(paths[0])
            if d:
                self.row_buttons["toggle"].setText("Enable" if not d.enabled else "Disable")
                self.preview.setPlainText(self.kb.preview(paths[0]))
                self._fill_chunks(paths[0])
                self.row_buttons["open"].setEnabled(d.source not in ("note", "qa"))
                self.row_buttons["rename"].setEnabled(True)
        else:
            self.chunk_list.clear()
            self.preview.setPlainText("Select a document to preview it." if not paths else f"{len(paths)} documents selected.")

    def _fill_chunks(self, path: str) -> None:
        self.chunk_list.blockSignals(True)
        self.chunk_list.clear()
        for c in self.kb.chunks_of(path):
            it = QListWidgetItem(f"p.{c.page}" + (f" · {c.section}" if c.section else "") + f" — {c.text[:140]}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c.enabled else Qt.Unchecked)
            it.setData(Qt.UserRole, c.chunk_id)
            it.setToolTip(c.text[:600])
            self.chunk_list.addItem(it)
        self.chunk_list.blockSignals(False)
        self._filter_chunks()

    def _filter_chunks(self) -> None:
        q = self.chunk_search.text().strip().lower()
        for i in range(self.chunk_list.count()):
            it = self.chunk_list.item(i)
            it.setHidden(bool(q) and q not in it.text().lower() and q not in (it.toolTip() or "").lower())

    def _chunk_toggled(self, it) -> None:
        self.set_chunk_enabled(it.data(Qt.UserRole), it.checkState() == Qt.Checked)

    def set_chunk_enabled(self, chunk_id: str, enabled: bool) -> None:
        self.kb.set_chunk_enabled(chunk_id, enabled)
        self.stats_label.setText(self.stats_label.text())          # stats refresh on next refresh()

    # ------------------------------------------------------------------------------------------ busy state
    def set_busy_state(self, busy: bool) -> None:
        for b in (self.btn_add, self.btn_add_folder, self.btn_index, self.btn_re, self.btn_clear, self.btn_browse):
            b.setEnabled(not busy and self.kb is not None)
        self._selection_changed()

    def after_task(self) -> None:
        self._indexing.clear()
        self.refresh()

    # ------------------------------------------------------------------------------------------ adding
    def _folders(self) -> list[str]:
        out = []
        for f in [self.folder.text().strip(), *list(self.s.get("knowledge.extra_folders", []) or [])]:
            if f and f not in out:
                out.append(f)
        return out

    def pick(self, directory: str = "") -> None:
        d = directory or QFileDialog.getExistingDirectory(self, "Select folder with documents", self.folder.text())
        if d:
            self.folder.setText(d)
            self.s.set("knowledge.folder", d)

    def _report_done(self, rep, err) -> None:
        if err is not None:
            if getattr(err, "cancelled", False):
                self.status.setText("Cancelled — finished documents were kept.")
            else:
                self.status.setText("Indexing failed: " + (getattr(err, "user_message", None) or str(err)))
            return
        self.status.setText(rep.summary())
        n = rep.added + rep.updated
        kind = "warn" if (rep.failed or rep.no_text) else "success"
        self.toast(kind, f"{n} document(s) indexed" + (f", {len(rep.failed)} failed" if rep.failed else "") + ".")

    def add_files(self, paths=None) -> None:
        if self.kb is None:
            return
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, "Add documents", "", "Documents (*.pdf *.docx *.txt *.md)")
        paths = [str(p) for p in (paths or [])]
        if not paths:
            return
        self._indexing.update(str(Path(p)) for p in paths)
        self.refresh()
        self.start(lambda ctx: self.kb.add_files(paths, progress_cb=ctx.progress_cb, cancel=ctx), self._report_done,
                   f"Indexing {len(paths)} file(s)", "kb")

    def add_folder(self, directory: str = "") -> None:
        d = directory or QFileDialog.getExistingDirectory(self, "Add a folder of documents", self.folder.text())
        if not d:
            return
        if not self.folder.text().strip():
            self.pick(d)
        else:
            extra = list(self.s.get("knowledge.extra_folders", []) or [])
            if d != self.folder.text().strip() and d not in extra:
                extra.append(d)
                self.s.set("knowledge.extra_folders", extra)
        self.index(False)

    def index(self, force: bool) -> None:
        if self.kb is None:
            return
        folders = self._folders()
        if not folders:
            self.status.setText("Select a folder first.")
            return
        if self.folder.text().strip():
            self.s.set("knowledge.folder", self.folder.text().strip())
        self.status.setText("Indexing… (unchanged files are skipped)")
        self.start(lambda ctx: self.kb.sync_folders(folders, progress_cb=ctx.progress_cb, force=force, cancel=ctx),
                   self._report_done, "Indexing folders", "kb")

    def new_entry(self, kind: str) -> str | None:
        dlg = EntryDialog(kind, self)
        if dlg.exec() != QDialog.Accepted:
            return None
        return self.create_entry(kind, **dlg.values())

    def create_entry(self, kind: str, **v) -> str | None:
        try:
            eid = self.kb.add_qa(v.get("question", ""), v.get("answer", ""), v.get("tags", ""), v.get("title", "")) \
                if kind == "qa" else self.kb.add_note(v.get("title", ""), v.get("body", ""), v.get("tags", ""))
        except Exception as e:
            self.fail(e)
            return None
        self.toast("success", "Q&A pair added." if kind == "qa" else "Note added.")
        self.refresh()
        self.select_path((QA_SCHEME if kind == "qa" else NOTE_SCHEME) + eid)
        return eid

    # ------------------------------------------------------------------------------------------ per-document actions
    def edit_or_preview(self) -> None:
        paths = self.selected_paths()
        if len(paths) != 1:
            return
        e = self.kb.entry_for_path(paths[0])
        if e is None:
            self.detail.setCurrentIndex(0)
            return
        dlg = EntryDialog(e.kind, self, e.title, e.body, e.question, e.answer, e.tags)
        if dlg.exec() == QDialog.Accepted:
            self.update_entry(e.id, **dlg.values())

    def update_entry(self, entry_id: str, **fields) -> bool:
        try:
            changed = self.kb.update_entry(entry_id, **fields)
        except Exception as e:
            self.fail(e)
            return False
        if changed:
            self.toast("success", "Saved — only the changed parts were re-embedded.")
        self.refresh()
        return changed

    def preview_selected(self) -> None:
        self.detail.setCurrentIndex(0)

    def reindex_selected(self, paths=None) -> None:
        paths = paths or self.selected_paths()
        if not paths:
            return
        self._indexing.update(paths)
        self.refresh()

        def work(ctx):
            res = []
            for i, p in enumerate(paths, 1):
                ctx.check()
                ctx.progress((i - 1) / len(paths), Path(p).name)
                try:
                    res.append(self.kb.reindex(p, cancel=ctx))
                except Exception as e:
                    if getattr(e, "cancelled", False):
                        raise
                    res.append(f"{Path(p).name}: {getattr(e, 'user_message', e)}")
            return res

        def done(res, err):
            if err is None:
                bad = [r for r in res if r not in ("added", "updated", "unchanged")]
                self.toast("warn" if bad else "success", bad[0] if bad else f"{len(res)} document(s) re-indexed.")
        self.start(work, done, f"Re-indexing {len(paths)} document(s)", "kb")

    def rename_selected(self, name: str | None = None) -> None:
        paths = self.selected_paths()
        if len(paths) != 1:
            return
        d = self.kb.get_document(paths[0])
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename document", "New name:", text=d.name)
            if not ok:
                return
        try:
            self.kb.rename(paths[0], name)
        except Exception as e:
            self.fail(e)
            return
        self.refresh()

    def toggle_selected(self, enabled: bool | None = None) -> None:
        for p in self.selected_paths():
            d = self.kb.get_document(p)
            if d:
                self.kb.set_enabled(p, (not d.enabled) if enabled is None else enabled)
        self.refresh()

    def open_location(self) -> None:
        paths = self.selected_paths()
        if len(paths) == 1 and "://" not in paths[0]:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(paths[0]).parent)))

    def delete_selected(self, paths=None, confirm: bool = True) -> None:
        paths = paths or self.selected_paths()
        if not paths:
            return
        if confirm and len(paths) > 1 and QMessageBox.question(self, "Delete documents", f"Delete {len(paths)} documents?") != QMessageBox.Yes:
            return
        done = []
        for p in paths:
            try:
                done.append(self.kb.delete_document(p))
            except Exception as e:
                self.fail(e)
        if done:
            self.toast("info", f"{len(done)} document(s) deleted.", "Undo", lambda ps=tuple(done): self.undo(ps))
        self.refresh()

    def undo(self, paths) -> None:
        ok = [p for p in paths if self.kb.undo_delete(p)]
        if ok:
            self.toast("success", f"{len(ok)} document(s) restored.")
        else:
            self.toast("warn", "Nothing could be restored — the undo window has passed.")
        self.refresh()

    def clear(self, confirm: bool = True) -> None:
        if confirm and QMessageBox.question(self, "Clear index", "Remove every document from the knowledge base?") != QMessageBox.Yes:
            return
        self.kb.clear()
        self.status.setText("Index cleared.")
        self.refresh()

    def _context_menu(self, pos) -> None:
        if not self.selected_paths():
            return
        m = QMenu(self)
        for text, fn in (("Preview", self.preview_selected), ("Re-index", self.reindex_selected), ("Rename…", self.rename_selected),
                         ("Disable / Enable", self.toggle_selected), ("Open file location", self.open_location),
                         ("Delete", self.delete_selected)):
            a = QAction(text, m)
            a.triggered.connect(lambda _c=False, f=fn: f())
            m.addAction(a)
        m.exec(self.table.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------------------------------ playground
    def run_playground(self, query: str | None = None) -> list:
        q = (query if query is not None else self.play_query.text()).strip()
        if not q or self.kb is None:
            return []
        hits = self.kb.search_debug(q, top_k=8)
        if not hits:
            self.play_out.setPlainText("No matching chunks. Add documents or try different words.")
            return []
        out = []
        for i, h in enumerate(hits, 1):
            out.append(f"#{i}  {h.label}   fused {h.score:.3f} · BM25 {h.bm25:.3f}"
                       + (f" (rank {h.bm25_rank})" if h.bm25_rank else "") + f" · vector {h.cosine:.3f}"
                       + (f" (rank {h.vec_rank})" if h.vec_rank else "") + f"\n   {h.text[:260].strip()}")
        self.play_out.setPlainText("\n\n".join(out))
        return hits

    # ------------------------------------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        files, folders = [], []
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if Path(f).is_dir():
                folders.append(f)
            elif f.lower().endswith(ACCEPT):
                files.append(f)
        if files:
            self.add_files(files)
        for d in folders[:1]:
            self.add_folder(d)
