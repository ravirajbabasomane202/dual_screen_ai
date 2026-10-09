"""Resume page: structured view, edit parsed fields, parser warnings, version history, several named profiles."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QFrame, QInputDialog, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QScrollArea, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from display import theme
from display.pages.base import ActionPage
from display.widgets.common import Chip, FlowLayout, badge, button, card, hbox, label, set_badge
from display.widgets.feedback import EmptyState
from resume.editor import (collect_edits, format_education, format_experience, format_lines, format_list, format_projects)

ACCEPT = (".pdf", ".docx", ".txt", ".md")
_LIST_FIELDS = ("skills", "programming_languages", "frameworks", "tools")
_LINE_FIELDS = ("certifications", "achievements")


def _ago(ts) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


class ResumePage(ActionPage):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.mgr = getattr(engine, "resume", None)
        self.setAcceptDrops(True)
        self.setObjectName("ResumePage")
        lay = QVBoxLayout(self)
        lay.setSpacing(theme.SPACE["md"])

        # --- toolbar
        self.profile_combo = QComboBox()
        self.profile_combo.setAccessibleName("Resume profile")
        self.profile_combo.setMinimumWidth(200)
        self.profile_combo.activated.connect(self._on_profile_chosen)
        self.buttons = {
            "up": button("Upload / Replace…", "primary", "upload", "Replace the active resume with a new file"),
            "add": button("Add profile…", "secondary", "plus", "Add another resume as a separate profile"),
            "re": button("Re-parse", "secondary", "refresh", "Parse the stored file again with the current parser"),
            "ren": button("Rename", "ghost", "edit"),
            "act": button("Set active", "ghost", "check"),
            "del": button("Delete", "danger", "trash"),
        }
        self.buttons["up"].clicked.connect(lambda: self.upload())
        self.buttons["add"].clicked.connect(lambda: self.upload(new_profile=True))
        self.buttons["re"].clicked.connect(lambda: self.reparse())
        self.buttons["ren"].clicked.connect(lambda: self.rename())
        self.buttons["act"].clicked.connect(lambda: self.set_active())
        self.buttons["del"].clicked.connect(lambda: self.delete())
        lay.addLayout(hbox(label("Profile", "caption"), self.profile_combo, *self.buttons.values(), stretch_end=True))
        self.layout_head(lay)
        lay.addWidget(self.status)

        # --- tabs
        self.tabs = QTabWidget()
        self.overview = QScrollArea()
        self.overview.setWidgetResizable(True)
        self.overview.setFrameShape(QFrame.NoFrame)
        self.empty = EmptyState("document", "No resume yet",
                                "Upload a PDF, DOCX or TXT (or drop a file here). Answers about your background will be "
                                "grounded in it.", "Upload your resume")
        self.empty.action.connect(lambda: self.upload())
        self.tabs.addTab(self.overview, "Overview")
        self.editor = self._build_editor()
        self.tabs.addTab(self.editor, "Edit parsed fields")
        self.warn_list = QListWidget()
        self.warn_list.setAccessibleName("Parser warnings")
        self.tabs.addTab(self.warn_list, "Parser warnings")
        self.history = self._build_history()
        self.tabs.addTab(self.history, "History")
        self.view = QTextBrowser()
        self.view.setAccessibleName("Resume as text")
        self.tabs.addTab(self.view, "Text")
        lay.addWidget(self.empty)
        lay.addWidget(self.tabs, 1)

        if self.mgr is None:
            self.status.setText("Resume storage is unavailable (no database).")
            for b in self.buttons.values():
                b.setEnabled(False)
        elif hasattr(self.mgr, "on_change"):
            self.mgr.on_change(self._external_change)       # always delivered on the GUI thread
        self.refresh()

    # ------------------------------------------------------------------------------------------ builders
    def _build_editor(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        note = label("Corrections you save here are kept when the resume is parsed again (unless you discard them). "
                     "Lists: one item per line or comma separated. Experience / projects: blank line between entries; "
                     "first line “Title — Company (start – end)”, bullets start with “-”.", "caption", wrap=True)
        lay.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        form = QFormLayout(inner)
        self.fields: dict[str, QWidget] = {}
        self.fields["name"] = QLineEdit()
        form.addRow("Name", self.fields["name"])
        for key, title, h in (("summary", "Summary", 70), ("skills", "Skills", 60), ("programming_languages", "Languages", 44),
                              ("frameworks", "Frameworks", 44), ("tools", "Tools", 44), ("experience", "Experience", 150),
                              ("projects", "Projects", 110), ("education", "Education", 70),
                              ("certifications", "Certifications", 50), ("achievements", "Achievements", 50)):
            e = QPlainTextEdit()
            e.setMinimumHeight(h)
            e.setAccessibleName(title)
            self.fields[key] = e
            form.addRow(title, e)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        self.btn_save_edits = button("Save corrections", "primary", "check")
        self.btn_discard_edits = button("Discard all corrections", "ghost", "undo")
        self.btn_save_edits.clicked.connect(lambda: self.save_edits())
        self.btn_discard_edits.clicked.connect(lambda: self.discard_edits())
        self.edit_note = label("", "caption")
        lay.addLayout(hbox(self.btn_save_edits, self.btn_discard_edits, self.edit_note, stretch_end=True))
        return w

    def _build_history(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Earlier versions of this profile are kept (last 15). Restoring is itself undoable.", "caption", wrap=True))
        self.history_list = QListWidget()
        self.history_list.setAccessibleName("Resume versions")
        lay.addWidget(self.history_list, 1)
        self.btn_restore = button("Restore selected version", "secondary", "undo")
        self.btn_restore.clicked.connect(lambda: self.restore_selected())
        lay.addLayout(hbox(self.btn_restore, stretch_end=True))
        return w

    # ------------------------------------------------------------------------------------------ refresh
    def _external_change(self, _cur) -> None:
        self.refresh()

    def refresh(self) -> None:
        assert_gui_thread("ResumePage.refresh")
        if self.mgr is None:
            return
        cur = self.mgr.get()
        profiles = self.mgr.profiles()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for p in profiles:
            self.profile_combo.addItem(("● " if p.active else "") + (p.label or p.filename), p.id)
            if p.active:
                self.profile_combo.setCurrentIndex(self.profile_combo.count() - 1)
        self.profile_combo.blockSignals(False)
        self.view.setPlainText(self.mgr.view_text())
        has = cur is not None
        for k in ("re", "ren", "del"):
            self.buttons[k].setEnabled(has and not self.busy())
        self.buttons["act"].setEnabled(has and len(profiles) > 1 and not self.busy())
        self.empty.setVisible(not has)
        self.tabs.setVisible(has)
        self._fill_overview(cur)
        self._fill_editor(cur)
        self._fill_warnings(cur)
        self._fill_history(cur)
        if cur:
            self.status.setText(f"Loaded: {cur.filename}" + (" · edited by you" if cur.edited else "")
                                + f" · parsed {_ago(cur.parsed_at)} · {len(profiles)} profile(s)")
        elif not self.busy():
            self.status.setText("No resume uploaded.")

    def set_busy_state(self, busy: bool) -> None:
        for k in ("up", "add"):
            self.buttons[k].setEnabled(not busy and self.mgr is not None)
        if busy:
            for k in ("re", "ren", "act", "del"):
                self.buttons[k].setEnabled(False)

    def after_task(self) -> None:
        self.refresh()

    def _fill_overview(self, cur) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(theme.SPACE["md"])
        if cur:
            p = cur.parsed
            head, hl = card()
            hl.addWidget(label(p.name or "(name not found — use Edit parsed fields)", "title"))
            meta = [f"{p.years_of_experience:g} years" if p.years_of_experience else "", cur.filename]
            hl.addWidget(label(" · ".join(m for m in meta if m), "caption"))
            if p.summary:
                hl.addWidget(label(p.summary, "", wrap=True))
            lay.addWidget(head)
            if p.skills:
                lay.addWidget(label("Skills", "section"))
                host = QWidget()
                fl = FlowLayout(host)
                for s in p.skills:
                    fl.addWidget(Chip(s, clickable=False))
                lay.addWidget(host)
            if p.experience:
                lay.addWidget(label("Experience", "section"))
                for e in p.experience:
                    c, cl = card()
                    cl.addWidget(label(f"{e.get('title') or '?'} — {e.get('company') or '?'}", "strong"))
                    cl.addWidget(label(f"{e.get('start', '')} – {e.get('end', '')}", "caption"))
                    for b in e.get("bullets", [])[:6]:
                        cl.addWidget(label("• " + b, "", wrap=True))
                    lay.addWidget(c)
            if p.projects:
                lay.addWidget(label("Projects", "section"))
                for pr in p.projects:
                    c, cl = card()
                    cl.addWidget(label(pr.get("name", ""), "strong"))
                    if pr.get("description"):
                        cl.addWidget(label(pr["description"], "", wrap=True))
                    if pr.get("technologies"):
                        cl.addWidget(label("Technologies: " + ", ".join(pr["technologies"]), "caption", wrap=True))
                    lay.addWidget(c)
            if p.education:
                lay.addWidget(label("Education", "section"))
                for e in p.education:
                    lay.addWidget(label(f"{e.get('degree', '')} — {e.get('institution', '')} {e.get('dates', '')}".strip(), "", wrap=True))
            if p.certifications:
                lay.addWidget(label("Certifications", "section"))
                lay.addWidget(label(", ".join(p.certifications), "", wrap=True))
        lay.addStretch()
        self.overview.setWidget(w)

    def _fill_editor(self, cur) -> None:
        if not cur:
            for e in self.fields.values():
                (e.setText if isinstance(e, QLineEdit) else e.setPlainText)("")
            return
        p = cur.parsed
        vals = {"name": p.name, "summary": p.summary, "experience": format_experience(p.experience),
                "projects": format_projects(p.projects), "education": format_education(p.education)}
        for k in _LIST_FIELDS:
            vals[k] = format_list(getattr(p, k))
        for k in _LINE_FIELDS:
            vals[k] = format_lines(getattr(p, k))
        for k, e in self.fields.items():
            (e.setText if isinstance(e, QLineEdit) else e.setPlainText)(vals.get(k, ""))
        self.edit_note.setText("Edited fields: " + ", ".join(sorted(cur.edits)) if cur.edits else "No manual corrections.")
        self.btn_discard_edits.setEnabled(bool(cur.edits))

    def _fill_warnings(self, cur) -> None:
        self.warn_list.clear()
        warns = list(cur.parsed.warnings) if cur else []
        for wmsg in warns:
            self.warn_list.addItem(QListWidgetItem("⚠ " + wmsg))
        if not warns:
            self.warn_list.addItem(QListWidgetItem("No parser warnings — all main sections were recognised."))
        i = self.tabs.indexOf(self.warn_list)
        self.tabs.setTabText(i, f"Parser warnings ({len(warns)})" if warns else "Parser warnings")

    def _fill_history(self, cur) -> None:
        self.history_list.clear()
        vs = self.mgr.versions() if (self.mgr and cur) else []
        for v in vs:
            it = QListWidgetItem(f"{_ago(v.created_at)} · {v.reason} · {v.filename}")
            it.setData(Qt.UserRole, v.id)
            self.history_list.addItem(it)
        self.btn_restore.setEnabled(bool(vs))

    # ------------------------------------------------------------------------------------------ actions
    def _ask(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _pick_file(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Select resume", "", "Resumes (*.pdf *.docx *.txt *.md)")
        return path

    def upload(self, path: str = "", new_profile: bool = False) -> None:
        assert_gui_thread("ResumePage.upload")
        if self.mgr is None:
            return
        path = path or self._pick_file()
        if not path:
            return
        name = Path(path).name

        def work(ctx):
            return self.mgr.upload(path, new_profile=new_profile, cancel=ctx,
                                   progress=lambda d, t: ctx.progress(d / t if t else None, f"page {d} of {t}"))

        def done(value, err):
            if err is None:
                self.toast("success", f"Resume “{name}” parsed and saved.")
                self.banner.hide()
            elif getattr(err, "cancelled", False):
                self.status.setText("Cancelled — previous resume kept.")
            else:
                self.status.setText("Failed — previous resume kept.")
        self.start(work, done, f"Parsing {name}", "resume")

    def reparse(self, keep_edits: bool | None = None) -> None:
        cur = self.mgr.get() if self.mgr else None
        if cur is None:
            return
        if keep_edits is None:
            keep_edits = True
            if cur.edited and not self._ask("Re-parse", "You have manual corrections. Keep them after re-parsing?\n\n"
                                                         "Yes = keep, No = discard them."):
                keep_edits = False
        self.start(lambda ctx: self.mgr.reparse(cur.id, keep_edits=keep_edits, cancel=ctx),
                   lambda v, e: self.toast("success", "Resume re-parsed.") if e is None else None,
                   "Re-parsing resume", "resume")

    def delete(self, confirm: bool = True) -> None:
        cur = self.mgr.get() if self.mgr else None
        if cur is None:
            return
        if confirm and not self._ask("Delete resume", f"Delete “{cur.label or cur.filename}” and its parsed data?\n"
                                                      "Earlier versions are deleted with it."):
            return
        self.mgr.delete(cur.id)
        self.toast("info", "Resume deleted.")
        self.refresh()

    def rename(self, label_text: str | None = None) -> None:
        cur = self.mgr.get() if self.mgr else None
        if cur is None:
            return
        if label_text is None:
            label_text, ok = QInputDialog.getText(self, "Rename profile", "Profile name:", text=cur.label or cur.filename)
            if not ok:
                return
        self.mgr.rename(cur.id, label_text)
        self.refresh()

    def set_active(self, profile_id: int | None = None) -> None:
        pid = profile_id if profile_id is not None else self.profile_combo.currentData()
        if pid is None:
            return
        self.mgr.activate(int(pid))
        self.toast("success", "Active resume profile changed.")
        self.refresh()

    def _on_profile_chosen(self, _i: int) -> None:
        pid = self.profile_combo.currentData()
        if pid is not None:
            self.mgr.activate(int(pid))
            self.refresh()

    def save_edits(self) -> None:
        cur = self.mgr.get() if self.mgr else None
        if cur is None:
            return
        fields = {k: (e.text() if isinstance(e, QLineEdit) else e.toPlainText()) for k, e in self.fields.items()}
        try:
            base = cur.parsed_base or cur.parsed
            changed = collect_edits(base, fields)
            revert = [k for k in cur.edits if k not in changed]
            if revert:
                self.mgr.reset_fields(revert, cur.id)
            if changed:
                self.mgr.edit_fields(changed, cur.id)
        except Exception as e:                      # surfaced, never swallowed
            self.fail(e)
            return
        self.toast("success", "Corrections saved." if changed else "No differences from the parsed values.")
        self.refresh()

    def discard_edits(self) -> None:
        if self.mgr and self.mgr.get() and self._ask("Discard corrections", "Go back to what the parser found?"):
            self.mgr.reset_fields(None)
            self.refresh()

    def restore_selected(self, version_id: int | None = None) -> None:
        if version_id is None:
            it = self.history_list.currentItem()
            version_id = it.data(Qt.UserRole) if it else None
        if version_id is None:
            return
        try:
            self.mgr.restore_version(int(version_id))
        except Exception as e:
            self.fail(e)
            return
        self.toast("success", "Version restored.")
        self.refresh()

    # ------------------------------------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(ACCEPT) for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if f.lower().endswith(ACCEPT):
                self.upload(f)
                break
