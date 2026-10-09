"""Custom pages of the Settings dialog. Resume and Knowledge Base live in display/pages/; the simpler action pages are
here. Long operations run through app.tasks (progress, cooperative cancel, GUI-thread delivery)."""
from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from pathlib import Path

from PySide6.QtWidgets import (QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from app.tasks import run_async
from display.pages.base import ActionPage
from display.widgets.common import button, hbox, label
from display.pages.knowledge_page import KnowledgePage          # noqa: F401  (re-exported)
from display.pages.resume_page import ResumePage                # noqa: F401  (re-exported)


def _err_text(e: Exception) -> str:
    return getattr(e, "user_message", None) or f"{type(e).__name__}: {e}"


def _row(*widgets) -> QHBoxLayout:
    lay = QHBoxLayout()
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch()
    return lay


class LLMTestPage(ActionPage):
    """Test the language model in one click; 'Pull model' downloads a missing Ollama model with a progress bar."""

    def __init__(self, engine, settings=None, parent=None):
        super().__init__(parent)
        self.engine, self.s = engine, settings
        lay = QVBoxLayout(self)
        self.btn = button("Test LLM", "primary", "play")
        self.btn.setEnabled(engine is not None)
        self.btn.clicked.connect(lambda: self.run())
        self.btn_pull = button("Pull model", "secondary", "download", "Download the selected model through Ollama")
        self.btn_pull.setEnabled(settings is not None)
        self.btn_pull.clicked.connect(lambda: self.pull())
        self.out = QTextBrowser()
        self.out.setMaximumHeight(170)
        self.out.setAccessibleName("LLM test result")
        lay.addLayout(hbox(self.btn, self.btn_pull, stretch_end=True))
        self.layout_head(lay)
        lay.addWidget(self.out)

    def set_busy_state(self, busy: bool) -> None:
        self.btn.setEnabled(not busy and self.engine is not None)
        self.btn_pull.setEnabled(not busy and self.s is not None)

    def run(self) -> None:
        self.out.setPlainText('Sending: "Answer this question in one sentence: What is Python?" …')

        def done(r, err):
            if err is None:
                self.out.setPlainText(r.format())
            else:
                self.out.setPlainText("Test crashed: " + _err_text(err))
        self.start(lambda ctx: self.engine.test_llm(), done, "Testing the language model", "llm", cancellable=False)

    def pull(self, model: str = "") -> None:
        from llm.model_manager import pull_ollama_model
        model = model or str(self.s.get("ai.model"))
        if model in ("", "auto"):
            self.out.setPlainText("Choose a specific model name in the Model box first (not “auto”).")
            return
        host = str(self.s.get("ai.host"))

        def work(ctx):
            def cb(pct, status):
                ctx.check()
                ctx.progress(pct / 100, f"{status} {pct}%")
            pull_ollama_model(host, model, cb)

        def done(_v, err):
            if err is None:
                self.toast("success", f"Model {model} is ready.")
                self.out.setPlainText(f"Model {model} downloaded.")
            elif getattr(err, "cancelled", False):
                self.out.setPlainText("Download cancelled.")
        self.start(work, done, f"Downloading {model}", "llm_model")


class MemoryPage(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        lay = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setAccessibleName("Interview memory")
        refresh, clear = button("Refresh", "secondary", "refresh"), button("Clear context", "danger", "trash")
        refresh.clicked.connect(lambda: self.refresh())
        clear.clicked.connect(lambda: self._clear())
        lay.addLayout(hbox(refresh, clear, stretch_end=True))
        lay.addWidget(self.view, 1)
        self.refresh()

    def refresh(self) -> None:
        if self.engine is None:
            return
        import json
        self.view.setPlainText(json.dumps(self.engine.context.snapshot(), indent=2, ensure_ascii=False))

    def _clear(self) -> None:
        if self.engine is not None:
            self.engine.clear_context()
            self.refresh()


class DebugPage(ActionPage):
    """Diagnostics (full self-test) plus log folder / redacted bundle export."""

    def __init__(self, engine, log_dir, parent=None, settings=None):
        super().__init__(parent)
        self.engine, self.log_dir, self.s = engine, log_dir, settings
        lay = QVBoxLayout(self)
        open_logs = button("Open logs folder", "secondary", "folder")
        export = button("Export diagnostics bundle", "secondary", "download", "A redacted zip of logs, activity and settings")
        self.diag = button("Run full diagnostics", "primary", "play")
        open_logs.clicked.connect(lambda: self.open_logs())
        export.clicked.connect(lambda: self.export_bundle())
        self.diag.clicked.connect(lambda: self.run())
        self.diag.setEnabled(engine is not None)
        self.out = QTextBrowser()
        self.out.setProperty("mono", True)
        self.out.setLineWrapMode(QTextBrowser.NoWrap)
        self.out.setAccessibleName("Diagnostics report")
        lay.addLayout(hbox(open_logs, export, self.diag, stretch_end=True))
        self.layout_head(lay)
        lay.addWidget(label(f"Logs: {log_dir}   (crash.log is written here if the app ever closes unexpectedly)", "caption", wrap=True))
        lay.addWidget(self.out, 1)

    def open_logs(self) -> None:
        if self.log_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_dir)))

    def export_bundle(self, dest: str = "") -> str:
        from app.bundle import export_diagnostics_bundle
        from app.status import get_bus
        dest = dest or QFileDialog.getSaveFileName(self, "Export diagnostics", "diagnostics.zip", "Zip (*.zip)")[0]
        if not dest:
            return ""
        try:
            names = export_diagnostics_bundle(dest, self.log_dir, get_bus(), self.s)
        except Exception as e:
            self.fail(e)
            return ""
        self.toast("success", f"Diagnostics bundle saved ({len(names)} files, personal text redacted).")
        return dest

    def run(self) -> None:
        self.out.setPlainText("Running diagnostics… (microphone, VAD, speech model and LLM checks can take a while)")

        def done(rep, err):
            self.out.setPlainText(("Diagnostics crashed: " + _err_text(err)) if err else rep.format())
        self.start(lambda ctx: self.engine.run_diagnostics(), done, "Running diagnostics", "diagnostics", cancellable=False)


class InterviewProfilePage(QWidget):
    """Company / role / pasted job description for this interview (used to tailor answers)."""

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.store = getattr(getattr(engine, "services", None), "profile_store", None)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.company, self.role = QLineEdit(), QLineEdit()
        self.company.setAccessibleName("Company")
        self.role.setAccessibleName("Role")
        self.jd = QPlainTextEdit()
        self.jd.setPlaceholderText("Paste the job description here (optional)")
        self.jd.setAccessibleName("Job description")
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(70)
        self.notes.setAccessibleName("Notes")
        form.addRow("Company", self.company)
        form.addRow("Role", self.role)
        form.addRow("Job description", self.jd)
        form.addRow("Notes", self.notes)
        lay.addLayout(form)
        self.status = label("", "caption")
        self.btn_save = button("Save interview profile", "primary", "check")
        self.btn_clear = button("Clear", "ghost", "trash")
        self.btn_save.clicked.connect(lambda: self.save())
        self.btn_clear.clicked.connect(lambda: self.clear())
        lay.addLayout(hbox(self.btn_save, self.btn_clear, self.status, stretch_end=True))
        if self.store is None:
            for w in (self.company, self.role, self.jd, self.notes, self.btn_save, self.btn_clear):
                w.setEnabled(False)
            self.status.setText("Unavailable (no database).")
        else:
            self.load()

    def load(self) -> None:
        p = self.store.load()
        self.company.setText(p.company)
        self.role.setText(p.role)
        self.jd.setPlainText(p.job_description)
        self.notes.setPlainText(p.notes)

    def save(self) -> None:
        from context.interview_profile import InterviewProfile
        self.store.save(InterviewProfile(self.company.text().strip(), self.role.text().strip(),
                                         self.jd.toPlainText().strip(), self.notes.toPlainText().strip()))
        if hasattr(self.engine, "apply_interview_profile"):
            self.engine.apply_interview_profile()
        self.status.setText("Saved — answers will be tailored to this interview.")

    def clear(self) -> None:
        self.store.delete()
        self.load()
        if hasattr(self.engine, "apply_interview_profile"):
            self.engine.apply_interview_profile()
        self.status.setText("Cleared.")


class PrivacyPage(ActionPage):
    """What is stored, backup / restore of everything, and 'Wipe all data'."""

    def __init__(self, engine, settings, db, log_dir=None, parent=None):
        super().__init__(parent)
        self.engine, self.s, self.db, self.log_dir = engine, settings, db, log_dir
        lay = QVBoxLayout(self)
        self.report = QTextBrowser()
        self.report.setAccessibleName("Stored data")
        self.report.setMaximumHeight(210)
        self.btn_export = button("Export everything…", "secondary", "download")
        self.btn_import = button("Import backup…", "secondary", "upload")
        self.btn_wipe = button("Wipe all data…", "danger", "trash")
        self.btn_export.clicked.connect(lambda: self.export_all())
        self.btn_import.clicked.connect(lambda: self.import_all())
        self.btn_wipe.clicked.connect(lambda: self.wipe())
        lay.addWidget(label("What this app stores on your computer", "section"))
        lay.addWidget(self.report)
        lay.addLayout(hbox(self.btn_export, self.btn_import, self.btn_wipe, stretch_end=True))
        lay.addWidget(self.status)
        self.layout_head(lay)
        lay.addStretch(1)
        if db is None:
            for b in (self.btn_export, self.btn_import, self.btn_wipe):
                b.setEnabled(False)
        self.refresh()

    def refresh(self) -> None:
        if self.db is None:
            self.report.setPlainText("No database is open.")
            return
        from app.backup import stored_data_report
        rows = stored_data_report(self.db, self.s, self.log_dir, getattr(self.engine, "services", None))
        self.report.setPlainText("\n".join(f"{'●' if on else '○'} {what}: {how}" + ("   (collecting)" if on else "") for what, how, on in rows))

    def _services(self):
        return getattr(self.engine, "services", None)

    def export_all(self, dest: str = "") -> dict | None:
        from app.backup import export_backup
        dest = dest or QFileDialog.getSaveFileName(self, "Export everything", "dual-screen-ai-backup.zip", "Zip (*.zip)")[0]
        if not dest:
            return None

        def done(c, err):
            if err is None:
                self.toast("success", f"Exported {c['resume_profiles']} resume profile(s), {c['kb_entries']} note/Q&A.")
        return self.start(lambda ctx: export_backup(dest, self.db, self.s, self._services()), done, "Exporting backup", "backup", False)

    def import_all(self, src: str = "") -> None:
        from app.backup import import_backup
        src = src or QFileDialog.getOpenFileName(self, "Import backup", "", "Zip (*.zip)")[0]
        if not src:
            return

        def done(c, err):
            if err is None:
                self.toast("success", "Backup imported: " + ", ".join(f"{k} {v}" for k, v in c.items() if v))
                self.refresh()
        self.start(lambda ctx: import_backup(src, self.db, self.s, self._services()), done, "Importing backup", "backup", False)

    def wipe(self, confirm: bool = True) -> None:
        if confirm and QMessageBox.warning(self, "Wipe all data", "This permanently deletes your resumes, knowledge base, history, "
                                           "feedback and logs. Settings are kept.\n\nContinue?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        from app.backup import wipe_all_data
        n = wipe_all_data(self.db, self.s, self.log_dir, self._services())
        self.toast("info", "All stored data was deleted.")
        self.status.setText(f"Deleted {sum(n.values())} stored item(s).")
        self.refresh()


class AboutPage(QWidget):
    def __init__(self, log_dir=None, parent=None):
        super().__init__(parent)
        import platform
        import sys
        import PySide6
        lay = QVBoxLayout(self)
        lay.addWidget(label("Dual-Screen Meeting AI Assistant", "title"))
        lay.addWidget(label("Local-first interview assistant: everything runs on this computer.", "muted", wrap=True))
        lay.addWidget(label(f"Python {sys.version.split()[0]} · PySide6 {PySide6.__version__} · {platform.platform()}", "caption", wrap=True))
        lay.addWidget(label(f"Logs and crash.log: {log_dir or '—'}", "caption", wrap=True, selectable=True))
        lay.addWidget(label("Shortcuts: Ctrl+K command palette · Ctrl+L activity log · ? shortcut list", "caption", wrap=True))
        lay.addStretch(1)
