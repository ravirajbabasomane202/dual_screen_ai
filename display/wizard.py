"""First-run wizard: Welcome → Audio → Speech model → LLM → Resume (optional) → Knowledge base (optional) → Finish.

A stepper with progress dots. Every step shows success / failure, has Skip, and long work runs through app.tasks
(progress + real Cancel, results delivered on the GUI thread). UI only; the work is done by the existing packages."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLineEdit, QProgressBar, QStackedWidget,
                               QVBoxLayout, QWidget)

from app.gui_thread import assert_gui_thread
from app.tasks import Task, run_task
from display import theme
from display.i18n import tr
from display.monitor_manager import MonitorManager
from display.widgets.common import button, hbox, label, set_prop

PENDING, OK, FAIL, SKIPPED = "pending", "ok", "fail", "skipped"


class Dots(QWidget):
    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.n, self.index, self.states = n, 0, [PENDING] * n
        self.setFixedHeight(22)
        self.setMinimumWidth(n * 22)
        self.setAccessibleName("Setup progress")

    def set_state(self, index: int, states: list[str]) -> None:
        self.index, self.states = index, list(states)
        self.setAccessibleDescription(f"Step {index + 1} of {self.n}")
        self.update()

    def paintEvent(self, _e) -> None:
        t = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        gap = 22
        x0 = (self.width() - gap * (self.n - 1)) / 2
        for i in range(self.n):
            col = {OK: t["success"], FAIL: t["danger"], SKIPPED: t["text_faint"]}.get(self.states[i], t["border_strong"])
            if i == self.index:
                col = t["accent"]
            r = 6 if i == self.index else 4
            p.setBrush(QColor(col))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(x0 + i * gap - r, self.height() / 2 - r, 2 * r, 2 * r))


class Step(QWidget):
    """One wizard step. `required=False` steps can be skipped."""
    title = ""
    text = ""
    required = False

    def __init__(self, wiz: "SetupWizard"):
        super().__init__()
        self.wiz = wiz
        self.state = PENDING
        self.task: Task | None = None
        lay = QVBoxLayout(self)
        lay.setSpacing(theme.SPACE["md"])
        lay.addWidget(label(tr(self.title), "title"))
        self.info = label(tr(self.text), "muted", wrap=True)
        lay.addWidget(self.info)
        self.body = QVBoxLayout()
        lay.addLayout(self.body)
        self.bar = QProgressBar()
        self.bar.hide()
        self.btn_cancel = button(tr("Cancel"), "ghost", "x")
        self.btn_cancel.hide()
        self.btn_cancel.clicked.connect(self.cancel)
        self.result = label("", "", wrap=True)
        lay.addLayout(hbox(self.bar, self.btn_cancel))
        lay.addWidget(self.result)
        lay.addStretch(1)

    # ---- hooks
    def on_enter(self) -> None:
        pass

    def can_continue(self) -> bool:
        return self.state in (OK, SKIPPED) or not self.required

    # ---- state
    def set_state(self, state: str, message: str = "") -> None:
        self.state = state
        mark = {OK: "✓ ", FAIL: "✗ ", SKIPPED: "— "}.get(state, "")
        self.result.setText(mark + message if message else "")
        set_prop(self.result, "role", {"ok": "success", "fail": "danger"}.get(state, ""))
        self.wiz.refresh_nav()

    def run(self, fn, title: str, on_ok=None) -> None:
        """Run fn(ctx) in the background with a progress bar; message from the returned string."""
        assert_gui_thread("Step.run")
        if self.task is not None and not self.task.finished:
            return
        self.bar.setRange(0, 0)
        self.bar.show()
        self.btn_cancel.show()
        self.btn_cancel.setEnabled(True)
        self.result.setText(tr("Working…"))

        def progress(f, m):
            if f is not None:
                self.bar.setRange(0, 1000)
                self.bar.setValue(int(f * 1000))
            if m:
                self.result.setText(m)

        def done(value, err):
            self.task = None
            self.btn_cancel.hide()
            self.bar.setRange(0, 1000)
            if err is None:
                self.bar.setValue(1000)
                self.set_state(OK, str(value or tr("Done")))
                if on_ok:
                    on_ok(value)
            elif getattr(err, "cancelled", False):
                self.bar.hide()
                self.set_state(PENDING, tr("Cancelled"))
            else:
                self.bar.setValue(0)
                self.set_state(FAIL, getattr(err, "user_message", None) or f"{type(err).__name__}: {err}")
        self.task = run_task(self, fn, done, progress, stage="setup", title=title)

    def cancel(self) -> None:
        if self.task is not None:
            self.btn_cancel.setEnabled(False)
            self.task.cancel()


class Welcome(Step):
    title = "Welcome"
    text = ("This assistant listens to your meeting audio, finds questions and answers them on your second monitor. "
            "Everything runs on this PC — no cloud, no API keys. Setup takes a few minutes (models are downloaded once).")

    def __init__(self, wiz):
        super().__init__(wiz)
        self.btn = button(tr("Detect my hardware"), "primary", "chip")
        self.btn.clicked.connect(lambda: self.detect())
        self.body.addWidget(self.btn)

    def detect(self) -> None:
        from hardware.detector import detect, recommend

        def work(ctx):
            hw = detect()
            rec = recommend(hw)
            self.wiz.hw, self.wiz.rec = hw, rec
            self.wiz.s.set_many({"speech.model": rec.stt_model, "ai.model": rec.llm_model})
            g = ", ".join(f"{x.name} {x.vram_gb:g} GB" for x in hw.gpus) or "none"
            return (f"CPU: {hw.cpu_name} ({hw.cpu_cores} cores) · RAM {hw.ram_gb} GB · GPU {g}\n"
                    f"Recommended → speech {rec.stt_model}, language model {rec.llm_model}, backend {rec.backend}")
        self.run(work, "Detecting hardware")


class AudioStep(Step):
    title = "Audio"
    text = "Choose the output device your meeting plays through (usually your headphones or speakers), then test it."

    def __init__(self, wiz):
        super().__init__(wiz)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setAccessibleName("Audio device")
        self.combo.addItem("")
        self.btn = button(tr("Test capture (play something now)"), "primary", "speaker")
        self.btn.clicked.connect(lambda: self.test())
        self.body.addWidget(self.combo)
        self.body.addWidget(self.btn)
        self._listed = False

    def on_enter(self) -> None:
        if self._listed:
            return
        self._listed = True
        try:
            from audio.wasapi_capture import WasapiLoopbackCapture
            for d in WasapiLoopbackCapture.list_devices():
                self.combo.addItem(d["name"])
        except Exception as e:
            self.info.setText(getattr(e, "user_message", str(e)))

    def test(self) -> None:
        self.wiz.s.set("audio.system_device", self.combo.currentText())

        def work(ctx):
            from audio.audio_manager import AudioManager
            frames = []
            mgr = AudioManager(self.wiz.s, frames.append)
            mgr.start()
            try:
                for i in range(6):
                    ctx.check()
                    time.sleep(1)
                    ctx.progress((i + 1) / 6, "Recording…")
            finally:
                mgr.stop()
            if not frames:
                raise RuntimeError("No audio was captured. Make sure something is playing on the selected device.")
            self.wiz.sample = np.concatenate(frames)
            peak = float(np.abs(self.wiz.sample).max())
            if peak < 0.005:
                raise RuntimeError("Captured only silence. Is audio playing on the selected device?")
            return f"Captured {len(self.wiz.sample) / 16000:.1f}s of audio (peak level {peak:.2f})."
        self.run(work, "Testing audio capture")


class SpeechStep(Step):
    title = "Speech model"
    text = "Downloads the speech-recognition model once (stays on this computer), then checks it on the audio you just recorded."

    def __init__(self, wiz):
        super().__init__(wiz)
        self.btn = button(tr("Download and check"), "primary", "download")
        self.btn.clicked.connect(lambda: self.go())
        self.body.addWidget(self.btn)

    def go(self) -> None:
        def work(ctx):
            from speech.stt_manager import download_whisper_model, is_whisper_downloaded
            m = self.wiz.s.get("speech.model")
            m = self.wiz.rec.stt_model if m == "auto" and self.wiz.rec else m
            if m == "auto":
                m = "base"
            self.wiz.s.set("speech.model", m)
            if not is_whisper_downloaded(m):
                download_whisper_model(m, lambda pct: (ctx.check(), ctx.progress(pct / 100, f"Downloading Whisper {m}… {pct}%")))
            msg = f"Whisper “{m}” is installed."
            if self.wiz.sample is not None:
                from hardware.detector import detect, recommend
                from speech.stt_manager import create_engine
                rec = recommend(self.wiz.hw or detect())
                dev = rec.compute_device if self.wiz.s.get("speech.device") == "auto" else self.wiz.s.get("speech.device")
                eng = create_engine(self.wiz.s, m, dev)
                ctx.progress(None, "Loading the model…")
                eng.load()
                msg += f" Heard: “{eng.transcribe(self.wiz.sample) or '(no speech recognised)'}”"
            return msg
        self.run(work, "Preparing the speech model")


class LLMStep(Step):
    title = "Language model"
    text = "Needs Ollama (https://ollama.com) or llama.cpp running on this computer. We look for it automatically."

    def __init__(self, wiz):
        super().__init__(wiz)
        self.btn_detect = button(tr("Detect"), "secondary", "search")
        self.btn_test = button(tr("Test & download model"), "primary", "play")
        self.btn_detect.clicked.connect(lambda: self.detect())
        self.btn_test.clicked.connect(lambda: self.test())
        self.body.addLayout(hbox(self.btn_detect, self.btn_test, stretch_end=True))

    def on_enter(self) -> None:
        if self.state == PENDING and not self.result.text():
            self.detect()

    def detect(self) -> None:
        def work(ctx):
            from llm.model_manager import detect_local_server
            found = detect_local_server()
            if found:
                self.wiz.s.set_many({"ai.provider": found[0], "ai.host": found[1]})
                return f"Found {found[0]} at {found[1]}."
            raise RuntimeError("No local language-model server found. Install Ollama and start it, then press Detect.")
        self.run(work, "Looking for a local model server")

    def test(self) -> None:
        def work(ctx):
            from llm.model_manager import create_provider, pull_ollama_model
            s = self.wiz.s
            m = s.get("ai.model")
            m = self.wiz.rec.llm_model if m == "auto" and self.wiz.rec else m
            if m == "auto":
                m = "qwen3:4b"
            s.set("ai.model", m)
            prov = create_provider(s, m)
            if not prov.is_available():
                from app.errors import LLMNotRunning
                raise LLMNotRunning()
            if not prov.has_model(m):
                pull_ollama_model(s.get("ai.host"), m, lambda pct, st: (ctx.check(), ctx.progress(pct / 100, f"Downloading {m}… {pct}%")))
            return f"{m} is installed and the server answers."
        self.run(work, "Checking the language model")


class ResumeStep(Step):
    title = "Your resume (optional)"
    text = "Answers about your background are grounded in it. Drop a PDF, DOCX or TXT here, or choose a file."

    def __init__(self, wiz):
        super().__init__(wiz)
        self.setAcceptDrops(True)
        self.btn = button(tr("Choose a file…"), "primary", "upload")
        self.btn.clicked.connect(lambda: self.choose())
        drop = QFrame(objectName="drop")
        drop.setMinimumHeight(70)
        dl = QVBoxLayout(drop)
        dl.addWidget(label(tr("Drop your resume here"), "muted"), 0, Qt.AlignCenter)
        self.body.addWidget(drop)
        self.body.addWidget(self.btn)

    def choose(self, path: str = "") -> None:
        path = path or QFileDialog.getOpenFileName(self, tr("Select resume"), "", "Resumes (*.pdf *.docx *.txt *.md)")[0]
        if not path:
            return
        if self.wiz.db is None:
            self.set_state(FAIL, "No database is open — upload it later in Settings → Resume.")
            return

        def work(ctx):
            from resume.manager import ResumeManager
            r = ResumeManager(self.wiz.db, self.wiz.s).upload(path, cancel=ctx)
            return f"Parsed “{Path(path).name}”: {len(r.parsed.skills)} skills, {len(r.parsed.experience)} jobs."
        self.run(work, "Parsing your resume")

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        for u in e.mimeData().urls():
            if u.toLocalFile().lower().endswith((".pdf", ".docx", ".txt", ".md")):
                self.choose(u.toLocalFile())
                break


class KnowledgeStep(Step):
    title = "Knowledge base (optional)"
    text = "Choose a folder with PDFs, DOCX, TXT or MD notes. They are indexed on this computer and cited in answers."

    def __init__(self, wiz):
        super().__init__(wiz)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder (optional)")
        self.folder.setAccessibleName("Knowledge folder")
        self.btn_pick = button(tr("Browse…"), "secondary", "folder")
        self.btn_go = button(tr("Index this folder"), "primary", "database")
        self.btn_pick.clicked.connect(lambda: self.pick())
        self.btn_go.clicked.connect(lambda: self.index())
        self.body.addLayout(hbox(self.folder, self.btn_pick))
        self.body.addWidget(self.btn_go)

    def pick(self, d: str = "") -> None:
        d = d or QFileDialog.getExistingDirectory(self, tr("Select folder"), self.folder.text())
        if d:
            self.folder.setText(d)

    def index(self) -> None:
        folder = self.folder.text().strip()
        if not folder:
            self.set_state(FAIL, "Choose a folder first, or press Skip.")
            return
        self.wiz.s.set("knowledge.folder", folder)
        if self.wiz.db is None:
            self.set_state(OK, "Saved. The folder is indexed when the app starts.")
            return

        def work(ctx):
            from knowledge.index import KnowledgeIndex
            rep = KnowledgeIndex(self.wiz.db, settings=self.wiz.s).sync_folder(folder, progress_cb=ctx.progress_cb, cancel=ctx)
            return rep.summary()
        self.run(work, "Indexing the folder")


class FinishStep(Step):
    title = "Ready"
    text = "Setup is complete. Pick the monitor for the assistant — your meeting stays on the other one."

    def __init__(self, wiz):
        super().__init__(wiz)
        self.combo = QComboBox()
        self.combo.setAccessibleName("AI monitor")
        self.body.addWidget(self.combo)
        self.summary = label("", "", wrap=True)
        self.body.addWidget(self.summary)

    def on_enter(self) -> None:
        self.combo.clear()
        mons = MonitorManager.list_monitors()
        for m in mons:
            self.combo.addItem(m.label, m.index)
        self.combo.setCurrentIndex(min(1, len(mons) - 1))
        if len(mons) < 2:
            self.info.setText(tr("Only one monitor detected — connect a second one for the full experience."))
        names = {"Welcome": "Hardware", "AudioStep": "Audio", "SpeechStep": "Speech model", "LLMStep": "Language model",
                 "ResumeStep": "Resume", "KnowledgeStep": "Knowledge base"}
        mark = {OK: "✓", FAIL: "✗ (fix later in Settings)", SKIPPED: "skipped", PENDING: "not done"}
        self.summary.setText("\n".join(f"{names[type(s).__name__]}: {mark[s.state]}" for s in self.wiz.steps
                                       if type(s).__name__ in names))
        self.state = OK


class SetupWizard(QDialog):
    def __init__(self, settings, parent=None, db=None):
        super().__init__(parent)
        self.s, self.db = settings, db
        self.hw = self.rec = self.sample = None
        self.setWindowTitle(tr("Dual-Screen AI — Setup"))
        self.setObjectName("window")
        self.setMinimumSize(760, 520)
        self.steps: list[Step] = [Welcome(self), AudioStep(self), SpeechStep(self), LLMStep(self), ResumeStep(self),
                                  KnowledgeStep(self), FinishStep(self)]
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        self.dots = Dots(len(self.steps))
        root.addWidget(self.dots)
        self.stack = QStackedWidget()
        for st in self.steps:
            self.stack.addWidget(st)
        root.addWidget(self.stack, 1)
        self.btn_back = button(tr("Back"), "ghost", "chevron_left")
        self.btn_skip = button(tr("Skip"), "ghost")
        self.btn_next = button(tr("Next"), "primary", "chevron_right")
        self.btn_back.clicked.connect(lambda: self.go(-1))
        self.btn_skip.clicked.connect(lambda: self.skip())
        self.btn_next.clicked.connect(lambda: self.go(1))
        nav = QHBoxLayout()
        nav.addWidget(self.btn_back)
        nav.addStretch(1)
        nav.addWidget(self.btn_skip)
        nav.addWidget(self.btn_next)
        root.addLayout(nav)
        self.index = 0
        self._show(0)

    def _show(self, i: int) -> None:
        self.index = max(0, min(len(self.steps) - 1, i))
        self.stack.setCurrentIndex(self.index)
        self.steps[self.index].on_enter()
        self.refresh_nav()

    def refresh_nav(self) -> None:
        st = self.steps[self.index]
        last = self.index == len(self.steps) - 1
        self.dots.set_state(self.index, [s.state for s in self.steps])
        self.btn_back.setEnabled(self.index > 0 and not (st.task is not None))
        self.btn_skip.setVisible(not st.required and not last and self.index > 0)
        self.btn_next.setText(tr("Finish") if last else tr("Next"))
        self.btn_next.setEnabled(st.can_continue() and st.task is None)

    def go(self, d: int) -> None:
        if d > 0 and self.index == len(self.steps) - 1:
            self.finish()
        else:
            self._show(self.index + d)

    def skip(self) -> None:
        st = self.steps[self.index]
        st.set_state(SKIPPED, tr("Skipped — you can do this later in Settings."))
        self._show(self.index + 1)

    def finish(self) -> None:
        fin = self.steps[-1]
        if fin.combo.currentData() is not None:
            self.s.set("display.ai_monitor", int(fin.combo.currentData()))
        self.s.set("app.first_run_done", True)
        self.accept()

