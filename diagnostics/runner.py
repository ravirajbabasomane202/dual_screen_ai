"""Full-pipeline diagnostics with a clear PASS / FAIL / WARN / SKIP report and recommended actions."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    action: str = ""
    seconds: float = 0.0


@dataclass
class DiagnosticReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status in (PASS, WARN, SKIP) for c in self.checks)

    def get(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def format(self) -> str:
        lines = ["SYSTEM DIAGNOSTICS", ""]
        for c in self.checks:
            lines.append(f"{c.name} ".ljust(24, ".") + f" {c.status}")
        for c in self.checks:
            if c.status in (FAIL, WARN) and (c.detail or c.action):
                lines += ["", f"{c.name} {'Error' if c.status == FAIL else 'Note'}:", c.detail or "—"]
                if c.action:
                    lines += ["", "Recommended action:", c.action]
        return "\n".join(lines)


def _run(report: DiagnosticReport, name: str, fn) -> Check:
    t = time.perf_counter()
    try:
        res = fn()
        status, detail, action = (res + ("", ""))[:3] if isinstance(res, tuple) else (PASS, str(res or ""), "")
    except Exception as e:
        from app.errors import AppError
        from applog import log_event
        from llm.diagnostics import recommend_action
        status = FAIL
        detail = (e.user_message if isinstance(e, AppError) else f"{type(e).__name__}: {e}").replace("\n\n", " ")
        action = recommend_action(e)
        log_event("llm", f"diagnostic '{name}' failed", level=40, exc_info=True, error=detail)
    c = Check(name, status, detail, action, time.perf_counter() - t)
    report.checks.append(c)
    return c


def run_full_diagnostics(settings, db=None, engine=None, *, skip_hardware: bool = False) -> DiagnosticReport:
    from database.sqlite import Database
    rep = DiagnosticReport()
    own_db = db is None
    db = db or Database(":memory:")
    scratch = Path(tempfile.mkdtemp(prefix="dsai_diag_"))

    # ---- database
    def database():
        db.execute("REPLACE INTO settings(key,value) VALUES('__diag__','1')")
        ok = db.query_one("SELECT value FROM settings WHERE key='__diag__'")
        db.execute("DELETE FROM settings WHERE key='__diag__'")
        return PASS, f"schema v{db.schema_version}, read/write OK" if ok else (FAIL, "write did not persist", "Check disk space/permissions")
    _run(rep, "Database", database)

    # ---- microphone / VAD / STT (hardware & models: WARN or SKIP rather than hard FAIL when optional)
    def microphone():
        try:
            from audio.microphone import MicrophoneCapture
            devs = MicrophoneCapture.list_devices()
        except Exception as e:
            return (FAIL if settings.get("audio.mic_enabled") else WARN), f"input devices unavailable ({type(e).__name__}: {e})", \
                "Microphone is optional (system audio is captured through WASAPI loopback). Install PortAudio/sounddevice to use it."
        if not devs:
            return (FAIL if settings.get("audio.mic_enabled") else WARN), "no input devices found", "Connect a microphone or disable it in Settings → Audio."
        return PASS, f"{len(devs)} input device(s)"
    _run(rep, "Microphone", (lambda: (SKIP, "skipped")) if skip_hardware else microphone)

    vad_holder = {}

    def vad():
        import numpy as np
        from vad.silero_vad import SileroVAD
        v = SileroVAD()
        p = max(v.prob(np.zeros(512, np.float32)) for _ in range(3))
        vad_holder["ok"] = True
        return PASS, f"silence probability {p:.2f}" if p < 0.5 else (WARN, f"silence scored {p:.2f} (expected < 0.5)", "")
    _run(rep, "VAD", (lambda: (SKIP, "skipped")) if skip_hardware else vad)

    def stt():
        import numpy as np
        from hardware.detector import detect, recommend
        from speech.stt_manager import create_engine, is_whisper_downloaded
        rec = recommend(detect())
        model = rec.stt_model if settings.get("speech.model") == "auto" else settings.get("speech.model")
        if settings.get("speech.engine") == "faster-whisper" and not is_whisper_downloaded(model):
            return FAIL, f"Whisper model '{model}' is not downloaded", "Run the Setup wizard (or Settings → Speech) to download it."
        eng = engine._stt if engine is not None and engine._stt is not None else create_engine(
            settings, model, rec.compute_device if settings.get("speech.device") == "auto" else settings.get("speech.device"))
        if engine is None or engine._stt is None:
            eng.load()
        txt = eng.transcribe(np.zeros(16000, np.float32))
        return PASS, f"{eng.name}/{model} transcribed 1 s of silence -> {txt!r}"
    _run(rep, "STT", (lambda: (SKIP, "skipped")) if skip_hardware else stt)

    # ---- question detector
    def detector():
        from question.classifier import QuestionClassifier
        c = QuestionClassifier()
        bad = [t for t in ("Tell us about yourself.", "Introduce yourself.", "Walk me through your resume.",
                           "What is your experience with Python?", "Let's talk about your previous role.")
               if not c.classify(t).is_actionable]
        if bad:
            return FAIL, "missed: " + "; ".join(bad), "See logs/question_detection.log"
        if c.classify("Thank you everyone.").is_actionable:
            return FAIL, "small talk classified as a question", ""
        return PASS, "interview prompts, questions and small talk classified correctly"
    _run(rep, "Question Detector", detector)

    # ---- context manager
    def context():
        from context.context_manager import ContextManager
        from context.token_manager import ContextBudgeter
        cm = ContextManager()
        cm.observe("You mentioned that you built a vision system using OpenCV and YOLO. Why YOLO?")
        for i in range(10):
            cm.add(f"old question {i}", "old answer " * 40)
        plan = ContextBudgeter(1024, 300).plan(question="Why did you choose YOLO?", system_tokens=300,
                                               history=cm.pairs(), facts=cm.facts(), memory_text=cm.memory_text())
        if "YOLO" not in " ".join(plan.facts) + plan.memory_text:
            return FAIL, "important fact lost during budgeting", ""
        return PASS, f"budget plan OK (history kept {len(plan.history)}, dropped {plan.dropped or 'nothing'})"
    _run(rep, "Context Manager", context)

    # ---- resume parser
    def resume():
        from diagnostics.samples import SAMPLE_RESUME
        from resume import parse_resume
        p = parse_resume(SAMPLE_RESUME)
        if not (p.name and p.skills and p.experience):
            return FAIL, f"parsed name={p.name!r} skills={len(p.skills)} experience={len(p.experience)}", ""
        return PASS, f"{p.name}: {len(p.skills)} skills, {len(p.experience)} job(s), {len(p.projects)} project(s)"
    _run(rep, "Resume Parser", resume)

    # ---- PDF parser + retrieval
    def pdf():
        from diagnostics.samples import SAMPLE_KB_PAGES, make_pdf
        from knowledge.chunker import chunk_document
        from knowledge.pdf_parser import parse_document
        f = scratch / "sample.pdf"
        f.write_bytes(make_pdf(SAMPLE_KB_PAGES))
        doc = parse_document(f)
        chunks = chunk_document(doc)
        if doc.status != "ok" or not chunks:
            return FAIL, f"status={doc.status}, chunks={len(chunks)}", "Install/upgrade pypdf."
        return PASS, f"{len(doc.pages)} page(s) -> {len(chunks)} chunk(s)"
    _run(rep, "PDF Parser", pdf)

    def retrieval():
        from database.sqlite import Database as DB
        from knowledge import KnowledgeIndex
        kdb = DB(":memory:")
        idx = KnowledgeIndex(kdb)
        rep1 = idx.sync_folder(scratch)
        t = time.perf_counter()
        hits = idx.search("what are python decorators")
        ms = (time.perf_counter() - t) * 1000
        again = idx.sync_folder(scratch)
        if not hits or "decorator" not in hits[0].text.lower():
            return FAIL, "relevant chunk not retrieved", "Re-index the knowledge base."
        if again.updated or again.added:
            return FAIL, "unchanged files were re-indexed", "Index cache is not working (see retrieval.log)."
        return PASS, f"top hit {hits[0].label} in {ms:.1f} ms; second sync reused the index ({again.unchanged} unchanged)"
    _run(rep, "PDF Retrieval", retrieval)

    # ---- LLM (one real test shared by three rows)
    from llm.diagnostics import test_llm
    holder = {}

    def llm_conn():
        from llm.model_manager import create_provider
        if engine is not None and engine._provider is not None:
            prov = engine._provider
        else:
            from hardware.detector import detect, recommend
            model = settings.get("ai.model")
            if model == "auto":
                model = recommend(detect()).llm_model
            prov = create_provider(settings, model)
        holder["provider"] = prov
        if not prov.is_available():
            from app.errors import LLMNotRunning
            raise LLMNotRunning()
        if not prov.has_model(prov.model):
            from app.errors import LLMModelMissing
            raise LLMModelMissing(f"Model '{prov.model}' is not installed.")
        return PASS, f"{prov.name} / {prov.model}"
    c = _run(rep, "LLM Connection", llm_conn)

    def llm_gen():
        if c.status != PASS:
            return SKIP, "LLM connection failed"
        r = test_llm(holder["provider"], settings)
        holder["result"] = r
        if not (r.generation and r.parsing and r.answer_valid):
            return FAIL, r.error or "generation failed", r.recommendation
        return PASS, f"{r.latency:.2f} s — {r.response[:80]}"
    _run(rep, "LLM Generation", llm_gen)

    def parser():
        from llm.prompt import parse_llm_json
        from llm.validation import validate_answer
        new = parse_llm_json('{"question":"What is Python?","intent":"definition","grounding":"general","answer":"Python is a high-level language.","confidence":0.9}')
        old = parse_llm_json('{"is_question":true,"confidence":0.9,"clean_question":"What is Python?","answer":"A language.","needs_clarification":false}')
        bad = validate_answer("What is Python?", "The speaker is asking what Python is.")
        if not (new.answer and old.answer) or bad.valid:
            return FAIL, "parser/validator contract broken", ""
        return PASS, "new + legacy JSON parsed; question-summary answers rejected"
    _run(rep, "Answer Parser", parser)

    # ---- display
    def display():
        try:
            from PySide6.QtWidgets import QApplication
        except Exception as e:
            return FAIL, f"PySide6 not importable: {e}", "pip install PySide6"
        if QApplication.instance() is None:
            if sys.platform != "win32" and not os.environ.get("DISPLAY"):
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            QApplication([])
        from display.ai_window import AIWindow
        w = AIWindow(settings)
        w.begin_question("diagnostic"); w.append_answer("ok"); w.finish_answer("q", "ok", 0.9, 0.1, False)
        from display.monitor_manager import MonitorManager
        n = len(MonitorManager.list_monitors())
        w.close()
        return PASS, f"AI window renders; {n} monitor(s) detected"
    _run(rep, "Display", display)

    if own_db:
        db.close()
    return rep
