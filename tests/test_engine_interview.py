"""Whole pipeline (detector -> retrieval -> prompt -> LLM -> validation -> events) with scripted STT/LLM."""
import json
import time
from threading import Event

from app.engine import MeetingEngine
from database.sqlite import Database
from llm.base import LLMProvider
from llm.service import LLMService
from settings.settings_manager import SettingsManager
from tests.pdfgen import make_pdf
from tests.test_resume import CLASSIC


class Prov(LLMProvider):
    name, model = "fake", "fake"
    def __init__(self, replies): self.replies, self.calls = list(replies), []
    def is_available(self): return True
    def list_models(self): return ["fake"]
    def has_model(self, n): return True
    def stream(self, messages, *, json_schema=None, cancel: Event = None, **kw):
        self.calls.append(messages)
        text = json.dumps(self.replies.pop(0))
        for i in range(0, len(text), 11):
            yield text[i:i + 11]


class STT:
    name = "fake"
    def load(self): pass
    def transcribe(self, audio, partial=False): return ""


def rep(q, a, g="general"):
    return {"question": q, "intent": "x", "grounding": g, "answer": a, "confidence": 0.9}


def make(replies, db=None):
    db = db or Database(":memory:")
    s = SettingsManager(db)
    ev = []
    eng = MeetingEngine(s, lambda n, **p: ev.append((n, p)), db=db, stt=STT(), vad=object(), provider=Prov(replies), hardware=object())
    eng.llm = LLMService(eng._provider, s, eng.context)
    return eng, ev


def names(ev): return [n for n, _ in ev]


def test_tell_us_about_yourself_is_answered_from_the_resume():
    eng, ev = make([rep("Tell us about yourself", "I'm Jane Doe, a software engineer with over six years of experience.", "resume")])
    eng.resume.upload_bytes("cv.txt", CLASSIC.encode())
    eng._handle_final("Tell us about yourself", time.monotonic())
    done = dict(ev)["answer_done"]
    assert done["answer"].startswith("I'm Jane Doe") and done["grounding"] == "resume"
    prompt = eng._provider.calls[0][-1]["content"]
    assert "CANDIDATE PROFILE" in prompt and "Jane Doe" in prompt and "QUESTION: Tell us about yourself" in prompt
    assert len(eng._provider.calls) == 1                   # no semantic-check call for a clear prompt


def test_introduce_yourself_without_question_mark_triggers_answer():
    eng, ev = make([rep("Introduce yourself.", "Hello, I'm a software engineer.")])
    eng._handle_final("Introduce yourself", time.monotonic())
    assert "answer_done" in names(ev)


def test_small_talk_never_reaches_the_llm():
    eng, ev = make([])
    for t in ("Thank you everyone.", "Can you hear me?", "How are you doing today?"):
        eng._handle_final(t, time.monotonic())
    assert eng._provider.calls == [] and "question" not in names(ev)


def test_question_summary_is_never_displayed_and_regenerates():
    eng, ev = make([rep("What is Python?", "The speaker is asking what Python is."),
                    rep("What is Python?", "Python is a high-level programming language used for many purposes.")])
    eng._handle_final("What is Python?", time.monotonic())
    shown = "".join(p["text"] for k, p in ev if k == "answer_delta")
    assert "speaker" not in shown.lower() and dict(ev)["answer_done"]["answer"].startswith("Python is a high-level")


def test_persistently_bad_answer_reports_error_and_restores_ui():
    eng, ev = make([rep("What is Python?", "The speaker is asking what Python is.")] * 3)
    eng._handle_final("What is Python?", time.monotonic())
    assert "answer_done" not in names(ev) and "restore_previous" in names(ev)
    assert any(k == "error" and "describing the question" in p["message"] for k, p in ev)


def test_followup_uses_current_project_as_retrieval_hint_and_memory():
    eng, ev = make([rep("Tell me about AI Interview Assistant", "It is a local LLM desktop app.", "resume"),
                    rep("How did you implement this?", "I built it with PySide6 and Ollama.", "resume")])
    eng.resume.upload_bytes("cv.txt", CLASSIC.encode())
    eng._handle_final("Tell me about the AI Interview Assistant project", time.monotonic())
    eng._handle_final("How did you implement this?", time.monotonic() + 100)
    second = eng._provider.calls[1][-1]["content"]
    assert eng.context.persistent.current_project == "AI Interview Assistant"
    assert "PySide6" in second                                 # resume project chunk retrieved via the project hint


def test_long_question_reaches_the_llm_intact_and_facts_are_remembered():
    q = ("You mentioned in your previous answer that you built a Python-based computer vision system using OpenCV and YOLO, "
         "and you said that the system was deployed on edge devices. Can you explain why you selected YOLO instead of another "
         "detection model and what challenges you faced during deployment?")
    eng, ev = make([rep(q, "I picked YOLO for its speed on edge hardware; deployment challenges were memory and thermal limits.")])
    eng._handle_final(q, time.monotonic())
    assert eng._provider.calls[0][-1]["content"].endswith("QUESTION: " + q)
    assert any("YOLO" in f for f in eng.context.facts())


def test_pdf_knowledge_base_feeds_the_answer(tmp_path):
    f = tmp_path / "kb"; f.mkdir()
    (f / "py.pdf").write_bytes(make_pdf([["Python Answers", "", "Q: What are Python decorators?",
                                          "A: A decorator wraps a function to extend its behaviour."]]))
    eng, ev = make([rep("What are Python decorators?", "A decorator wraps a function to extend its behaviour.", "knowledge_base")])
    eng.kb.sync_folder(f)
    eng._handle_final("What are Python decorators?", time.monotonic())
    assert "REFERENCE NOTES" in eng._provider.calls[0][-1]["content"]
    assert dict(ev)["answer_done"]["sources"] and "py.pdf" in dict(ev)["answer_done"]["sources"][0]


def test_detection_is_logged_with_reason(tmp_path):
    import json as j
    from applog import configure
    configure(tmp_path, debug=True)
    eng, ev = make([rep("Tell us about yourself", "I am an engineer.")])
    eng._handle_final("Tell us about yourself", time.monotonic())
    eng._handle_final("Thank you everyone.", time.monotonic())
    lines = [j.loads(l) for l in (tmp_path / "question_detection.log").read_text().splitlines() if l.startswith("{")]
    assert lines[0]["type"] == "INTERVIEW_PROMPT" and lines[0]["action"] == "answer" and lines[0]["reason"]
    assert lines[1]["action"].startswith("ignored") and "[QUESTION DETECTION]" in (tmp_path / "question_detection.log").read_text()
    configure(tmp_path, debug=False)


def test_debug_trace_event_has_every_stage():
    from applog import configure
    import tempfile
    configure(tempfile.mkdtemp(), debug=True)
    eng, ev = make([rep("What is Python?", "Python is a programming language used widely today.")])
    eng._handle_final("What is Python?", time.monotonic())
    trace = dict(ev)["debug_trace"]["text"]
    for part in ("ORIGINAL TRANSCRIPT", "CLEAN TRANSCRIPT", "DETECTED QUESTION", "RETRIEVED RESUME", "RETRIEVED PDF",
                 "CONTEXT USED", "LLM PROMPT", "LLM RESPONSE", "ANSWER VALIDATION", "FINAL ANSWER"):
        assert part in trace
    configure(tempfile.mkdtemp(), debug=False)
    eng2, ev2 = make([rep("What is Python?", "Python is a programming language used widely today.")])
    eng2._handle_final("What is Python?", time.monotonic())
    assert "debug_trace" not in names(ev2)


def test_clear_context_keeps_resume_profile():
    eng, ev = make([rep("Tell us about yourself", "I am Jane.")])
    eng.resume.upload_bytes("cv.txt", CLASSIC.encode())
    eng._handle_final("Tell us about yourself", time.monotonic())
    eng.clear_context()
    assert eng.context.pairs() == [] and "Jane Doe" in eng.context.persistent.resume_context


def test_interview_memory_persists_when_storing_is_enabled(tmp_path):
    db = Database(tmp_path / "m.db")
    eng, ev = make([rep("q", "A decent answer about edge deployment.")], db=db)
    eng.s.set("privacy.store_questions", True)
    eng._handle_final("You said that you deployed models on Jetson boards. Why did you pick Jetson over Raspberry Pi?", time.monotonic())
    eng2, _ = make([], db=Database(tmp_path / "m.db"))
    eng2.services.load_memory()
    assert any("Jetson" in f for f in eng2.context.facts())
