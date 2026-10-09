"""Pipeline plumbing with a scripted STT/LLM (the *logic* is real: detector, thresholds, dup filter,
streaming parser, context). Real-model coverage lives in test_integration.py."""
import json
import time
from threading import Event

import numpy as np
import pytest

from app.engine import MeetingEngine
from database.sqlite import Database
from llm.base import LLMProvider
from settings.settings_manager import SettingsManager


class FakeProvider(LLMProvider):
    name, model = "fake", "fake"
    def __init__(self, replies, available=True):
        self.replies, self.calls, self.available = replies, [], available
    def is_available(self): return self.available
    def list_models(self): return ["fake"]
    def has_model(self, n): return True
    def stream(self, messages, *, json_schema=None, cancel: Event = None, **kw):
        self.calls.append(messages)
        text = json.dumps(self.replies.pop(0))
        for i in range(0, len(text), 9):          # dribble chunks like a real stream
            if cancel is not None and cancel.is_set():
                return
            yield text[i:i + 9]


class FakeSTT:
    name = "fake"
    def load(self): pass
    def transcribe(self, audio, partial=False): return ""


def make(replies, **settings):
    s = SettingsManager(Database(":memory:"))
    s.set_many(settings)
    events = []
    eng = MeetingEngine(s, lambda n, **p: events.append((n, p)), stt=FakeSTT(), vad=object(),
                        provider=FakeProvider(replies), hardware=object())
    from llm.service import LLMService
    eng.llm = LLMService(eng._provider, s, eng.context)
    return eng, events


def reply(q, a, conf=0.95, is_q=True):
    return {"is_question": is_q, "confidence": conf, "clean_question": q, "answer": a, "needs_clarification": False}


def names(events): return [n for n, _ in events]


def test_clear_question_streams_clean_question_and_answer():
    eng, ev = make([reply("What is the difference between classification and regression?",
                          "Classification predicts categories. Regression predicts numbers.")])
    eng._handle_final("so can you tell me like what is the diff between classification and regression basically", time.monotonic())
    n = names(ev)
    assert n.index("question") < n.index("clean_question") < n.index("answer_delta") < n.index("answer_done")
    done = dict(ev)["answer_done"]
    assert done["answer"].startswith("Classification predicts") and done["question"].endswith("regression?")
    streamed = "".join(p["text"] for k, p in ev if k == "answer_delta")
    assert streamed == done["answer"]
    assert [s["state"] for k, s in ev if k == "status"][-1] == "ready"


def test_non_question_never_calls_llm():
    eng, ev = make([])
    eng._handle_final("Today we will talk about overfitting.", time.monotonic())
    assert eng._provider.calls == [] and "question" not in names(ev)


def test_ambiguous_goes_through_semantic_check_then_answers():
    eng, ev = make([reply("Why do models overfit?", "", conf=0.97),             # classify
                    reply("Why do models overfit?", "They memorise noise.")])    # answer
    eng._handle_final("why do models overfit", time.monotonic())
    assert len(eng._provider.calls) == 2 and "answer_done" in names(ev)


def test_confirm_zone_waits_for_user_then_manual_trigger_answers():
    eng, ev = make([reply("Is this model overfitting?", "", conf=0.9), reply("Is this model overfitting?", "Yes, it is.")])
    eng._handle_final("is this model overfitting", time.monotonic())
    assert "pending" in names(ev) and "answer_done" not in names(ev)
    eng._brain_q.put(("manual", eng._pending[0], time.monotonic()))
    kind, text, t = eng._brain_q.get_nowait()
    eng._answer(text, 1.0, t, manual=True)
    assert "answer_done" in names(ev)


def test_duplicate_question_is_ignored_within_timeout():
    eng, ev = make([reply("What is overfitting?", "Memorising noise.")])
    eng._handle_final("What is overfitting?", time.monotonic())
    eng._handle_final("what is overfitting", time.monotonic())
    assert len(eng._provider.calls) == 1 and names(ev).count("answer_done") == 1


def test_llm_says_not_a_question_stops_early_and_restores_ui():
    eng, ev = make([reply("Okay.", "", conf=0.2, is_q=False)])
    eng._handle_final("What is going on here?", time.monotonic())
    assert "answer_done" not in names(ev) and "restore_previous" in names(ev)


def test_context_resolves_followups_when_enabled():
    eng, ev = make([reply("What is supervised learning?", "Learning from labelled data."),
                    reply("What are examples of supervised learning?", "Spam detection.")])
    eng._handle_final("What is supervised learning?", time.monotonic())
    eng._handle_final("What are its examples?", time.monotonic())
    second = eng._provider.calls[1]
    assert any("supervised learning" in m["content"] for m in second[:-1])


def test_context_off_sends_no_history():
    eng, ev = make([reply("What is supervised learning?", "x"), reply("What are examples?", "y")], **{"ai.context": False})
    eng.context.enabled = False
    eng._handle_final("What is supervised learning?", time.monotonic())
    eng.context.enabled = False
    eng._handle_final("What are its examples?", time.monotonic())
    assert len(eng._provider.calls[1]) == 2   # system + current only


def test_ollama_down_reports_friendly_error_and_does_not_crash():
    eng, ev = make([])
    eng._provider.available = False
    with pytest.raises(Exception) as ei:
        eng._answer("What is overfitting?", 0.99, time.monotonic())
    assert "Ollama is not running" in ei.value.user_message


def test_privacy_defaults_store_nothing_until_enabled():
    eng, ev = make([reply("What is overfitting?", "Memorising noise."),
                    reply("What is a neural network?", "Layers of units.")])
    eng.db = Database(":memory:")
    eng._handle_final("What is overfitting?", time.monotonic())
    assert eng.db.recent_questions() == []
    eng.s.set("privacy.store_questions", True)
    eng._handle_final("What is a neural network?", time.monotonic())
    assert [q for q, _ in eng.db.recent_questions()] == ["What is a neural network?"]
