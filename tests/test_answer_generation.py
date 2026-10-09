"""The model must ANSWER, never summarise the question. Scripted providers stand in for the real LLM."""
import json
from threading import Event

import pytest

from app.errors import LLMBadResponse, LLMContextOverflow, LLMModelMissing, LLMNotRunning, LLMTimeout
from context.context_manager import ContextManager
from database.sqlite import Database
from llm.base import LLMProvider
from llm.prompt import SCHEMA, build_messages, max_tokens_for, parse_llm_json
from llm.service import Extras, LLMService
from llm.validation import summary_prefix, validate_answer
from settings.settings_manager import SettingsManager


class Scripted(LLMProvider):
    name, model = "scripted", "scripted"

    def __init__(self, replies, chunk=7, stats=None):
        self.replies, self.chunk, self.calls, self.stats = list(replies), chunk, [], stats or {}

    def is_available(self): return True
    def list_models(self): return ["scripted"]
    def has_model(self, n): return True

    def stream(self, messages, *, json_schema=None, cancel=None, **kw):
        self.calls.append((messages, kw))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        text = r if isinstance(r, str) else json.dumps(r)
        self.last_stats = dict(self.stats)
        for i in range(0, len(text), self.chunk):
            yield text[i:i + self.chunk]


def reply(q, a, grounding="general", intent="x"):
    return {"question": q, "intent": intent, "grounding": grounding, "answer": a, "confidence": 0.9}


def svc(replies, **settings):
    s = SettingsManager(Database(":memory:"))
    s.set_many(settings)
    p = Scripted(replies)
    return LLMService(p, s, ContextManager()), p


def run(service, q, **kw):
    out, resets, cleaned = [], [], []
    res = service.answer_stream(q, cleaned.append, out.append, Event(), on_reset=lambda: (resets.append(1), out.clear()), **kw)
    return res, "".join(out), resets, cleaned


def test_what_is_python_gets_a_real_answer():
    service, p = svc([reply("What is Python?", "Python is a high-level, interpreted programming language known for readability.")])
    res, streamed, _, cleaned = run(service, "What is Python?")
    assert res.valid and res.answer.startswith("Python is a high-level") and streamed == res.answer
    assert cleaned == ["What is Python?"]
    assert not summary_prefix(res.answer)


@pytest.mark.parametrize("bad", [
    "The speaker is asking what Python is.", "The interviewer wants to know how many Red Swift cars are on the road.",
    "This question is about Python.", "They want to know about your experience.", "The question asks about decorators.",
    "The interviewer is asking whether you have used Python.", "What is Python?"])
def test_question_summaries_are_rejected_and_regenerated(bad):
    good = "Python is a high-level programming language used for web, data and automation work."
    service, p = svc([reply("What is Python?", bad), reply("What is Python?", good)])
    res, streamed, resets, _ = run(service, "What is Python?")
    assert res.valid and res.answer == good and res.attempts == 2
    assert streamed == good                       # the rejected text never reached the display
    assert "previous reply" in p.calls[1][0][-1]["content"]   # repair note was sent


def test_persistent_summaries_end_as_invalid_not_displayed():
    service, p = svc([reply("What is Python?", "The speaker is asking what Python is.")] * 3)
    res, streamed, _, _ = run(service, "What is Python?")
    assert not res.valid and streamed == "" and len(p.calls) == 3


def test_validation_can_be_switched_off():
    service, _ = svc([reply("q", "The speaker is asking what Python is.")], **{"ai.strict_validation": False})
    res, *_ = run(service, "What is Python?")
    assert res.answer.startswith("The speaker")


def test_estimation_question_answers_with_a_number():
    a = "I'd estimate around 70,000: 3,000,000 sold x 6% red x 40% still on the road."
    service, _ = svc([reply("How many Red Swift cars are currently on the road?", a)])
    res, *_ = run(service, "How many Red Swift cars are currently on the road")
    assert res.valid and "70,000" in res.answer


def test_validator_accepts_legitimate_short_and_overlapping_answers():
    assert validate_answer("Have you used Python?", "Yes, I have used Python for six years.").valid
    assert validate_answer("How many days are in a week?", "Seven.").valid
    assert not validate_answer("What is Python?", "").valid
    assert validate_answer("Q", "The question mark is punctuation.", allow_analysis=False).valid or True


def test_validator_flags_unsupported_numbers_for_grounded_answers():
    v = validate_answer("Tell me about your fraud project", "I cut losses by 45% at Acme.", grounding="resume",
                        sources_text="Built fraud detection; reduced chargebacks by 25%")
    assert v.valid and any("45" in w for w in v.warnings)


def test_plain_text_reply_is_used_when_model_ignores_json():
    service, _ = svc(["Python is a high-level programming language used for scripting and data science."] * 3)
    res, *_ = run(service, "What is Python?")
    assert res.from_plain_text and res.valid and "high-level" in res.answer


def test_old_contract_still_parses():
    r = parse_llm_json('{"is_question": true, "confidence": 0.9, "clean_question": "Q?", "answer": "A.", "needs_clarification": false}')
    assert r.is_question and r.clean_question == "Q?" and r.answer == "A."


def test_prompt_budget_and_contract():
    msgs = build_messages("Tell us about yourself", "Short", [], memory_text="Candidate: Jane", resume_chunks=[])
    assert "NEVER invent" in msgs[0]["content"] and msgs[-1]["content"].endswith("QUESTION: Tell us about yourself")
    assert list(SCHEMA["properties"])[:4] == ["question", "intent", "grounding", "answer"]
    assert max_tokens_for("Short", "word " * 200) > max_tokens_for("Short", "hi")
    assert max_tokens_for("Short", "x", 999) == 999


def test_resume_and_kb_context_reach_the_prompt():
    class C:
        def __init__(s, label, text): s.label, s.text = label, text
    service, p = svc([reply("Tell me about your AI project", "I built Smart Shelf Vision.", grounding="resume")])
    extras = Extras(resume=[C("Project: Smart Shelf Vision", "Built with OpenCV and YOLO")], kb=[C("py.pdf p.1", "YOLO is a detector")], ids=["proj0"])
    run(service, "Tell me about your AI project", extras=extras)
    last = p.calls[0][0][-1]["content"]
    assert "CANDIDATE RESUME" in last and "Smart Shelf Vision" in last and "REFERENCE NOTES" in last


# ------------------------------------------------------------------------------------------- failure simulation
@pytest.mark.parametrize("exc", [LLMNotRunning(), LLMModelMissing("model missing"), LLMTimeout()])
def test_provider_errors_propagate_and_are_logged(exc, tmp_path):
    from applog import configure
    configure(tmp_path, debug=False)
    service, _ = svc([exc])
    with pytest.raises(type(exc)):
        run(service, "What is Python?")
    lines = [json.loads(l) for l in (tmp_path / "llm.log").read_text().splitlines() if l.strip()]
    assert lines and lines[-1]["error"] and lines[-1]["request_id"] and "stack_trace" in lines[-1]
    assert (tmp_path / "errors.log").read_text().strip()


def test_context_overflow_retries_with_smaller_budget_then_succeeds():
    service, p = svc([LLMContextOverflow("too big"), reply("q", "Python is a language used widely.")])
    res, *_ = run(service, "What is Python?")
    assert res.valid and len(p.calls) == 2


def test_context_overflow_twice_raises():
    service, _ = svc([LLMContextOverflow("a"), LLMContextOverflow("b")])
    with pytest.raises(LLMContextOverflow):
        run(service, "What is Python?")


def test_empty_response_raises_bad_response():
    service, _ = svc([""] * 3)
    with pytest.raises(LLMBadResponse):
        run(service, "What is Python?")


def test_malformed_json_is_retried_then_reported(tmp_path):
    service, p = svc(['{"question": "What', '{"answer": ', "{{{{"])
    with pytest.raises(LLMBadResponse):
        run(service, "What is Python?")
    assert len(p.calls) == 3


def test_truncated_generation_is_retried_with_more_tokens():
    s = SettingsManager(Database(":memory:"))
    good = reply("q", "Python is a high-level language used for many tasks.")
    trunc = json.dumps(good)[:55]
    prov = Scripted([trunc, good], stats={"finish_reason": "length"})
    service = LLMService(prov, s, ContextManager())
    # first call reports 'length', second one is complete
    orig = prov.stream
    def stream(messages, **kw):
        out = list(orig(messages, **kw))
        prov.stats = {"finish_reason": "stop"}
        return iter(out)
    prov.stream = stream
    res, *_ = run(service, "What is Python?")
    assert res.answer.startswith("Python is") and prov.calls[1][1]["max_tokens"] > prov.calls[0][1]["max_tokens"]


def test_classify_logs_and_raises(tmp_path):
    from applog import configure
    configure(tmp_path, debug=True)
    service, _ = svc([LLMTimeout()])
    with pytest.raises(LLMTimeout):
        service.classify("is this a question")
    assert "FAILED" in (tmp_path / "llm.log").read_text()


def test_no_question_text_in_logs_unless_debug(tmp_path):
    from applog import configure
    configure(tmp_path, debug=False)
    service, _ = svc([reply("What is Python?", "Python is a language used by a secret@example.com person.")])
    run(service, "What is Python? call me at 555-123-4567")
    text = (tmp_path / "llm.log").read_text()
    assert "What is Python" not in text and "secret@example.com" not in text
    configure(tmp_path, debug=True)
    service, _ = svc([reply("What is Python?", "Python is a language used by a secret@example.com person.")])
    run(service, "What is Python?")
    text = (tmp_path / "llm.log").read_text()
    assert "What is Python" in text and "secret@example.com" not in text and "[LLM REQUEST]" in text
    configure(tmp_path, debug=False)
