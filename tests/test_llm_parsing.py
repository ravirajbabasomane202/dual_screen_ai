import json
import pytest
from llm.prompt import JSONFieldStreamer, ParseError, build_messages, parse_llm_json

GOOD = {"is_question": True, "confidence": 0.94, "clean_question": "What is the difference between A and B?",
        "answer": "A predicts \"categories\".\nB predicts numbers.", "needs_clarification": False}


def test_parse_plain_fenced_and_thinking():
    raw = json.dumps(GOOD)
    for variant in (raw, f"```json\n{raw}\n```", f"<think>hmm</think>\n{raw}", f"Sure! {raw} Hope this helps"):
        r = parse_llm_json(variant)
        assert r.is_question and r.confidence == 0.94 and r.clean_question.startswith("What is")


def test_parse_truncated_json_salvages_fields():
    raw = json.dumps(GOOD)[:-40]
    r = parse_llm_json(raw)
    assert r.is_question and r.clean_question and r.answer.startswith("A predicts")


def test_parse_garbage_raises():
    with pytest.raises(ParseError):
        parse_llm_json("I cannot help with that.")


def test_streamer_reassembles_answer_across_arbitrary_chunks():
    raw = json.dumps(GOOD)
    for size in (1, 2, 3, 7, 50):
        s = JSONFieldStreamer("answer")
        out = "".join(s.feed(raw[i:i + size]) for i in range(0, len(raw), size))
        assert out == GOOD["answer"] and s.done


def test_streamer_handles_unicode_escape():
    s = JSONFieldStreamer("answer")
    assert s.feed('{"answer": "caf\\u00e9 ok"}') == "café ok"


def test_messages_include_history_and_length_rule():
    m = build_messages("what are its examples", "Very Short", [("What is supervised learning?", "Learning from labels.")])
    assert m[0]["role"] == "system" and "max 40 words" in m[0]["content"]
    assert "supervised learning" in m[1]["content"] and m[-1]["content"].endswith("what are its examples")
