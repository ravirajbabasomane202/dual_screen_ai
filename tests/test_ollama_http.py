"""The REAL OllamaProvider against a stub Ollama server (no model needed): payloads, error bodies, truncation signals."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.errors import LLMContextOverflow, LLMModelMissing
from context.context_manager import ContextManager
from database.sqlite import Database
from llm.ollama import OllamaProvider
from llm.service import LLMService
from settings.settings_manager import SettingsManager
from threading import Event


class Stub(BaseHTTPRequestHandler):
    script: list = []
    seen: list = []

    def log_message(self, *a): pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "stub:latest"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append(payload)
        status, lines = Stub.script.pop(0)
        body = "\n".join(json.dumps(l) if not isinstance(l, str) else l for l in lines).encode()
        self.send_response(status); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


@pytest.fixture
def server():
    Stub.script, Stub.seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def chat(text, **done):
    mid = [{"message": {"content": text[i:i + 15]}, "done": False} for i in range(0, len(text), 15)]
    return [*mid, {"message": {"content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 321, "eval_count": 40, **done}]


GOOD = json.dumps({"question": "What is Python?", "intent": "definition", "grounding": "general",
                   "answer": "Python is a high-level programming language used widely.", "confidence": 0.9})


def make(server, **kw):
    s = SettingsManager(Database(":memory:"))
    p = OllamaProvider(server, "stub", timeout=5, num_ctx=kw.get("num_ctx", 6144))
    return p, LLMService(p, s, ContextManager())


def test_num_ctx_and_stats_are_sent_and_captured(server):
    Stub.script = [(200, chat(GOOD))]
    p, svc = make(server)
    res = svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())
    assert res.valid and Stub.seen[0]["options"]["num_ctx"] == 6144 and Stub.seen[0]["format"]["required"][0] == "question"
    assert p.last_stats["prompt_tokens"] == 321 and p.last_stats["finish_reason"] == "stop"


def test_think_option_rejected_by_model_is_retried_without_it(server):
    Stub.script = [(400, ['{"error":"\\"stub\\" does not support thinking"}']), (200, chat(GOOD))]
    p, svc = make(server)
    res = svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())
    assert res.valid and "think" in Stub.seen[0] and "think" not in Stub.seen[1]


def test_http_error_body_is_surfaced_not_swallowed(server):
    Stub.script = [(500, ['{"error":"llama runner process has terminated: out of memory"}'])]
    p, svc = make(server)
    with pytest.raises(RuntimeError, match="out of memory"):
        svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())


def test_context_overflow_is_detected_and_retried_smaller(server):
    Stub.script = [(400, ['{"error":"input length exceeds the context length"}']), (200, chat(GOOD))]
    p, svc = make(server)
    assert svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event()).valid
    assert len(Stub.seen) == 2


def test_missing_model_message(server):
    Stub.script = [(404, ["{}"])]
    p, svc = make(server)
    with pytest.raises(LLMModelMissing, match="ollama pull stub"):
        svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())


def test_prompt_filling_the_window_is_flagged(server):
    Stub.script = [(200, chat(GOOD, prompt_eval_count=6140))]
    p, svc = make(server)
    svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())
    assert p.last_stats.get("context_truncated") is True


def test_length_cutoff_triggers_retry_with_more_tokens(server):
    cut = chat(GOOD[:60], done_reason="length")
    Stub.script = [(200, cut), (200, chat(GOOD))]
    p, svc = make(server)
    res = svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())
    assert res.valid and Stub.seen[1]["options"]["num_predict"] > Stub.seen[0]["options"]["num_predict"]


def test_test_llm_end_to_end_against_stub(server):
    from llm.diagnostics import test_llm
    Stub.script = [(200, chat("Python is a high-level programming language used for many things.")), (200, chat(GOOD))]
    p, _ = make(server)
    p.model = "stub"
    r = test_llm(p)
    assert r.ok, r.format()
