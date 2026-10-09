"""'Test LLM': exercises the model WITHOUT any interview pipeline, in two layers, so a failure points at one stage:
   1. connection   – runtime reachable, model installed
   2. generation   – plain completion works (no JSON contract)
   3. parsing      – the real structured path (prompt → JSON → parse → answer validation) on a known question
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Event

from app.errors import (AnswerRejected, AppError, LLMBadResponse, LLMContextOverflow, LLMModelMissing, LLMNotRunning,
                        LLMTimeout)

TEST_PROMPT = "Answer this question in one sentence:\nWhat is Python?"


def recommend_action(exc: BaseException, settings=None) -> str:
    g = (lambda k, d: settings.get(k, d)) if settings is not None else (lambda k, d: d)
    if isinstance(exc, LLMNotRunning):
        return "Start the local LLM runtime (Ollama: run 'ollama serve'; llama.cpp: start llama-server) and check Settings → LLM → endpoint."
    if isinstance(exc, LLMModelMissing):
        return "Install the model (e.g. 'ollama pull qwen3:4b') or pick an installed model in Settings → LLM."
    if isinstance(exc, LLMTimeout):
        return f"Increase the timeout (now {g('ai.timeout', 60)} s), choose a smaller model, or use a shorter answer length."
    if isinstance(exc, LLMContextOverflow):
        k = int(g("retrieval.top_k", 4))
        return (f"Reduce retrieved knowledge-base chunks from {k} to {max(1, k // 2)} (Settings → Knowledge Base) or raise the "
                f"context window (now {g('ai.context_window', 8192)}).")
    if isinstance(exc, LLMBadResponse):
        return "Set temperature to 0.0-0.2, try a larger instruct model, and inspect logs/llm.log (raw response)."
    if isinstance(exc, AnswerRejected):
        return "The model describes questions instead of answering them: use a larger/instruct-tuned model; see logs/llm.log."
    return "See logs/errors.log for the stack trace."


@dataclass
class LLMTestResult:
    model: str = ""
    provider: str = ""
    connection: bool = False
    generation: bool = False
    parsing: bool = False
    answer_valid: bool = False
    latency: float = 0.0
    response: str = ""
    error: str = ""
    recommendation: str = ""
    steps: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.connection and self.generation and self.parsing and self.answer_valid

    def format(self) -> str:
        yn = lambda b: "OK" if b else "FAILED"
        lines = [f"Model: {self.model}", f"Provider: {self.provider}", f"Connection: {yn(self.connection)}",
                 f"Generation: {yn(self.generation)}", f"Parsing: {yn(self.parsing)}",
                 f"Answer validation: {yn(self.answer_valid)}", f"Latency: {self.latency:.2f} s",
                 f"Response: {self.response or '—'}"]
        if self.error:
            lines += [f"Error: {self.error}", f"Recommended action: {self.recommendation}"]
        return "\n".join(lines)


def test_llm(provider, settings=None) -> LLMTestResult:
    from context.context_manager import ContextManager
    from llm.service import LLMService
    from settings.settings_manager import SettingsManager
    r = LLMTestResult(model=getattr(provider, "model", ""), provider=getattr(provider, "name", ""))
    t0 = time.perf_counter()
    try:
        if not provider.is_available():
            raise LLMNotRunning()
        if not provider.has_model(provider.model):
            raise LLMModelMissing(f"Model '{provider.model}' is not installed.")
        r.connection = True
        t = time.perf_counter()
        plain = provider.generate([{"role": "user", "content": TEST_PROMPT}], temperature=0.0, max_tokens=80).strip()
        r.steps["generation_s"] = round(time.perf_counter() - t, 2)
        if not plain:
            raise LLMBadResponse("The model returned an empty completion for a trivial prompt.")
        r.generation, r.response = True, plain
        if settings is None:
            from database.sqlite import Database
            settings = SettingsManager(Database(":memory:"))
        svc = LLMService(provider, settings, ContextManager())
        t = time.perf_counter()
        res = svc.answer_stream("What is Python?", lambda q: None, lambda d: None, Event())
        r.steps["structured_s"] = round(time.perf_counter() - t, 2)
        r.parsing = bool(res.answer)
        r.answer_valid = bool(res.valid)
        r.response = res.answer or plain
        if not r.answer_valid:
            raise AnswerRejected("The structured answer failed validation: " + "; ".join((res.validation or {}).get("reasons", [])))
    except Exception as e:
        r.error = (e.user_message if isinstance(e, AppError) else f"{type(e).__name__}: {e}").replace("\n\n", " ")
        r.recommendation = recommend_action(e, settings)
        from applog import log_event
        log_event("llm", "test_llm failed", level=40, exc_info=True, error=r.error)
    r.latency = time.perf_counter() - t0
    return r
