"""One LLMTrace per request: collected while generating, written to llm.log (+ readable block in debug mode)."""
from __future__ import annotations

import logging
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field

from applog import format_llm_block, is_debug, log_event, redact


@dataclass
class LLMTrace:
    kind: str = "answer"                   # answer | classify | test
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    provider: str = ""
    model_path: str = ""
    prompt_length: int = 0                 # characters
    context_length: int = 0                # estimated prompt tokens
    context_window: int = 0
    question: str = ""
    retrieved_context_count: int = 0
    retrieved_context_ids: list = field(default_factory=list)
    generation_parameters: dict = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 0
    timeout: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    latency: float = 0.0
    raw_response: str | None = None
    parsed_response: dict | None = None
    validation_result: dict | None = None
    provider_stats: dict = field(default_factory=dict)
    attempts: int = 1
    plan: dict = field(default_factory=dict)
    prompt: list | None = None             # full messages — only written in debug mode
    error: str | None = None
    stack_trace: str | None = None

    def fail(self, exc: BaseException) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        self.stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def finish(self) -> None:
        self.end_time = time.time()
        self.latency = self.end_time - self.start_time
        d = asdict(self)
        d["timestamp"] = _iso(self.timestamp)
        d["question"] = redact(self.question)
        d["raw_response"] = redact(self.raw_response) if self.raw_response is not None else None
        d["parsed_response"] = _redact_parsed(self.parsed_response)
        d["prompt"] = [{"role": m["role"], "content": redact(m["content"])} for m in self.prompt] if (self.prompt and is_debug()) else None
        if not is_debug():
            d.pop("prompt", None)
        stack = d.pop("stack_trace", None)
        level = logging.ERROR if self.error else logging.INFO
        log_event("llm", f"llm {self.kind} {'FAILED' if self.error else 'ok'}", level=level, stack_trace=stack, **d)
        if is_debug():
            import logging as _l
            _l.getLogger("dsai.llm").debug("block", extra={"pretty": format_llm_block(
                {**d, "stack_trace": stack, "raw_response": self.raw_response, "question": self.question})})


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


def _redact_parsed(p: dict | None):
    if not p:
        return p
    return {k: (redact(v) if isinstance(v, str) and k in ("question", "answer", "clean_question", "intent") else v)
            for k, v in p.items()}
