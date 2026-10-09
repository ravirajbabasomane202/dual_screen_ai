"""Structured logging (named `applog` because a top-level `logging` package would shadow the standard library).

Files written to the log directory:
    application.log          general application messages (plain text)
    llm.log                  one JSON line per LLM request (+ a readable block when debug mode is on)
    question_detection.log   one JSON line per transcript segment that reached the detector
    retrieval.log            resume / PDF knowledge-base retrieval
    errors.log               every ERROR (with stack traces) from any component
"""
from applog.structured import (CHANNELS, LogConfig, configure, format_llm_block, get_log_dir, is_debug, log_event,
                               redact, scrub_pii)

__all__ = ["CHANNELS", "LogConfig", "configure", "format_llm_block", "get_log_dir", "is_debug", "log_event",
           "redact", "scrub_pii"]
