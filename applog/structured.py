"""Structured, privacy-aware logging.

* `log_event(channel, message, **fields)` writes one JSON line to the channel's file.
* Free text that can contain private content (questions, transcripts, prompts, raw model output) goes through
  `redact()`: with debug mode OFF only its length and a short hash are logged; with debug mode ON the text is logged
  (e-mail addresses / phone numbers / long digit runs are always scrubbed).
* `configure()` is idempotent and can be called again whenever the user toggles debug mode in Settings.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

CHANNELS = {
    "llm": "llm.log",
    "question": "question_detection.log",
    "retrieval": "retrieval.log",
}
APP_LOG, ERROR_LOG = "application.log", "errors.log"
MAX_TEXT = 6000          # characters of free text kept per field, even in debug mode
_TAG = "_dsai_handler"

_state = {"debug": False, "dir": None}
_lock = threading.Lock()

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
_LONGNUM = re.compile(r"\b\d{9,}\b")


@dataclass
class LogConfig:
    directory: Path
    debug: bool = False
    level: str = "INFO"


def is_debug() -> bool:
    return bool(_state["debug"])


def get_log_dir() -> Path | None:
    return _state["dir"]


def scrub_pii(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONGNUM.sub("[number]", text)


def redact(text, full: bool | None = None):
    """Return text safe for the logs: the (scrubbed) text in debug mode, otherwise `<N chars sha1=…>`."""
    if text is None:
        return None
    text = str(text)
    if (is_debug() if full is None else full):
        t = scrub_pii(text)
        return t if len(t) <= MAX_TEXT else t[:MAX_TEXT] + f"…[+{len(t) - MAX_TEXT} chars]"
    return f"<{len(text)} chars sha1={hashlib.sha1(text.encode('utf-8', 'ignore')).hexdigest()[:8]}>"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        pretty = getattr(record, "pretty", None)
        if pretty:
            return pretty
        base = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "channel": record.name.split(".")[-1],
            "msg": record.getMessage(),
        }
        base.update(getattr(record, "event", None) or {})
        if record.exc_info:
            base["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False, default=str)


class _NotChannel(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("dsai.")


def _file_handler(path: Path, fmt: logging.Formatter, level=logging.DEBUG) -> logging.Handler:
    h = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(fmt)
    h.setLevel(level)
    setattr(h, _TAG, True)
    return h


def _strip(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if getattr(h, _TAG, False):
            logger.removeHandler(h)
            h.close()


def configure(directory: str | Path, debug: bool = False, level: str = "INFO") -> LogConfig:
    """(Re)configure all log files. Safe to call repeatedly."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    lvl = logging.DEBUG if debug else getattr(logging, str(level).upper(), logging.INFO)
    with _lock:
        _state["debug"], _state["dir"] = bool(debug), d
        root = logging.getLogger()
        _strip(root)
        root.setLevel(min(lvl, logging.INFO) if not debug else logging.DEBUG)
        plain = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        app_h = _file_handler(d / APP_LOG, plain, lvl)
        app_h.addFilter(_NotChannel())
        err_h = _file_handler(d / ERROR_LOG, JsonFormatter(), logging.ERROR)
        root.addHandler(app_h)
        root.addHandler(err_h)
        for ch, fname in CHANNELS.items():
            lg = logging.getLogger(f"dsai.{ch}")
            _strip(lg)
            lg.setLevel(logging.DEBUG if debug else logging.INFO)
            lg.propagate = True            # so errors.log also sees ERROR records
            lg.addHandler(_file_handler(d / fname, JsonFormatter()))
    return LogConfig(d, debug, str(level))


def log_event(channel: str, message: str, level: int = logging.INFO, exc_info=None, **fields) -> None:
    """Write a structured record to `channel` ('llm' | 'question' | 'retrieval'). Never raises."""
    try:
        logging.getLogger(f"dsai.{channel}").log(level, message, exc_info=exc_info, extra={"event": fields})
    except Exception:  # logging must never take the pipeline down
        print(f"[log failure] {channel}: {message}", file=sys.stderr)


def format_llm_block(t: dict) -> str:
    """Human-readable block for debug mode, e.g.:

        [LLM REQUEST] ... [LLM RESPONSE] ... [VALIDATION] ... [LATENCY] ...
    """
    lines = [
        "", "[LLM REQUEST]",
        f"Request ID: {t.get('request_id')}",
        f"Model: {t.get('model')}", f"Provider: {t.get('provider')}",
        f"Question: {t.get('question')}",
        f"Context tokens: {t.get('context_tokens', t.get('context_length'))}",
        f"Retrieved chunks: {t.get('retrieved_context_count', 0)}",
        f"Temperature: {t.get('temperature')}   Max tokens: {t.get('max_tokens')}   Timeout: {t.get('timeout')}",
    ]
    if t.get("error"):
        lines += ["", "[LLM ERROR]", f"Request ID: {t.get('request_id')}", f"Error: {t.get('error')}"]
        if t.get("raw_response") is not None:
            lines += ["Raw response:", str(t.get("raw_response"))]
        if t.get("stack_trace"):
            lines += ["", "Traceback:", str(t.get("stack_trace"))]
    else:
        lines += ["", "[LLM RESPONSE]", str(t.get("raw_response")),
                  "", "[VALIDATION]", f"Valid answer: {'YES' if (t.get('validation_result') or {}).get('valid') else 'NO'}"]
        reasons = (t.get("validation_result") or {}).get("reasons")
        if reasons:
            lines.append(f"Reasons: {', '.join(reasons)}")
    lines += ["", "[LATENCY]", f"{t.get('latency', 0):.2f} seconds", ""]
    return scrub_pii("\n".join(lines))
