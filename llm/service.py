"""LLM orchestration: cheap classification and the structured answer pipeline.

answer_stream():  budget the prompt -> build messages -> stream JSON -> guard the first words of the answer
                  -> parse -> validate (reject "The speaker is asking…") -> regenerate with a repair note if needed.
Every request is traced (llm.log); nothing is swallowed — failures are logged with stack traces and re-raised.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Callable

from app.errors import AnswerRejected, LLMBadResponse, LLMContextOverflow
from context.token_manager import ContextBudgeter, estimate_tokens
from llm.base import LLMProvider
from llm.prompt import (CLASSIFY_SCHEMA, LENGTH_RULES, SCHEMA, SYSTEM_PROMPT, JSONFieldStreamer, LLMResult, ParseError,
                        build_classify_messages, build_messages, max_tokens_for, parse_llm_json)
from llm.trace import LLMTrace
from llm.validation import summary_prefix, validate_answer

log = logging.getLogger(__name__)
_IS_Q = re.compile(r'"is_question"\s*:\s*(true|false)')
GUARD_CHARS = 60          # answer text held back until this many characters (or a full sentence) were checked


@dataclass
class Extras:
    """Retrieved material for one question (duck-typed with knowledge.retriever.RetrievalResult)."""
    resume: list = field(default_factory=list)
    kb: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    length: str = ""               # one-off answer length ("Shorter" / "Longer" buttons); "" = the configured length


class LLMService:
    def __init__(self, provider: LLMProvider, settings, context):
        self.provider, self.s, self.context = provider, settings, context
        self.last_debug: dict = {}

    # ------------------------------------------------------------------------------------------ helpers
    def _g(self, key, default=None):
        v = self.s.get(key, default)
        return default if v is None else v

    def _window(self) -> int:
        w = int(self._g("ai.context_window", 8192)) or 4096
        real = None
        try:
            real = self.provider.context_window()
        except Exception:
            pass
        return min(w, real) if real else w

    def _trace(self, kind: str, question: str, msgs, max_tokens: int, temperature: float) -> LLMTrace:
        p = self.provider
        return LLMTrace(kind=kind, model=getattr(p, "model", ""), provider=getattr(p, "name", ""),
                        model_path=getattr(p, "model_path", "") or getattr(p, "host", ""),
                        prompt_length=sum(len(m["content"]) for m in msgs),
                        context_length=sum(estimate_tokens(m["content"]) for m in msgs), context_window=self._window(),
                        question=question, max_tokens=max_tokens, temperature=temperature,
                        timeout=float(self._g("ai.timeout", 60)), prompt=msgs,
                        generation_parameters={"max_tokens": max_tokens, "temperature": temperature,
                                               "json_schema": kind == "answer"})

    # ------------------------------------------------------------------------------------------ classify
    def classify(self, text: str, cancel: Event | None = None) -> LLMResult:
        msgs = build_classify_messages(text)
        tr = self._trace("classify", text, msgs, 120, 0.0)
        try:
            raw = self.provider.generate(msgs, temperature=0.0, max_tokens=120, json_schema=CLASSIFY_SCHEMA, cancel=cancel)
            tr.raw_response = raw
            res = parse_llm_json(raw)
            tr.parsed_response = {"is_question": res.is_question, "confidence": res.confidence,
                                  "clean_question": res.clean_question}
            tr.provider_stats = dict(getattr(self.provider, "last_stats", {}) or {})
            return res
        except BaseException as e:
            tr.fail(e)
            raise
        finally:
            tr.finish()

    # ------------------------------------------------------------------------------------------ answer
    def answer_stream(self, text: str, on_clean: Callable[[str], None], on_delta: Callable[[str], None],
                      cancel: Event, extras: Extras | None = None,
                      on_reset: Callable[[], None] | None = None) -> LLMResult:
        extras = extras or Extras()
        length = extras.length or self._g("ai.answer_length", "Short")
        style = str(self._g("ai.style_preset", "Concise"))
        strict = bool(self._g("ai.strict_validation", True))
        max_attempts = 1 + (int(self._g("ai.max_retries", 2)) if strict else 0)
        temperature = float(self._g("ai.temperature", 0.2))
        mt = max_tokens_for(length, text, int(self._g("ai.max_tokens", 0) or 0))
        sys_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(LENGTH_RULES.get(length, LENGTH_RULES["Short"])[0]) + 12
        shrink, overflow_retries = 1.0, 0
        repair, streamed, clean_sent = "", False, [False]
        last_res, last_raw, last_vr, history_dbg = None, "", None, []

        def clean_once(q: str) -> None:
            if not clean_sent[0]:
                clean_sent[0] = True
                on_clean(q)

        attempt = 0
        while attempt < max_attempts:
            budgeter = ContextBudgeter(int(self._window() * shrink), reserve_output=mt)
            plan = budgeter.plan(question=text, system_tokens=sys_tokens, history=self.context.pairs(),
                                 kb=list(extras.kb), resume=list(extras.resume), facts=self.context.facts(),
                                 memory_text=self.context.memory_text())
            msgs = build_messages(text, length, plan.history, resume_chunks=plan.resume, kb_chunks=plan.kb,
                                  facts=plan.facts, memory_text=plan.memory_text, repair=repair, style=style)
            tr = self._trace("answer", text, msgs, mt, temperature)
            tr.attempts, tr.plan = attempt + 1, plan.summary()
            tr.retrieved_context_count = len(plan.resume) + len(plan.kb)
            tr.retrieved_context_ids = list(extras.ids)
            try:
                res, raw, stop = self._generate(msgs, text, mt, temperature, cancel, on_delta, clean_once, strict, streamed)
                streamed = streamed or bool(res.answer and stop == "ok")
            except LLMContextOverflow as e:
                tr.fail(e); tr.finish()
                if overflow_retries >= 1:
                    raise
                overflow_retries, shrink = overflow_retries + 1, shrink * 0.55
                log.warning("context overflow; retrying with a smaller prompt budget (%.0f%%)", shrink * 100)
                continue
            except BaseException as e:
                tr.fail(e); tr.finish()
                raise
            tr.raw_response = raw
            tr.provider_stats = dict(getattr(self.provider, "last_stats", {}) or {})
            last_raw = raw

            if stop == "not_question":
                res.valid = True
                tr.parsed_response = {"is_question": False}; tr.finish()
                return res
            if cancel.is_set() and not res.answer:
                tr.finish(); return res
            if stop == "length":                      # generation hit max_tokens before the JSON closed: give it room
                mt = int(mt * 1.6)
                tr.error = "generation truncated (finish_reason=length); retrying with more tokens"; tr.finish()
                if streamed and on_reset:
                    on_reset(); streamed = False
                attempt += 1
                continue
            if stop == "parse_error":
                tr.error = "malformed JSON / empty response"; tr.finish()
                repair = "was not valid JSON"
                if streamed and on_reset:
                    on_reset(); streamed = False
                attempt += 1
                continue

            sources = "\n".join(getattr(c, "text", "") for c in list(plan.resume) + list(plan.kb))
            vr = validate_answer(res.clean_question or text, res.answer, intent=res.intent, grounding=res.grounding,
                                 sources_text=sources + " " + self.context.memory_text() + " " + " ".join(plan.facts))
            res.validation, res.attempts = vr.to_dict(), attempt + 1
            tr.parsed_response = {"question": res.clean_question, "intent": res.intent, "grounding": res.grounding,
                                  "answer": res.answer, "confidence": res.confidence}
            tr.validation_result = vr.to_dict()
            last_res, last_vr = res, vr
            if vr.valid or not strict:
                res.valid = vr.valid or not strict
                tr.finish()
                self.last_debug = {"messages": msgs, "plan": plan.summary(), "raw": raw, "validation": vr.to_dict(),
                                   "attempts": attempt + 1, "stats": tr.provider_stats, "request_id": tr.request_id}
                return res
            tr.error = "answer rejected: " + "; ".join(vr.reasons)
            tr.finish()
            repair = ("restated or described the question" if vr.kind in ("summary", "restatement")
                      else "was empty or unusable")
            if streamed and on_reset:
                on_reset(); streamed = False
            attempt += 1

        # ---- every attempt failed
        self.last_debug = {"messages": [], "raw": last_raw, "validation": last_vr.to_dict() if last_vr else None,
                           "attempts": max_attempts}
        if last_res is not None and last_vr is not None and not last_vr.valid and last_vr.kind != "empty":
            last_res.valid = False
            log.error("answer rejected after %d attempts: %s", max_attempts, last_vr.reasons)
            return last_res
        if not last_raw.strip():
            raise LLMBadResponse("The model returned an empty response.\n\nCheck that the model is installed and loaded "
                                 "(Settings → Debug → Test LLM). Details: logs/llm.log")
        plain = last_raw.strip()
        if not plain.lstrip().startswith(("{", "[")) and len(plain.split()) >= 4:     # model ignored the JSON contract
            vr = validate_answer(text, plain)
            res = LLMResult(is_question=True, confidence=0.4, clean_question=text, answer=plain, from_plain_text=True,
                            valid=vr.valid, validation=vr.to_dict(), attempts=max_attempts, grounding="unknown")
            log.warning("model ignored the JSON contract; using its plain-text reply (valid=%s)", vr.valid)
            return res
        raise LLMBadResponse(f"Malformed model output after {max_attempts} attempts. See logs/llm.log (raw response).")

    # ------------------------------------------------------------------------------------------ one generation
    def _generate(self, msgs, text, mt, temperature, cancel, on_delta, clean_once, strict, already_streamed):
        """Stream one completion. Returns (LLMResult, raw, stop) with stop in
        ok | not_question | summary_prefix | length | parse_error | cancelled."""
        clean_streamers = [JSONFieldStreamer("question"), JSONFieldStreamer("clean_question")]
        ans_s = JSONFieldStreamer("answer")
        raw: list[str] = []
        held: list[str] = []
        released, partial, stop = False, "", "ok"
        for chunk in self.provider.stream(msgs, temperature=temperature, max_tokens=mt, json_schema=SCHEMA, cancel=cancel):
            raw.append(chunk)
            for cs in clean_streamers:
                if not cs.done:
                    cs.feed(chunk)
                    if cs.done and cs.value.strip():
                        clean_once(cs.value.strip())
            if (m := _IS_Q.search("".join(raw))) and m.group(1) == "false":
                cancel.set()
                stop = "not_question"
                break
            piece = ans_s.feed(chunk)
            if not piece:
                continue
            partial += piece
            if released:
                on_delta(piece)
                continue
            held.append(piece)
            if len(partial) >= GUARD_CHARS or ans_s.done or re.search(r"[.!?]\s*$", partial):
                if strict:
                    why = summary_prefix(partial)
                    if not why and len(partial) >= 40:
                        v = validate_answer(text, partial)
                        why = None if v.valid else v.reasons[0]
                    if why:
                        stop = "summary_prefix"
                        break
                released = True
                on_delta("".join(held)); held = []
        if held and stop == "ok":                 # short answer that ended before the guard threshold
            if strict and (summary_prefix(partial)):
                stop = "summary_prefix"
            else:
                on_delta("".join(held))
        full = "".join(raw)
        stats = getattr(self.provider, "last_stats", {}) or {}
        if stop == "summary_prefix":
            return LLMResult(is_question=True, clean_question=text, answer=partial, valid=False,
                             validation={"valid": False, "reasons": ["answer began by describing the question"],
                                         "kind": "summary"}), full, "summary_prefix"
        try:
            res = parse_llm_json(full)
        except ParseError:
            if stop == "not_question":
                return LLMResult(is_question=False), full, stop
            if ans_s.value.strip():
                res = LLMResult(is_question=True, confidence=0.5, answer=ans_s.value.strip(),
                                clean_question=next((c.value.strip() for c in clean_streamers if c.value), ""))
            else:
                return LLMResult(), full, "parse_error"
        if stop == "not_question":
            res.is_question = False
            return res, full, stop
        if stats.get("finish_reason") == "length" and not ans_s.done:
            return res, full, "length"
        return res, full, "ok"
