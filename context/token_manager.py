"""Token estimation and budget-aware prompt planning.

No tokenizer dependency: tokens are over-estimated (max of chars/3.6 and words*1.35) so the prompt stays inside the
model's window. When everything does not fit, content is removed in this order — never the question itself:

    1. old / irrelevant conversation turns (lowest relevance + recency first)
    2. lowest-ranked knowledge-base chunks
    3. oldest remembered facts
    4. lowest-ranked resume entries (the best one is kept)
    5. finally the remaining chunks are trimmed at sentence boundaries
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

MESSAGE_OVERHEAD = 6      # role markers / separators per message
_WORD = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(math.ceil(len(text) / 3.6), math.ceil(len(_WORD.findall(text)) * 1.35)) + 1


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Cut `text` to roughly `max_tokens`, preferring a sentence boundary; adds an ellipsis when cut."""
    if estimate_tokens(text) <= max_tokens:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences:
        cand = (out + " " + s).strip()
        if estimate_tokens(cand) > max_tokens:
            break
        out = cand
    if not out:  # one huge sentence: hard cut on a word boundary
        words, out = text.split(), ""
        for w in words:
            cand = (out + " " + w).strip()
            if estimate_tokens(cand) > max_tokens:
                break
            out = cand
    return out.rstrip() + " …"


_TOK = re.compile(r"[a-z0-9+#]+")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were", "you", "your", "i", "we",
         "it", "that", "this", "for", "with", "what", "how", "why", "do", "did", "does", "can", "me", "us", "about"}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


@dataclass
class PromptPlan:
    history: list[tuple[str, str]] = field(default_factory=list)
    kb: list = field(default_factory=list)            # retrieval chunks (objects with .text / .label)
    resume: list = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    memory_text: str = ""
    dropped: dict = field(default_factory=dict)       # category -> count removed
    trimmed: int = 0
    total_tokens: int = 0
    available_tokens: int = 0
    question_tokens: int = 0
    over_budget: bool = False                         # even the bare question + system prompt exceed the window

    def summary(self) -> dict:
        return {"history": len(self.history), "kb": len(self.kb), "resume": len(self.resume), "facts": len(self.facts),
                "dropped": dict(self.dropped), "trimmed": self.trimmed, "total_tokens": self.total_tokens,
                "available_tokens": self.available_tokens, "question_tokens": self.question_tokens,
                "over_budget": self.over_budget}


class ContextBudgeter:
    """Decide what fits. `window` = the model's context size, `reserve_output` = tokens kept free for the answer."""

    def __init__(self, window: int = 8192, reserve_output: int = 600, safety: float = 0.92):
        self.window, self.reserve_output, self.safety = window, reserve_output, safety

    def plan(self, *, question: str, system_tokens: int, history: list[tuple[str, str]] | None = None,
             kb: list | None = None, resume: list | None = None, facts: list[str] | None = None,
             memory_text: str = "") -> PromptPlan:
        history, kb, resume = list(history or []), list(kb or []), list(resume or [])
        facts = list(facts or [])
        q_tok = estimate_tokens(question) + MESSAGE_OVERHEAD
        avail = int(self.window * self.safety) - self.reserve_output - system_tokens - q_tok - 40
        plan = PromptPlan(question_tokens=q_tok, available_tokens=max(avail, 0))
        if avail <= 0:                                       # the question alone is (nearly) the whole window
            plan.over_budget = avail < -self.reserve_output // 2
            plan.dropped = {"history": len(history), "kb": len(kb), "resume": len(resume), "facts": len(facts)}
            plan.total_tokens = q_tok + system_tokens
            return plan

        qtok = _content_tokens(question)

        def hist_cost(h):  return estimate_tokens(h[0]) + estimate_tokens(h[1]) + 2 * MESSAGE_OVERHEAD
        def chunk_cost(c): return estimate_tokens(getattr(c, "text", str(c))) + estimate_tokens(getattr(c, "label", "")) + 8
        def fact_cost(f):  return estimate_tokens(f) + 2

        def total() -> int:
            return (sum(map(hist_cost, history)) + sum(map(chunk_cost, kb)) + sum(map(chunk_cost, resume))
                    + sum(map(fact_cost, facts)) + estimate_tokens(memory_text))

        n_hist = len(history)

        def hist_score(i: int) -> float:                     # higher = more worth keeping
            rel = len(qtok & _content_tokens(history[i][0] + " " + history[i][1])) / (len(qtok) or 1)
            return 0.8 * rel + 0.2 * (i + 1) / n_hist

        drop = {"history": 0, "kb": 0, "resume": 0, "facts": 0}
        while total() > avail and history:
            history.pop(min(range(len(history)), key=hist_score)); drop["history"] += 1
            n_hist = len(history) or 1
        while total() > avail and kb:
            kb.pop(); drop["kb"] += 1                        # lists arrive ranked best-first
        while total() > avail and facts:
            facts.pop(0); drop["facts"] += 1                 # oldest first
        while total() > avail and len(resume) > 1:
            resume.pop(); drop["resume"] += 1
        if total() > avail and memory_text:
            memory_text = trim_to_tokens(memory_text, max(60, estimate_tokens(memory_text) // 2))
        if total() > avail and resume:                       # trim what is left, longest first
            for c in sorted(resume, key=chunk_cost, reverse=True):
                over = total() - avail
                if over <= 0:
                    break
                keep = max(60, estimate_tokens(c.text) - over)
                c.text = trim_to_tokens(c.text, keep); plan.trimmed += 1
        plan.history, plan.kb, plan.resume, plan.facts, plan.memory_text = history, kb, resume, facts, memory_text
        plan.dropped = {k: v for k, v in drop.items() if v}
        plan.total_tokens = total() + q_tok + system_tokens
        return plan
