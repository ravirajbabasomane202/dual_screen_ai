"""Combines resume retrieval and PDF knowledge-base retrieval for one question."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from applog import log_event, redact


@dataclass
class RetrievalResult:
    resume: list = field(default_factory=list)
    kb: list = field(default_factory=list)
    query: str = ""
    seconds: float = 0.0
    ids: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.resume and not self.kb


class Retriever:
    def __init__(self, kb_index=None, resume_retriever=None, settings=None, memory=None):
        self.kb, self.resume, self.s, self.memory = kb_index, resume_retriever, settings, memory

    def _g(self, key, default):
        return self.s.get(key, default) if self.s is not None else default

    def retrieve(self, question: str, *, hint: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        q = (question + " " + hint).strip() if hint else question
        res = RetrievalResult(query=q)
        min_score = float(self._g("retrieval.min_score", 0.25))
        if self.resume is not None and self._g("retrieval.use_resume", True):
            try:
                res.resume = list(self.resume.search(q, int(self._g("retrieval.resume_top_k", 3)), min_score))
            except Exception as e:
                log_event("retrieval", "resume retrieval failed", level=40, exc_info=True, error=str(e))
        if self.kb is not None and self._g("retrieval.use_knowledge_base", True):
            try:
                res.kb = list(self.kb.search(question if not hint else q, int(self._g("retrieval.top_k", 4)), min_score))
            except Exception as e:
                log_event("retrieval", "kb retrieval failed", level=40, exc_info=True, error=str(e))
        seen, ded = set(), []
        for h in res.kb:
            key = " ".join(h.text.split())[:120]
            if key not in seen:
                seen.add(key); ded.append(h)
        res.kb = ded
        res.ids = [c.id for c in res.resume] + [h.chunk_id for h in res.kb]
        res.seconds = time.perf_counter() - t0
        log_event("retrieval", "retrieve", query=redact(q), resume_ids=[c.id for c in res.resume],
                  kb_ids=[h.chunk_id for h in res.kb], ms=round(res.seconds * 1000, 2))
        return res
