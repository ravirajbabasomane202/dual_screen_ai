"""Ignore a question that was already handled within the timeout (normalised text + similarity)."""
import difflib
import math
import re
import time
from typing import Callable

_NOISE = re.compile(r"\b(um+|uh+|like|so|basically|actually|you know|please|okay|ok|well)\b")


def normalize(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", text.lower())
    t = _NOISE.sub(" ", t)
    return " ".join(t.split())


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb)
    return max(jaccard, difflib.SequenceMatcher(None, a, b).ratio())


def _cos(u, v) -> float:
    d = sum(x * y for x, y in zip(u, v))
    n = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    return d / n if n else 0.0


class DuplicateFilter:
    def __init__(self, timeout: float = 30.0, text_threshold: float = 0.8, embed_threshold: float = 0.92,
                 embed_fn: Callable[[str], list[float]] | None = None, clock: Callable[[], float] = time.monotonic):
        self.timeout, self.text_threshold, self.embed_threshold = timeout, text_threshold, embed_threshold
        self.embed_fn, self.clock = embed_fn, clock
        self._recent: list[tuple[float, str, list[float] | None]] = []

    def _prune(self) -> None:
        now = self.clock()
        self._recent = [r for r in self._recent if now - r[0] <= self.timeout]

    def is_duplicate(self, text: str) -> bool:
        self._prune()
        emb = None
        for _, old, old_emb in self._recent:
            if similarity(text, old) >= self.text_threshold:
                return True
            if self.embed_fn:
                try:
                    emb = emb or self.embed_fn(text)
                    if old_emb and _cos(emb, old_emb) >= self.embed_threshold:
                        return True
                except Exception:
                    self.embed_fn = None  # embeddings unavailable; fall back to text only
        return False

    def register(self, text: str) -> None:
        emb = None
        if self.embed_fn:
            try:
                emb = self.embed_fn(text)
            except Exception:
                self.embed_fn = None
        self._recent.append((self.clock(), text, emb))

    def clear(self) -> None:
        self._recent.clear()
