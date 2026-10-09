"""Lightweight, offline hybrid retrieval: BM25 (keyword) + dense vectors, fused with Reciprocal Rank Fusion.

* Tokenizer keeps technology names intact (C++, C#, .NET, Node.js, CI/CD ...).
* `HashingEmbedder` needs no model and no network: signed feature hashing of stemmed words, word bigrams and
  character trigrams (robust to speech-recognition spelling noise). `OllamaEmbedder` is used instead when the user
  configures an embedding model (Settings → Question Detection → embedding model).
* `HybridSearcher` is in-memory and rebuilt cheaply from stored chunks, so a query costs well under a few ms for
  thousands of chunks.
"""
from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

STOPWORDS = frozenset("""a an the of to in on at by for from with without and or but if then so as is are was were be been
being am do does did done have has had having i me my we us our you your he she it its they them their this that these those
there here what which who whom whose when where why how can could would should will shall may might must not no yes about
into over under again also just very more most some any each other than too such own same only up down out off""".split())

_SPECIAL = [
    (re.compile(r"c\+\+", re.I), " cpp "), (re.compile(r"c#", re.I), " csharp "), (re.compile(r"f#", re.I), " fsharp "),
    (re.compile(r"\.net\b", re.I), " dotnet "), (re.compile(r"node\.?js", re.I), " nodejs "),
    (re.compile(r"next\.?js", re.I), " nextjs "), (re.compile(r"vue\.?js", re.I), " vuejs "),
    (re.compile(r"react\.?js", re.I), " react "), (re.compile(r"ci\s*/\s*cd", re.I), " cicd "),
    (re.compile(r"\bk8s\b", re.I), " kubernetes "), (re.compile(r"\bml\b", re.I), " machine learning "),
    (re.compile(r"\bai\b", re.I), " ai "), (re.compile(r"scikit[- ]learn", re.I), " sklearn "),
]
_WORD = re.compile(r"[a-z0-9]+")


_PROTECT = frozenset("nodejs nextjs vuejs cicd sklearn dotnet cpp csharp fsharp kubernetes".split())


def stem(w: str) -> str:
    if len(w) <= 4 or w.isdigit() or w in _PROTECT:
        return w
    for suf in ("ization", "ations", "ation", "ingly", "ing", "edly", "ies", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return w


def tokenize(text: str, *, keep_stop: bool = False) -> list[str]:
    t = text.lower()
    for pat, rep in _SPECIAL:
        t = pat.sub(rep, t)
    out = []
    for w in _WORD.findall(t):
        if not keep_stop and w in STOPWORDS:
            continue
        out.append(stem(w))
    return out


# ------------------------------------------------------------------------------------------------ embedders
class Embedder(Protocol):
    signature: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    signature = "hash512-v1"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.signature = f"hash{dim}-v1"

    @staticmethod
    def _h(s: str) -> int:
        return zlib.crc32(s.encode("utf-8"))

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        toks = tokenize(text)
        for t, c in Counter(toks).items():
            h = self._h(t)
            v[h % self.dim] += (1.0 + math.log(c)) * (1 if (h >> 17) & 1 else -1)
            if len(t) >= 5:                                   # char trigrams: tolerate "tensor flow" / typos
                for i in range(len(t) - 2):
                    g = self._h("#" + t[i:i + 3])
                    v[g % self.dim] += 0.25 * (1 if (g >> 17) & 1 else -1)
        for a, b in zip(toks, toks[1:]):
            h = self._h(a + "_" + b)
            v[h % self.dim] += 0.7 * (1 if (h >> 17) & 1 else -1)
        n = float(np.linalg.norm(v))
        return v / n if n else v


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    """Dense embeddings from a local Ollama model (e.g. nomic-embed-text). Falls back is the caller's job."""

    def __init__(self, provider, model: str, cache: int = 512):
        self.provider, self.model = provider, model
        self.signature = f"ollama:{model}"
        self.dim = 0
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache

    def embed(self, text: str) -> np.ndarray:
        key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            vec = np.asarray(self.provider.embed(text, self.model), np.float32)
        except Exception as e:
            raise EmbeddingError(f"embedding model '{self.model}' failed: {e}") from e
        n = float(np.linalg.norm(vec))
        vec = vec / n if n else vec
        self.dim = len(vec)
        self._cache[key] = vec
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vec


def make_embedder(settings=None, provider=None) -> Embedder:
    """Ollama embeddings if the user configured a model and the provider supports it, else the offline hasher."""
    model = (settings.get("question.embedding_model") if settings is not None else "") or ""
    if model and provider is not None and hasattr(provider, "embed"):
        return OllamaEmbedder(provider, model)
    return HashingEmbedder()


# ------------------------------------------------------------------------------------------------ search
@dataclass
class SearchItem:
    id: str
    text: str
    vector: np.ndarray | None = None
    boost_text: str = ""            # e.g. the stored question of a Q&A chunk / section title: counts double in BM25


@dataclass
class SearchHit:
    id: str
    score: float                    # fused, normalised to 0..1
    bm25: float = 0.0
    cosine: float = 0.0
    bm25_rank: int | None = None
    vec_rank: int | None = None
    matched_terms: list[str] = field(default_factory=list)


class HybridSearcher:
    def __init__(self, embedder: Embedder | None = None, rrf_k: int = 60, vec_min: float = 0.18,
                 k1: float = 1.5, b: float = 0.75):
        self.embedder = embedder or HashingEmbedder()
        self.rrf_k, self.vec_min, self.k1, self.b = rrf_k, vec_min, k1, b
        self._ids: list[str] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []
        self._avg_len = 0.0
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, items: list[SearchItem]) -> "HybridSearcher":
        self._ids = [i.id for i in items]
        self._postings, self._doc_len = {}, []
        rows = []
        for idx, it in enumerate(items):
            toks = tokenize(it.boost_text) * 2 + tokenize(it.text)
            self._doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self._postings.setdefault(term, []).append((idx, tf))
            vec = it.vector if it.vector is not None else self.embedder.embed(it.boost_text + "\n" + it.text if it.boost_text else it.text)
            rows.append(vec)
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._matrix = np.vstack(rows).astype(np.float32) if rows else None
        return self

    def _bm25(self, qtoks: list[str]) -> dict[int, float]:
        n, scores = len(self._ids), {}
        for term in set(qtoks):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for idx, tf in plist:
                dl = self._doc_len[idx]
                scores[idx] = scores.get(idx, 0.0) + idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1)))
        return scores

    def search(self, query: str, top_k: int = 5, min_score: float = 0.25, query_vector: np.ndarray | None = None
               ) -> list[SearchHit]:
        if not self._ids or not query.strip():
            return []
        qtoks = tokenize(query)
        bm = self._bm25(qtoks)
        bm_rank = {idx: r for r, (idx, _) in enumerate(sorted(bm.items(), key=lambda kv: -kv[1]), 1)}
        cos: dict[int, float] = {}
        if self._matrix is not None:
            qv = query_vector if query_vector is not None else self.embedder.embed(query)
            if qv.shape[0] == self._matrix.shape[1] and float(np.linalg.norm(qv)) > 0:
                sims = self._matrix @ qv
                for idx in np.argsort(-sims)[: max(top_k * 4, 20)]:
                    if sims[idx] >= self.vec_min:
                        cos[int(idx)] = float(sims[idx])
        vec_rank = {idx: r for r, (idx, _) in enumerate(sorted(cos.items(), key=lambda kv: -kv[1]), 1)}
        best = 2.0 / (self.rrf_k + 1)
        hits = []
        for idx in set(bm_rank) | set(vec_rank):
            fused = sum(1.0 / (self.rrf_k + r) for r in (bm_rank.get(idx), vec_rank.get(idx)) if r)
            hits.append(SearchHit(self._ids[idx], fused / best, bm.get(idx, 0.0), cos.get(idx, 0.0),
                                  bm_rank.get(idx), vec_rank.get(idx),
                                  [t for t in dict.fromkeys(qtoks) if any(i == idx for i, _ in self._postings.get(t, ()))]))
        hits.sort(key=lambda h: -h.score)
        return [h for h in hits if h.score >= min_score][:top_k]
