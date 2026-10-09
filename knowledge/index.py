"""Persistent, hash-aware knowledge base: files, folders, in-app notes and Q&A, with a hybrid searcher on top.

Concurrency model (the UI and the interview pipeline must never wait for an index run):
* `_write` serialises writers (sync, add, edit, delete). It is held while parsing/embedding, NOT while searching.
* `_lock` only guards the in-memory searcher; searches take it briefly and work on a snapshot built from SQLite.
* Every document is replaced in one SQLite transaction, so cancelling or crashing between two documents leaves the
  index consistent; a document is either its old version or its new version, never half of each.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.errors import Cancelled, KnowledgeBaseError
from applog import log_event, redact
from context.token_manager import estimate_tokens
from database.repositories import KBChunkRow, KBDocRow, KBEntryRow, KnowledgeRepository
from knowledge import pdf_parser
from knowledge.chunker import Chunk, _split_long, chunk_document
from knowledge.hybrid import (EmbeddingError, HashingEmbedder, HybridSearcher, SearchItem, make_embedder)
from knowledge.pdf_parser import SUPPORTED, ParsedDocument, ParsedPage, file_sha256
from knowledge.structure import structure_text
from knowledge.pdf_parser import clean_text

QA_BONUS = 0.35                 # user-written Q&A pairs outrank ordinary chunks when they are relevant at all
UNDO_SECONDS = 600              # how long a deleted document can be restored before it is purged
NOTE_SCHEME, QA_SCHEME = "note://", "qa://"


@dataclass
class KBHit:
    chunk_id: str
    document: str
    page: int
    section: str
    kind: str
    question: str
    text: str
    score: float
    doc_path: str = ""
    source: str = "folder"
    bm25: float = 0.0
    cosine: float = 0.0
    bm25_rank: int | None = None
    vec_rank: int | None = None
    matched_terms: list = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.source in ("note", "qa"):
            return f"{self.document} ({'Q&A' if self.source == 'qa' else 'note'})"
        return f"{self.document} p.{self.page}" + (f" · {self.section}" if self.section else "")


@dataclass
class SyncReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    excluded: int = 0                                  # present in the folder but deleted by the user earlier
    failed: list = field(default_factory=list)         # (name, reason)
    no_text: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    seconds: float = 0.0
    total_chunks: int = 0
    cancelled: bool = False

    def summary(self) -> str:
        s = (f"{self.added} added, {self.updated} updated, {self.unchanged} unchanged, {self.removed} removed, "
             f"{len(self.failed)} failed — {self.total_chunks} chunks ({self.seconds:.1f}s)")
        if self.excluded:
            s += f"; {self.excluded} excluded by you"
        if self.no_text:
            s += f"; no text (scanned?): {', '.join(self.no_text)}"
        if self.failed:
            s += "; failed: " + "; ".join(f"{n}: {r}" for n, r in self.failed)
        if self.cancelled:
            s += "; cancelled before the end (finished documents are kept)"
        return s


def _embed_text(question: str, section: str, text: str) -> str:
    return (question + "\n" if question else "") + section + "\n" + text


def _yield() -> None:
    time.sleep(0)                  # lets the GUI thread run between chunks of work


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


class KnowledgeIndex:
    def __init__(self, db, embedder=None, provider=None, settings=None):
        self.repo = KnowledgeRepository(db)
        self.settings = settings
        self.embedder = embedder or make_embedder(settings, provider)
        self._fallback = HashingEmbedder()
        self._lock = threading.RLock()            # guards the searcher snapshot
        self._write = threading.RLock()           # serialises writers
        self._searcher: HybridSearcher | None = None
        self._chunks: dict = {}
        self._sources: dict[str, str] = {}
        self._qemb = self.embedder
        self.warnings: list[str] = []
        self.on_changed = None                    # optional callable(); invoked after the content changed

    # ------------------------------------------------------------------ helpers
    def _invalidate(self) -> None:
        with self._lock:
            self._searcher = None
        cb = self.on_changed
        if cb:
            try:
                cb()
            except Exception:
                pass

    def _embed(self, text: str) -> tuple[np.ndarray, str]:
        try:
            return self.embedder.embed(text), self.embedder.signature
        except EmbeddingError as e:
            self.warnings.append(str(e) + " — using offline hashing embeddings")
            return self._fallback.embed(text), self._fallback.signature

    def _params(self) -> tuple[int, int]:
        g = (lambda k, d: self.settings.get(k, d)) if self.settings is not None else (lambda k, d: d)
        return int(g("knowledge.chunk_tokens", 220)), int(g("knowledge.chunk_overlap", 40))

    def _rows_for(self, path: str, name: str, chunks: list[Chunk], cancel=None) -> list[KBChunkRow]:
        """Embed chunks; chunks whose text is unchanged since the previous index reuse their stored vector."""
        sig = self.embedder.signature
        cache = {}
        for old in self.repo.chunks_of(path):
            if old.embedding is not None and old.embedder == sig:
                cache[hashlib.sha1(_embed_text(old.question, old.section, old.text).encode()).hexdigest()] = old.embedding
        rows = []
        for c in chunks:
            _check(cancel)
            et = _embed_text(c.question, c.section, c.text)
            key = hashlib.sha1(et.encode()).hexdigest()
            if key in cache:
                vec, s = cache[key], sig
            else:
                vec, s = self._embed(et)
            rows.append(KBChunkRow(c.chunk_id, path, name, c.page, c.section, c.kind, c.question, c.text, c.tokens, vec, s))
            _yield()
        return rows

    # ------------------------------------------------------------------ files
    def _index_file(self, path: Path, sha: str, source: str, cancel=None) -> int:
        old = self.repo.get_document(str(path))
        try:
            doc = pdf_parser.parse_document(path)
            _check(cancel)
            target, overlap = self._params()
            chunks = chunk_document(doc, target, overlap)
            name = old.name if (old is not None and old.name != path.name) else path.name    # keep a user's rename
            rows = self._rows_for(str(path), name, chunks, cancel)
            status = doc.status if chunks or doc.status != "ok" else "no_text"
            self.repo.replace_document(KBDocRow(str(path), name, sha, doc.size, len(doc.pages), len(rows),
                                                status, "; ".join(doc.warnings)[:500], time.time(), source), rows)
            return len(rows) if status == "ok" else -1
        except Cancelled:
            raise
        except Exception as e:
            if old is None or old.sha256 != sha:
                self.repo.replace_document(KBDocRow(str(path), old.name if old else path.name, sha, 0, 0, 0, "error",
                                                    str(e)[:500], time.time(), source), [])
            raise

    def sync_file(self, path, source: str = "file", force: bool = False, cancel=None) -> str:
        """Returns 'added' | 'updated' | 'unchanged' (raises on parse failure). A document the user deleted comes back
        when they add it again explicitly."""
        p = Path(path)
        with self._write:
            sha = file_sha256(p)
            old = self.repo.get_document(str(p))
            if old and old.deleted_at is not None:
                self.repo.undelete(str(p))
                old = self.repo.get_document(str(p))
                force = True
            if old and old.sha256 == sha and old.status != "error" and not force:
                return "unchanged"
            self._index_file(p, sha, source, cancel)
            self._invalidate()
            return "updated" if old else "added"

    def add_files(self, paths, progress_cb=None, cancel=None, force: bool = False) -> SyncReport:
        """Index individual files (kept even when no folder is configured). Failures don't stop the batch."""
        t0 = time.perf_counter()
        rep = SyncReport()
        paths = [Path(p) for p in paths]
        with self._write:
            self.warnings = []
            for i, p in enumerate(paths, 1):
                if progress_cb:
                    progress_cb(i - 1, len(paths), p.name)
                try:
                    if p.suffix.lower() not in SUPPORTED:
                        raise KnowledgeBaseError(f"{p.name}: unsupported type (use PDF, DOCX, TXT or MD)")
                    res = self.sync_file(p, "file", force=force, cancel=cancel)
                    rep.added += res == "added"
                    rep.updated += res == "updated"
                    rep.unchanged += res == "unchanged"
                    d = self.repo.get_document(str(p))
                    if d and d.status == "no_text":
                        rep.no_text.append(p.name)
                except Cancelled:
                    rep.cancelled = True
                    break
                except Exception as e:
                    rep.failed.append((p.name, getattr(e, "user_message", str(e))))
                    log_event("retrieval", "kb add file failure", level=40, file=p.name, error=str(e))
            if progress_cb and not rep.cancelled:
                progress_cb(len(paths), len(paths), "")
            rep.total_chunks = self.repo.chunk_count()
            rep.warnings = list(self.warnings)
        rep.seconds = time.perf_counter() - t0
        log_event("retrieval", "kb add files", added=rep.added, updated=rep.updated, failed=len(rep.failed))
        return rep

    def _reembed_stale(self, cancel=None) -> int:
        sig = self.embedder.signature
        stale = [c for c in self.repo.all_chunks() if c.embedder != sig]
        if not stale:
            return 0
        items = []
        for c in stale:
            _check(cancel)
            vec, s = self._embed(_embed_text(c.question, c.section, c.text))
            items.append((c.chunk_id, vec, s))
            _yield()
        self.repo.set_embeddings(items)
        self._invalidate()
        return len(stale)

    def sync_folder(self, folder, progress_cb=None, force: bool = False, cancel=None) -> SyncReport:
        t0 = time.perf_counter()
        rep = SyncReport()
        root = Path(folder)
        if not root.is_dir():
            raise ValueError(f"Not a folder: {folder}")
        with self._write:
            self.warnings = []
            files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
                           and not any(part.startswith(".") for part in p.relative_to(root).parts))
            present = {str(p) for p in files}
            try:
                prefix = str(root) + ("" if str(root).endswith(("/", "\\")) else __import__("os").sep)
                for d in self.repo.list_documents(include_deleted=True):
                    if d.source == "folder" and d.path.startswith(prefix) and d.path not in present:
                        self.repo.delete_document(d.path)
                        rep.removed += d.deleted_at is None
                for i, p in enumerate(files, 1):
                    _check(cancel)
                    if progress_cb:
                        progress_cb(i - 1, len(files), p.name)
                    try:
                        sha = file_sha256(p)
                        old = self.repo.get_document(str(p))
                        if old is not None and old.deleted_at is not None:
                            rep.excluded += 1
                            continue
                        if not force and old and old.sha256 == sha:
                            rep.unchanged += 1
                            if old.status == "no_text":
                                rep.no_text.append(p.name)
                            elif old.status == "error":
                                rep.failed.append((p.name, old.error or "previous failure (unchanged file)"))
                            continue
                        res = self._index_file(p, sha, "folder", cancel)
                        self._invalidate()
                        rep.updated += 1 if old else 0
                        rep.added += 0 if old else 1
                        if res == -1:
                            rep.no_text.append(p.name)
                    except Cancelled:
                        raise
                    except Exception as e:
                        rep.failed.append((p.name, str(e)))
                        log_event("retrieval", "kb index failure", level=40, file=p.name, error=str(e))
                self._reembed_stale(cancel)
                if progress_cb:
                    progress_cb(len(files), len(files), "")
                self.repo.set_meta("folder", str(root))
                self.repo.set_meta("last_sync", str(time.time()))
            except Cancelled:
                rep.cancelled = True
            if rep.removed:
                self._invalidate()
            self.purge_expired()
            rep.total_chunks = self.repo.chunk_count()
            rep.warnings = list(self.warnings)
        rep.seconds = time.perf_counter() - t0
        log_event("retrieval", "kb sync", **{k: getattr(rep, k) for k in ("added", "updated", "unchanged", "removed")},
                  failed=len(rep.failed), chunks=rep.total_chunks, seconds=round(rep.seconds, 2), cancelled=rep.cancelled)
        return rep

    def sync_folders(self, folders, progress_cb=None, force: bool = False, cancel=None) -> SyncReport:
        """Sync several folders and merge the reports (one folder's problem never blocks the others)."""
        total = SyncReport()
        folders = [f for f in dict.fromkeys(str(f) for f in folders if f)]
        for i, f in enumerate(folders):
            cb = None
            if progress_cb:
                cb = (lambda d, n, name, i=i: progress_cb(d, n, (f"{Path(folders[i]).name}/" if len(folders) > 1 else "") + name))
            try:
                r = self.sync_folder(f, cb, force, cancel)
            except ValueError as e:                       # a folder that no longer exists
                total.failed.append((f, str(e)))
                continue
            for k in ("added", "updated", "unchanged", "removed", "excluded"):
                setattr(total, k, getattr(total, k) + getattr(r, k))
            total.failed += r.failed
            total.no_text += r.no_text
            total.warnings += r.warnings
            total.seconds += r.seconds
            total.cancelled = total.cancelled or r.cancelled
            if r.cancelled:
                break
        total.total_chunks = self.repo.chunk_count()
        return total

    # ------------------------------------------------------------------ notes and Q&A
    def _index_entry(self, e: KBEntryRow) -> int:
        path = (QA_SCHEME if e.kind == "qa" else NOTE_SCHEME) + e.id
        name = e.title.strip() or (e.question.strip()[:70] if e.kind == "qa" else "Untitled note")
        tags = f"\nTags: {e.tags}" if e.tags.strip() else ""
        target, overlap = self._params()
        sha = hashlib.sha256(f"{e.kind}|{e.title}|{e.body}|{e.question}|{e.answer}|{e.tags}".encode()).hexdigest()
        if e.kind == "qa":
            parts = _split_long(e.answer.strip(), target * 3) or [e.answer.strip()]
            chunks = [Chunk(hashlib.sha1(f"{sha}|{i}".encode()).hexdigest()[:16], name, 1, "", "qa", e.question.strip(),
                            (part + (tags if i == len(parts) - 1 else "")).strip(), estimate_tokens(part))
                      for i, part in enumerate(parts)]
            size = len(e.answer)
        else:
            doc = ParsedDocument(name, path, sha, len(e.body))
            doc.pages.append(ParsedPage(1, structure_text(clean_text(f"# {name}\n\n{e.body.strip()}{tags}"))))
            chunks = chunk_document(doc, target, overlap)
            if not chunks and e.body.strip():                # very short note: keep it whole
                txt = (e.body.strip() + tags).strip()
                chunks = [Chunk(hashlib.sha1(f"{sha}|0".encode()).hexdigest()[:16], name, 1, name, "text", "", txt,
                                estimate_tokens(txt))]
            size = len(e.body)
        rows = self._rows_for(path, name, chunks)
        status = "ok" if rows else "no_text"
        self.repo.replace_document(KBDocRow(path, name, sha, size, 1, len(rows), status, "", time.time(),
                                            "qa" if e.kind == "qa" else "note"), rows)
        return len(rows)

    @staticmethod
    def entry_path(e: KBEntryRow | str, kind: str = "note") -> str:
        if isinstance(e, KBEntryRow):
            return (QA_SCHEME if e.kind == "qa" else NOTE_SCHEME) + e.id
        return (QA_SCHEME if kind == "qa" else NOTE_SCHEME) + e

    def add_note(self, title: str, body: str, tags: str = "") -> str:
        if not (title.strip() or body.strip()):
            raise KnowledgeBaseError("A note needs a title or some text.")
        now = time.time()
        e = KBEntryRow(uuid.uuid4().hex[:12], "note", title.strip(), body, "", "", tags.strip(), 1, now, now)
        with self._write:
            self.repo.save_entry(e)
            self._index_entry(e)
            self._invalidate()
        log_event("retrieval", "kb note added", id=e.id, chars=len(body))
        return e.id

    def add_qa(self, question: str, answer: str, tags: str = "", title: str = "") -> str:
        if not question.strip() or not answer.strip():
            raise KnowledgeBaseError("A Q&A entry needs both a question and the answer you want to give.")
        now = time.time()
        e = KBEntryRow(uuid.uuid4().hex[:12], "qa", title.strip(), "", question.strip(), answer.strip(), tags.strip(), 1, now, now)
        with self._write:
            self.repo.save_entry(e)
            self._index_entry(e)
            self._invalidate()
        log_event("retrieval", "kb qa added", id=e.id)
        return e.id

    def update_entry(self, entry_id: str, **fields) -> bool:
        """Edit a note / Q&A. Returns False (and does nothing) when nothing changed, so unchanged items are never
        re-embedded; otherwise only chunks whose text changed are re-embedded."""
        with self._write:
            e = self.repo.get_entry(entry_id)
            if e is None:
                raise KnowledgeBaseError("That entry no longer exists.")
            new = KBEntryRow(**{**e.__dict__, **{k: (v.strip() if isinstance(v, str) and k != "body" else v)
                                                   for k, v in fields.items() if k in ("title", "body", "question", "answer", "tags")}})
            if (new.title, new.body, new.question, new.answer, new.tags) == (e.title, e.body, e.question, e.answer, e.tags):
                return False
            if new.kind == "qa" and not (new.question and new.answer):
                raise KnowledgeBaseError("A Q&A entry needs both a question and an answer.")
            new.version, new.updated_at = e.version + 1, time.time()
            self.repo.save_entry(new)
            self._index_entry(new)
            self._invalidate()
        return True

    def get_entry(self, entry_id: str) -> KBEntryRow | None:
        return self.repo.get_entry(entry_id)

    def entry_for_path(self, path: str) -> KBEntryRow | None:
        return self.repo.get_entry(path.split("://", 1)[1]) if "://" in path else None

    def entries(self) -> list[KBEntryRow]:
        return self.repo.list_entries()

    # ------------------------------------------------------------------ per-document management
    def documents(self, include_deleted: bool = False) -> list[KBDocRow]:
        return self.repo.list_documents(include_deleted)

    def get_document(self, path) -> KBDocRow | None:
        return self.repo.get_document(str(path))

    def set_enabled(self, path, enabled: bool) -> None:
        with self._write:
            self.repo.set_doc_enabled(str(path), enabled)
            self._invalidate()

    def rename(self, path, name: str) -> None:
        name = name.strip()
        if not name:
            raise KnowledgeBaseError("The name cannot be empty.")
        with self._write:
            d = self.repo.get_document(str(path))
            if d is None:
                raise KnowledgeBaseError("That document no longer exists.")
            self.repo.rename_document(str(path), name)
            e = self.entry_for_path(str(path))
            if e is not None and e.kind == "note":
                self.repo.save_entry(KBEntryRow(**{**e.__dict__, "title": name, "updated_at": time.time(), "version": e.version + 1}))
            self._invalidate()

    def reindex(self, path, cancel=None) -> str:
        """Re-parse one document from its source (files) or re-chunk it (notes / Q&A)."""
        path = str(path)
        with self._write:
            d = self.repo.get_document(path)
            if d is None:
                raise KnowledgeBaseError("That document is not in the knowledge base.")
            e = self.entry_for_path(path)
            if e is not None:
                self._index_entry(e)
                self._invalidate()
                return "updated"
            p = Path(path)
            if not p.exists():
                raise KnowledgeBaseError(f"{d.name}: the file no longer exists at {p.parent}.")
            return self.sync_file(p, d.source, force=True, cancel=cancel)

    def delete_document(self, path) -> str:
        """Soft delete (Undo for UNDO_SECONDS). Returns the path to pass to undo_delete()."""
        path = str(path)
        with self._write:
            if self.repo.get_document(path) is None:
                raise KnowledgeBaseError("That document is not in the knowledge base.")
            self.repo.soft_delete(path, time.time())
            self._invalidate()
        return path

    def undo_delete(self, path) -> bool:
        with self._write:
            d = self.repo.get_document(str(path))
            if d is None or d.deleted_at is None:
                return False
            if self.repo.chunks_of(str(path)) == [] and d.source == "folder":
                return False                                  # already purged; a folder sync / add_files restores it
            self.repo.undelete(str(path))
            self._invalidate()
            return True

    def purge_expired(self, now: float | None = None, grace: float = UNDO_SECONDS) -> int:
        with self._write:
            return self.repo.purge_deleted((now or time.time()) - grace)

    def remove_document(self, path) -> None:
        """Hard delete (no undo)."""
        with self._write:
            self.repo.delete_document(str(path))
            e = self.entry_for_path(str(path))
            if e is not None:
                self.repo.delete_entry(e.id)
            self._invalidate()

    def clear(self) -> None:
        with self._write:
            self.repo.clear()
            self._invalidate()

    # ------------------------------------------------------------------ chunks
    def chunks_of(self, path) -> list[KBChunkRow]:
        return self.repo.chunks_of(str(path))

    def set_chunk_enabled(self, chunk_id: str, enabled: bool) -> None:
        with self._write:
            self.repo.set_chunk_enabled(chunk_id, enabled)
            self._invalidate()

    def preview(self, path, max_chars: int = 8000) -> str:
        out, n = [], 0
        for c in self.repo.chunks_of(str(path)):
            head = f"— p.{c.page}" + (f" · {c.section}" if c.section else "") + (f" · Q: {c.question}" if c.question else "")
            piece = f"{head}\n{c.text}"
            out.append(piece)
            n += len(piece)
            if n >= max_chars:
                out.append("… (preview truncated)")
                break
        return "\n\n".join(out) or "(no text was extracted from this document)"

    # ------------------------------------------------------------------ statistics
    def stats(self) -> dict:
        docs = self.repo.list_documents()
        return {"documents": len(docs), "disabled": sum(1 for d in docs if not d.enabled),
                "chunks": self.repo.chunk_count(), "active_chunks": self.repo.chunk_count(only_active=True),
                "folder": self.repo.get_meta("folder"), "embedder": self.embedder.signature,
                "last_sync": self.repo.get_meta("last_sync"), "index_bytes": self.repo.storage_bytes(),
                "pending_undo": len(self.repo.list_deleted()),
                "per_document": [{"name": d.name, "path": d.path, "pages": d.pages, "chunks": d.chunk_count,
                                  "status": d.status, "error": d.error, "source": d.source, "enabled": d.enabled}
                                 for d in docs]}

    # ------------------------------------------------------------------ search
    def _build(self) -> HybridSearcher:
        chunks = self.repo.all_chunks(only_active=True)
        self._chunks = {c.chunk_id: c for c in chunks}
        self._sources = self.repo.source_map()
        sig = self.embedder.signature
        items = []
        dim = next((len(c.embedding) for c in chunks if c.embedding is not None and c.embedder == sig), None)
        self._qemb = self.embedder if dim is not None else self._fallback
        for c in chunks:
            ok = c.embedding is not None and c.embedder == self._qemb.signature
            items.append(SearchItem(c.chunk_id, c.text, c.embedding if ok else self._qemb.embed(c.text),
                                    boost_text=(c.question + " " + c.section).strip()))
        return HybridSearcher(self._qemb).build(items)

    def _snapshot(self):
        with self._lock:
            if self._searcher is None:
                self._searcher = self._build()
            return self._searcher, self._chunks, self._sources, self._qemb

    def _query_vector(self, qemb, query: str):
        try:
            return qemb.embed(query)
        except EmbeddingError as e:
            self.warnings.append(str(e))
            return self._fallback.embed(query)

    def _hits(self, query: str, top_k: int, min_score: float, wide: int = 4) -> list[KBHit]:
        searcher, chunks, sources, qemb = self._snapshot()
        if len(searcher) == 0:
            return []
        raw = searcher.search(query, top_k + wide, min_score, query_vector=self._query_vector(qemb, query))
        out = []
        for h in raw:
            c = chunks.get(h.id)
            if c is None:
                continue
            src = sources.get(c.doc_path, "folder")
            score = min(1.0, h.score + QA_BONUS) if src == "qa" else h.score
            out.append(KBHit(c.chunk_id, c.document, c.page, c.section, c.kind, c.question, c.text, score, c.doc_path, src,
                             h.bm25, h.cosine, h.bm25_rank, h.vec_rank, list(h.matched_terms)))
        out.sort(key=lambda k: -k.score)
        return out[:top_k]

    def search(self, query: str, top_k: int = 4, min_score: float = 0.25) -> list[KBHit]:
        t0 = time.perf_counter()
        out = self._hits(query, top_k, min_score)
        log_event("retrieval", "kb search", query=redact(query), ids=[h.chunk_id for h in out],
                  scores=[round(h.score, 3) for h in out], ms=round((time.perf_counter() - t0) * 1000, 2))
        return out

    def search_debug(self, query: str, top_k: int = 6) -> list[KBHit]:
        """The 'search playground': no relevance floor, with BM25 / vector / fused scores and ranks."""
        return self._hits(query, top_k, 0.0, wide=0)
