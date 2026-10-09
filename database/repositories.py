"""Repositories: all SQL for the resume profile, the PDF knowledge base and persisted interview memory.
Domain logic lives elsewhere (resume/, knowledge/, context/); these classes only read and write rows."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from database.sqlite import Database


# --------------------------------------------------------------------------------------------- resume
@dataclass
class ResumeRow:
    id: int
    filename: str
    ext: str
    file_hash: str
    raw_text: str
    parsed: dict
    parser_version: int
    active: bool
    uploaded_at: float
    parsed_at: float | None
    original: bytes | None = None
    label: str = ""
    edits: dict = field(default_factory=dict)


@dataclass
class ResumeVersionRow:
    id: int
    profile_id: int
    reason: str
    filename: str
    created_at: float
    file_hash: str = ""


MAX_VERSIONS = 15
MAX_VERSION_ORIGINAL = 8 * 1024 * 1024        # larger originals are not duplicated into history


class ResumeRepository:
    """All resume SQL. Several profiles may exist (e.g. 'Backend role', 'Data role'); exactly one is active.
    Every change that overwrites content first snapshots the previous state into resume_versions, inside the same
    transaction, so a failure leaves the old data in place."""

    _COLS = "id,filename,ext,file_hash,raw_text,parsed_json,parser_version,active,uploaded_at,parsed_at,label,edits_json"

    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row(r, original=None) -> ResumeRow:
        return ResumeRow(r[0], r[1], r[2], r[3], r[4], json.loads(r[5] or "{}"), r[6], bool(r[7]), r[8], r[9], original,
                         r[10] or "", json.loads(r[11] or "{}"))

    # ---- reading
    def get(self, row_id: int, with_original: bool = False) -> ResumeRow | None:
        r = self.db.query_one(f"SELECT {self._COLS} FROM resume_profiles WHERE id=?", (row_id,))
        return self._with_original(r, with_original)

    def get_active(self, with_original: bool = False) -> ResumeRow | None:
        r = self.db.query_one(f"SELECT {self._COLS} FROM resume_profiles WHERE active=1")
        return self._with_original(r, with_original)

    def _with_original(self, r, with_original: bool):
        if not r:
            return None
        orig = None
        if with_original:
            o = self.db.query_one("SELECT original FROM resume_profiles WHERE id=?", (r[0],))
            orig = bytes(o[0]) if o and o[0] is not None else None
        return self._row(r, orig)

    def list_profiles(self) -> list[ResumeRow]:
        return [self._row(r) for r in self.db.query(f"SELECT {self._COLS} FROM resume_profiles ORDER BY id")]

    # ---- writing
    @staticmethod
    def _snapshot(c, row_id: int, reason: str) -> None:
        cur = c.execute("SELECT filename,file_hash,original,raw_text,parsed_json,edits_json,parser_version "
                        "FROM resume_profiles WHERE id=?", (row_id,)).fetchone()
        if not cur:
            return
        orig = cur[2]
        if orig is not None and (len(orig) > MAX_VERSION_ORIGINAL or reason in ("edited", "reparsed", "restored")):
            orig = None                                    # same file as the live row; shared, not duplicated
        c.execute("INSERT INTO resume_versions(profile_id,reason,filename,file_hash,original,raw_text,parsed_json,"
                  "edits_json,parser_version,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (row_id, reason, cur[0], cur[1], orig, cur[3], cur[4], cur[5], cur[6], time.time()))
        c.execute("DELETE FROM resume_versions WHERE profile_id=? AND id NOT IN "
                  "(SELECT id FROM resume_versions WHERE profile_id=? ORDER BY id DESC LIMIT ?)",
                  (row_id, row_id, MAX_VERSIONS))

    def create_profile(self, *, filename: str, ext: str, file_hash: str, original: bytes, raw_text: str, parsed: dict,
                       parser_version: int, label: str = "", activate: bool = True) -> int:
        now = time.time()
        with self.db.transaction() as c:
            if activate:
                c.execute("UPDATE resume_profiles SET active=0 WHERE active=1")
            has_active = c.execute("SELECT 1 FROM resume_profiles WHERE active=1").fetchone() is not None
            cur = c.execute(
                "INSERT INTO resume_profiles(filename,ext,file_hash,original,raw_text,parsed_json,parser_version,"
                "active,uploaded_at,parsed_at,label,edits_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,'{}')",
                (filename, ext, file_hash, original, raw_text, json.dumps(parsed), parser_version,
                 0 if has_active else 1, now, now, label))
            return cur.lastrowid

    def replace_profile(self, row_id: int, *, filename: str, ext: str, file_hash: str, original: bytes, raw_text: str,
                        parsed: dict, parser_version: int) -> int:
        """New file for an existing profile. The previous content (and its manual edits) stays in the history."""
        now = time.time()
        with self.db.transaction() as c:
            old = c.execute("SELECT original,file_hash FROM resume_profiles WHERE id=?", (row_id,)).fetchone()
            if old is None:
                raise KeyError(row_id)
            self._snapshot(c, row_id, "replaced")
            if old[0] is not None:                          # back-fill older snapshots that shared this file
                c.execute("UPDATE resume_versions SET original=? WHERE profile_id=? AND original IS NULL AND file_hash=?",
                          (old[0], row_id, old[1]))
            c.execute("UPDATE resume_profiles SET filename=?,ext=?,file_hash=?,original=?,raw_text=?,parsed_json=?,"
                      "parser_version=?,uploaded_at=?,parsed_at=?,edits_json='{}' WHERE id=?",
                      (filename, ext, file_hash, original, raw_text, json.dumps(parsed), parser_version, now, now, row_id))
        return row_id

    def replace_active(self, *, filename: str, ext: str, file_hash: str, original: bytes, raw_text: str,
                       parsed: dict, parser_version: int, label: str = "") -> int:
        """Replace the active profile's content, or create the first profile. Old content goes to the history."""
        act = self.get_active()
        kw = dict(filename=filename, ext=ext, file_hash=file_hash, original=original, raw_text=raw_text, parsed=parsed,
                  parser_version=parser_version)
        if act is None:
            return self.create_profile(label=label, **kw)
        return self.replace_profile(act.id, **kw)

    def update_parsed(self, row_id: int, raw_text: str, parsed: dict, parser_version: int, keep_edits: bool = True) -> None:
        with self.db.transaction() as c:
            self._snapshot(c, row_id, "reparsed")
            c.execute("UPDATE resume_profiles SET raw_text=?, parsed_json=?, parser_version=?, parsed_at=?"
                      + ("" if keep_edits else ", edits_json='{}'") + " WHERE id=?",
                      (raw_text, json.dumps(parsed), parser_version, time.time(), row_id))

    def set_edits(self, row_id: int, edits: dict) -> None:
        with self.db.transaction() as c:
            self._snapshot(c, row_id, "edited")
            c.execute("UPDATE resume_profiles SET edits_json=? WHERE id=?", (json.dumps(edits), row_id))

    def set_label(self, row_id: int, label: str) -> None:
        self.db.execute("UPDATE resume_profiles SET label=? WHERE id=?", (label, row_id))

    def activate(self, row_id: int) -> None:
        with self.db.transaction() as c:
            if c.execute("SELECT 1 FROM resume_profiles WHERE id=?", (row_id,)).fetchone() is None:
                raise KeyError(row_id)
            c.execute("UPDATE resume_profiles SET active=0 WHERE active=1")
            c.execute("UPDATE resume_profiles SET active=1 WHERE id=?", (row_id,))

    def delete(self, row_id: int) -> None:
        self.db.execute("DELETE FROM resume_profiles WHERE id=?", (row_id,))

    def delete_all(self) -> None:
        self.db.execute("DELETE FROM resume_profiles")

    # ---- history
    def list_versions(self, row_id: int) -> list[ResumeVersionRow]:
        return [ResumeVersionRow(*r) for r in self.db.query(
            "SELECT id,profile_id,reason,filename,created_at,file_hash FROM resume_versions WHERE profile_id=? "
            "ORDER BY id DESC", (row_id,))]

    def restore_version(self, version_id: int) -> int:
        with self.db.transaction() as c:
            v = c.execute("SELECT profile_id,filename,file_hash,original,raw_text,parsed_json,edits_json,parser_version "
                          "FROM resume_versions WHERE id=?", (version_id,)).fetchone()
            if v is None:
                raise KeyError(version_id)
            pid = v[0]
            self._snapshot(c, pid, "restored")
            ext = v[1].rsplit(".", 1)[-1].lower() if "." in v[1] else ""
            if v[3] is not None:
                c.execute("UPDATE resume_profiles SET filename=?,ext=?,file_hash=?,original=?,raw_text=?,parsed_json=?,"
                          "edits_json=?,parser_version=?,parsed_at=? WHERE id=?",
                          (v[1], ext, v[2], v[3], v[4], v[5], v[6], v[7], time.time(), pid))
            else:                                           # same file as the live row: keep its bytes
                c.execute("UPDATE resume_profiles SET filename=?,raw_text=?,parsed_json=?,edits_json=?,parser_version=?,"
                          "parsed_at=? WHERE id=?", (v[1], v[4], v[5], v[6], v[7], time.time(), pid))
            return pid


# --------------------------------------------------------------------------------------------- knowledge base
@dataclass
class KBDocRow:
    path: str
    name: str
    sha256: str
    size: int
    pages: int
    chunk_count: int
    status: str
    error: str
    indexed_at: float
    source: str = "folder"            # folder | file | note | qa
    enabled: bool = True              # excluded from retrieval when False (the content stays indexed)
    deleted_at: float | None = None   # soft delete (Undo window); folder tombstones keep it so a sync won't resurrect it


@dataclass
class KBChunkRow:
    chunk_id: str
    doc_path: str
    document: str
    page: int
    section: str
    kind: str
    question: str
    text: str
    tokens: int
    embedding: np.ndarray | None
    embedder: str
    enabled: bool = True


@dataclass
class KBEntryRow:
    id: str
    kind: str                         # note | qa
    title: str
    body: str
    question: str
    answer: str
    tags: str
    version: int
    created_at: float
    updated_at: float


_DOC_COLS = "path,name,sha256,size,pages,chunk_count,status,error,indexed_at,source,enabled,deleted_at"
_CHUNK_COLS = "chunk_id,doc_path,document,page,section,kind,question,text,tokens,embedding,dim,embedder,enabled"
_ENTRY_COLS = "id,kind,title,body,question,answer,tags,version,created_at,updated_at"


class KnowledgeRepository:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _doc(r) -> KBDocRow:
        return KBDocRow(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], bool(r[10]), r[11])

    @staticmethod
    def _chunk(r) -> KBChunkRow:
        return KBChunkRow(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
                          None if r[9] is None else np.frombuffer(r[9], np.float32).copy(), r[11], bool(r[12]))

    # documents
    def get_document(self, path: str) -> KBDocRow | None:
        r = self.db.query_one(f"SELECT {_DOC_COLS} FROM kb_documents WHERE path=?", (path,))
        return self._doc(r) if r else None

    def list_documents(self, include_deleted: bool = False) -> list[KBDocRow]:
        where = "" if include_deleted else " WHERE deleted_at IS NULL"
        return [self._doc(r) for r in self.db.query(f"SELECT {_DOC_COLS} FROM kb_documents{where} ORDER BY name COLLATE NOCASE")]

    def list_deleted(self) -> list[KBDocRow]:
        return [self._doc(r) for r in self.db.query(
            f"SELECT {_DOC_COLS} FROM kb_documents WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC")]

    def replace_document(self, doc: KBDocRow, chunks: list[KBChunkRow]) -> None:
        """Replace the rows of one document in one transaction. User choices survive a re-index: the document's
        enabled flag, its source, and the enabled flag of every chunk whose id is unchanged."""
        with self.db.transaction() as c:
            old = c.execute("SELECT source,enabled,deleted_at FROM kb_documents WHERE path=?", (doc.path,)).fetchone()
            flags = {r[0]: r[1] for r in c.execute("SELECT chunk_id,enabled FROM kb_chunks WHERE doc_path=?", (doc.path,))}
            source, enabled, deleted_at = (doc.source, int(doc.enabled), doc.deleted_at)
            if old is not None:
                source, enabled, deleted_at = old[0], old[1], old[2]          # the user's choices survive a re-index
            c.execute("DELETE FROM kb_chunks WHERE doc_path=?", (doc.path,))
            c.execute("DELETE FROM kb_documents WHERE path=?", (doc.path,))
            c.execute("INSERT INTO kb_documents(path,name,sha256,size,pages,chunk_count,status,error,indexed_at,source,"
                      "enabled,deleted_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (doc.path, doc.name, doc.sha256, doc.size, doc.pages, len(chunks), doc.status, doc.error,
                       doc.indexed_at, source, enabled, deleted_at))
            c.executemany(
                "INSERT INTO kb_chunks(chunk_id,doc_path,document,page,section,kind,question,text,tokens,embedding,dim,"
                "embedder,enabled) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(k.chunk_id, k.doc_path, k.document, k.page, k.section, k.kind, k.question, k.text, k.tokens,
                  None if k.embedding is None else np.asarray(k.embedding, np.float32).tobytes(),
                  0 if k.embedding is None else int(len(k.embedding)), k.embedder, flags.get(k.chunk_id, int(k.enabled)))
                 for k in chunks])

    def delete_document(self, path: str) -> None:
        with self.db.transaction() as c:
            c.execute("DELETE FROM kb_chunks WHERE doc_path=?", (path,))
            c.execute("DELETE FROM kb_documents WHERE path=?", (path,))

    def set_doc_enabled(self, path: str, enabled: bool) -> None:
        self.db.execute("UPDATE kb_documents SET enabled=? WHERE path=?", (int(enabled), path))

    def rename_document(self, path: str, name: str) -> None:
        with self.db.transaction() as c:
            c.execute("UPDATE kb_documents SET name=? WHERE path=?", (name, path))
            c.execute("UPDATE kb_chunks SET document=? WHERE doc_path=?", (name, path))

    def soft_delete(self, path: str, when: float) -> None:
        self.db.execute("UPDATE kb_documents SET deleted_at=? WHERE path=?", (when, path))

    def undelete(self, path: str) -> None:
        self.db.execute("UPDATE kb_documents SET deleted_at=NULL WHERE path=?", (path,))

    def purge_deleted(self, older_than: float) -> int:
        """Hard-delete soft-deleted documents older than the cutoff. Folder tombstones keep their row (without chunks)
        so a later folder sync does not bring a deliberately removed file back."""
        with self.db.transaction() as c:
            rows = c.execute("SELECT path,source FROM kb_documents WHERE deleted_at IS NOT NULL AND deleted_at<?",
                             (older_than,)).fetchall()
            for path, source in rows:
                c.execute("DELETE FROM kb_chunks WHERE doc_path=?", (path,))
                if source == "folder":
                    c.execute("UPDATE kb_documents SET chunk_count=0 WHERE path=?", (path,))
                else:
                    c.execute("DELETE FROM kb_documents WHERE path=?", (path,))
                    c.execute("DELETE FROM kb_entries WHERE id=?", (path.split("://", 1)[-1],))
            return len(rows)

    def clear(self) -> None:
        with self.db.transaction() as c:
            c.execute("DELETE FROM kb_chunks")
            c.execute("DELETE FROM kb_documents")
            c.execute("DELETE FROM kb_entries")

    # chunks
    def all_chunks(self, only_active: bool = False) -> list[KBChunkRow]:
        where = (" WHERE enabled=1 AND doc_path IN (SELECT path FROM kb_documents WHERE enabled=1 AND deleted_at IS NULL)"
                 if only_active else "")
        rows = self.db.query(f"SELECT {_CHUNK_COLS} FROM kb_chunks{where} ORDER BY doc_path,rowid")
        return [self._chunk(r) for r in rows]

    def chunks_of(self, path: str) -> list[KBChunkRow]:
        return [self._chunk(r) for r in self.db.query(
            f"SELECT {_CHUNK_COLS} FROM kb_chunks WHERE doc_path=? ORDER BY rowid", (path,))]

    def set_chunk_enabled(self, chunk_id: str, enabled: bool) -> None:
        self.db.execute("UPDATE kb_chunks SET enabled=? WHERE chunk_id=?", (int(enabled), chunk_id))

    def chunk_count(self, only_active: bool = False) -> int:
        if only_active:
            return self.db.query_one(
                "SELECT COUNT(*) FROM kb_chunks WHERE enabled=1 AND doc_path IN "
                "(SELECT path FROM kb_documents WHERE enabled=1 AND deleted_at IS NULL)")[0]
        return self.db.query_one("SELECT COUNT(*) FROM kb_chunks")[0]

    def storage_bytes(self) -> int:
        r = self.db.query_one("SELECT COALESCE(SUM(LENGTH(text)+COALESCE(LENGTH(embedding),0)),0) FROM kb_chunks")
        return int(r[0])

    def source_map(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self.db.query("SELECT path,source FROM kb_documents")}

    def set_embeddings(self, items: list[tuple[str, np.ndarray, str]]) -> None:
        """Re-embed existing chunks (embedder changed) without re-parsing any PDF."""
        with self.db.transaction() as c:
            c.executemany("UPDATE kb_chunks SET embedding=?, dim=?, embedder=? WHERE chunk_id=?",
                          [(np.asarray(v, np.float32).tobytes(), int(len(v)), sig, cid) for cid, v, sig in items])

    # entries (notes / Q&A written in the app)
    @staticmethod
    def _entry(r) -> KBEntryRow:
        return KBEntryRow(*r)

    def save_entry(self, e: KBEntryRow) -> None:
        self.db.execute(f"REPLACE INTO kb_entries({_ENTRY_COLS}) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (e.id, e.kind, e.title, e.body, e.question, e.answer, e.tags, e.version, e.created_at, e.updated_at))

    def get_entry(self, entry_id: str) -> KBEntryRow | None:
        r = self.db.query_one(f"SELECT {_ENTRY_COLS} FROM kb_entries WHERE id=?", (entry_id,))
        return self._entry(r) if r else None

    def list_entries(self) -> list[KBEntryRow]:
        return [self._entry(r) for r in self.db.query(f"SELECT {_ENTRY_COLS} FROM kb_entries ORDER BY updated_at DESC")]

    def delete_entry(self, entry_id: str) -> None:
        self.db.execute("DELETE FROM kb_entries WHERE id=?", (entry_id,))

    # meta
    def get_meta(self, key: str, default: str = "") -> str:
        r = self.db.query_one("SELECT value FROM kb_meta WHERE key=?", (key,))
        return r[0] if r else default

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("REPLACE INTO kb_meta(key,value) VALUES(?,?)", (key, value))


# --------------------------------------------------------------------------------------------- interview memory
class ContextRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, data: dict) -> None:
        self.db.execute("REPLACE INTO interview_context(id,ts,data) VALUES(1,?,?)", (time.time(), json.dumps(data)))

    def load(self) -> dict | None:
        r = self.db.query_one("SELECT data FROM interview_context WHERE id=1")
        return json.loads(r[0]) if r else None

    def clear(self) -> None:
        self.db.execute("DELETE FROM interview_context")
