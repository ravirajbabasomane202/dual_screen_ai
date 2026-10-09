"""Versioned schema migrations. Each migration runs once, inside a transaction; the applied version is stored in
`schema_version`. Version 1 is the original schema (CREATE IF NOT EXISTS, so existing databases adopt it silently)."""
import sqlite3
import time

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, question TEXT, answer TEXT);
    CREATE TABLE IF NOT EXISTS context (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, question TEXT, answer TEXT);
    """,
    # resume / profile ---------------------------------------------------------------------------------------
    2: """
    CREATE TABLE resume_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        ext TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        original BLOB,                      -- the uploaded file, kept so it can be re-parsed
        raw_text TEXT NOT NULL DEFAULT '',
        parsed_json TEXT NOT NULL DEFAULT '{}',
        parser_version INTEGER NOT NULL DEFAULT 1,
        active INTEGER NOT NULL DEFAULT 0,
        uploaded_at REAL NOT NULL,
        parsed_at REAL
    );
    CREATE UNIQUE INDEX one_active_resume ON resume_profiles(active) WHERE active = 1;
    """,
    # PDF knowledge base -------------------------------------------------------------------------------------
    3: """
    CREATE TABLE kb_documents (
        path TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size INTEGER NOT NULL DEFAULT 0,
        pages INTEGER NOT NULL DEFAULT 0,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ok',      -- ok | no_text | error
        error TEXT NOT NULL DEFAULT '',
        indexed_at REAL NOT NULL
    );
    CREATE TABLE kb_chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_path TEXT NOT NULL REFERENCES kb_documents(path) ON DELETE CASCADE,
        document TEXT NOT NULL,
        page INTEGER NOT NULL DEFAULT 0,
        section TEXT NOT NULL DEFAULT '',
        kind TEXT NOT NULL DEFAULT 'text',      -- text | qa | list | table
        question TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL,
        tokens INTEGER NOT NULL DEFAULT 0,
        embedding BLOB,
        dim INTEGER NOT NULL DEFAULT 0,
        embedder TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX kb_chunks_doc ON kb_chunks(doc_path);
    CREATE TABLE kb_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """,
    # persistent interview context (only written when the user opted in to storing history) -------------------
    4: """
    CREATE TABLE interview_context (
        id INTEGER PRIMARY KEY CHECK (id = 1), ts REAL NOT NULL, data TEXT NOT NULL);
    """,
    # multiple resume profiles, manual field edits, version history -------------------------------------------
    5: """
    ALTER TABLE resume_profiles ADD COLUMN label TEXT NOT NULL DEFAULT '';
    ALTER TABLE resume_profiles ADD COLUMN edits_json TEXT NOT NULL DEFAULT '{}';
    CREATE TABLE resume_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER NOT NULL REFERENCES resume_profiles(id) ON DELETE CASCADE,
        reason TEXT NOT NULL DEFAULT '',
        filename TEXT NOT NULL DEFAULT '',
        file_hash TEXT NOT NULL DEFAULT '',
        original BLOB,
        raw_text TEXT NOT NULL DEFAULT '',
        parsed_json TEXT NOT NULL DEFAULT '{}',
        edits_json TEXT NOT NULL DEFAULT '{}',
        parser_version INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL
    );
    CREATE INDEX resume_versions_profile ON resume_versions(profile_id, id);
    """,
    # knowledge-base manager: added files, notes, Q&A, disable, soft delete, chunk toggles ----------------------
    6: """
    ALTER TABLE kb_documents ADD COLUMN source TEXT NOT NULL DEFAULT 'folder';
    ALTER TABLE kb_documents ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE kb_documents ADD COLUMN deleted_at REAL;
    ALTER TABLE kb_chunks ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1;
    CREATE TABLE kb_entries (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        body TEXT NOT NULL DEFAULT '',
        question TEXT NOT NULL DEFAULT '',
        answer TEXT NOT NULL DEFAULT '',
        tags TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    # feedback, pinned answers, interview target profile -------------------------------------------------------
    7: """
    CREATE TABLE question_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, text TEXT NOT NULL, norm TEXT NOT NULL,
        label TEXT NOT NULL, detected TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0,
        action TEXT NOT NULL DEFAULT '');
    CREATE INDEX question_feedback_norm ON question_feedback(norm);
    CREATE TABLE answer_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
        rating INTEGER NOT NULL, grounding TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT '');
    CREATE TABLE pinned_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL);
    CREATE TABLE interview_profile (
        id INTEGER PRIMARY KEY CHECK (id = 1), ts REAL NOT NULL, data TEXT NOT NULL);
    """,
}


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def latest_version() -> int:
    return max(MIGRATIONS)


def migrate(conn: sqlite3.Connection, before=None) -> int:
    """Apply every pending migration; returns the resulting schema version.
    `before(have, latest)` is called once, only when something is pending (used to take a backup first)."""
    have = current_version(conn)
    has_data = have > 0 or conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name IN ('settings','questions')").fetchone()
    if have < max(MIGRATIONS) and before is not None and has_data:      # legacy databases (no version table) too
        before(have, max(MIGRATIONS))
    for version in sorted(MIGRATIONS):
        if version <= have:
            continue
        try:
            conn.execute("BEGIN")
            for stmt in _statements(MIGRATIONS[version]):
                conn.execute(stmt)
            conn.execute("INSERT INTO schema_version(version, applied_at) VALUES(?,?)", (version, time.time()))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return max(MIGRATIONS)


def _statements(script: str) -> list[str]:
    # sqlite3.complete_statement keeps this correct even though no statement contains an embedded ';'
    out, buf = [], ""
    for line in script.splitlines():
        buf += line + "\n"
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out
