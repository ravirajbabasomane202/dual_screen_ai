"""SQLite storage: settings (always), question history / context (only if the user enables it), and — through
`database.migrations` / `database.repositories` — the resume profile, the PDF knowledge base and interview memory."""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from database.migrations import migrate

BACKUPS_KEEP = 5


class Database:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            from app.paths import app_dir
            path = app_dir() / "data.db"
        self.path = str(path)
        self._lock = threading.RLock()          # re-entrant: repositories may nest calls inside a transaction
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)  # autocommit; we BEGIN explicitly
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self.schema_version = migrate(self._conn, before=self._backup_before_migration)

    def _backup_before_migration(self, have: int, latest: int) -> None:
        """Copy the database to backups/ before any schema change, so an upgrade can always be rolled back."""
        if self.path == ":memory:":
            return
        try:
            src = Path(self.path)
            dst_dir = src.parent / "backups"
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / f"{src.stem}-v{have}-{time.strftime('%Y%m%d-%H%M%S')}.db"
            out = sqlite3.connect(dst)
            with out:
                self._conn.backup(out)
            out.close()
            old = sorted(dst_dir.glob(f"{src.stem}-v*.db"), key=lambda p: p.stat().st_mtime)
            for p in old[:-BACKUPS_KEEP]:
                p.unlink(missing_ok=True)
            import logging
            logging.getLogger("app.db").info("schema v%s -> v%s: backup written to %s", have, latest, dst)
        except Exception:                      # a failed backup must not block start-up, but it must be visible
            import logging
            logging.getLogger("app.db").error("pre-migration backup failed", exc_info=True)

    # generic helpers (used by repositories) -----------------------------------------------------------
    @contextmanager
    def transaction(self):
        """All-or-nothing block: `with db.transaction() as conn: conn.execute(...)`."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params=()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params=()):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # settings -----------------------------------------------------------
    def set_kv(self, key: str, value) -> None:
        self.execute("REPLACE INTO settings(key,value) VALUES(?,?)", (key, json.dumps(value)))

    def all_kv(self) -> dict:
        return {k: json.loads(v) for k, v in self.query("SELECT key,value FROM settings")}

    # history ------------------------------------------------------------
    def add_question(self, question: str, answer: str) -> None:
        self.execute("INSERT INTO questions(ts,question,answer) VALUES(?,?,?)", (time.time(), question, answer))

    def recent_questions(self, limit: int = 50) -> list[tuple[str, str]]:
        rows = self.query("SELECT question,answer FROM questions ORDER BY id DESC LIMIT ?", (limit,))
        return list(reversed(rows))

    def clear_questions(self) -> None:
        self.execute("DELETE FROM questions")

    # context ------------------------------------------------------------
    def save_context(self, pairs: list[tuple[str, str]]) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM context")
            c.executemany("INSERT INTO context(ts,question,answer) VALUES(?,?,?)",
                          [(time.time(), q, a) for q, a in pairs])

    def load_context(self) -> list[tuple[str, str]]:
        return self.query("SELECT question,answer FROM context ORDER BY id")

    def clear_context(self) -> None:
        self.execute("DELETE FROM context")
        self.execute("DELETE FROM interview_context")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
