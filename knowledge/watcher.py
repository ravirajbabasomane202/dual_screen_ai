"""Optional folder watcher: polls the knowledge-base folder and re-syncs when files were added / changed / removed.
Polling (not OS notifications) keeps it dependency-free and identical on every platform; the signature check is cheap
(names, sizes, mtimes), and the sync itself is hash-aware, so unchanged PDFs are never re-parsed."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from knowledge.pdf_parser import SUPPORTED

log = logging.getLogger("app.kb.watch")


def folder_signature(folders) -> tuple:
    """(path, size, mtime) of every supported file under one folder or a list of folders."""
    if isinstance(folders, (str, Path)):
        folders = [folders]
    sig = []
    for folder in folders:
        root = Path(folder)
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in SUPPORTED and not any(x.startswith(".") for x in p.relative_to(root).parts):
                    st = p.stat()
                    sig.append((str(p), st.st_size, int(st.st_mtime)))
            except OSError:
                continue
    return tuple(sorted(sig))


class FolderWatcher:
    def __init__(self, index, folders_getter, on_report=None, interval: float = 30.0):
        """`folders_getter()` returns the list of watched folders (a single path string is accepted too)."""
        self.index, self.folders_getter, self.on_report, self.interval = index, folders_getter, on_report, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sig: tuple | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="kb-watch")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _folders(self) -> list[str]:
        f = self.folders_getter()
        return [f] if isinstance(f, str) and f else [x for x in (f or []) if x]

    def check_now(self):
        folders = self._folders()
        if not folders:
            return None
        sig = folder_signature(folders)
        if self._sig is not None and sig == self._sig:
            return None
        first = self._sig is None
        self._sig = sig
        if first:
            return None                      # the initial state was synced at start-up (or by the user)
        rep = self.index.sync_folders(folders)
        if self.on_report:
            self.on_report(rep)
        return rep

    def prime(self) -> None:
        folders = self._folders()
        if folders:
            self._sig = folder_signature(folders)

    def _run(self) -> None:
        self.prime()
        while not self._stop.wait(max(1.0, float(self.interval))):
            try:
                self.check_now()
            except Exception:
                log.exception("knowledge-base watch failed")
