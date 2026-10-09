"""Resume lifecycle: upload / view / replace / re-parse / delete, several named profiles, manual edits, history.

Safety rules
* A new file replaces the old profile only AFTER it parsed successfully (single DB transaction; the previous content
  also goes to the version history, so "replace" is undoable).
* Listeners registered with `on_change(cb)` are UI code and are ALWAYS called on the GUI thread, whichever thread did
  the upload. Data-model listeners (retriever, memory) register with `thread="inline"` and must not touch widgets.
* Nothing here raises a raw exception to the UI: failures become `ResumeParseError` with a plain-English message.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from app.crashguard import breadcrumb
from app.errors import Cancelled, FileTooLarge, ResumeParseError
from app.gui_thread import call_on_gui
from app.status import get_bus
from applog import log_event
from resume.parser import extract_text, parse_resume
from resume.storage import EDITABLE, ResumeStore, StoredResume

MAX_RESUME_BYTES = 30 * 1024 * 1024


class ResumeManager:
    def __init__(self, db, settings=None, bus=None):
        self.store, self.settings = ResumeStore(db), settings
        self.bus = bus or get_bus()
        self._lock = threading.RLock()
        self._listeners: list[tuple[Callable[[StoredResume | None], None], str]] = []

    # ------------------------------------------------------------------------------------------ listeners
    def on_change(self, cb: Callable[[StoredResume | None], None], thread: str = "gui") -> None:
        self._listeners.append((cb, thread))

    def _fire(self) -> None:
        cur = self.get()
        for cb, thread in list(self._listeners):
            if thread == "inline":
                self._call(cb, cur)
            else:
                call_on_gui(self._call, cb, cur)

    @staticmethod
    def _call(cb, cur) -> None:
        try:
            cb(cur)
        except Exception:
            log_event("retrieval", "resume listener failed", level=40, exc_info=True)

    # ------------------------------------------------------------------------------------------ upload
    def upload(self, path, *, new_profile: bool = False, label: str = "", cancel=None, progress=None) -> StoredResume:
        """Read, parse and store a resume file. Replaces the ACTIVE profile unless new_profile=True."""
        p = Path(path)
        breadcrumb("resume.upload: reading", file=p.name)
        try:
            size = p.stat().st_size
            if size > MAX_RESUME_BYTES:
                raise FileTooLarge(f"{p.name}: the file is {size / 1_048_576:.0f} MB; the limit is "
                                   f"{MAX_RESUME_BYTES // 1_048_576} MB. Your previous resume was kept.")
            data = p.read_bytes()
        except OSError as e:
            raise ResumeParseError(f"{p.name}: cannot read file ({e.strerror or e}).") from e
        res = self.upload_bytes(p.name, data, new_profile=new_profile, label=label or p.stem, cancel=cancel,
                                progress=progress)
        if self.settings is not None:
            self.settings.set("resume.path", str(p))          # safe from any thread: listeners run on the GUI thread
        return res

    def upload_bytes(self, filename: str, data: bytes, *, new_profile: bool = False, label: str = "",
                     profile_id: int | None = None, cancel=None, progress=None) -> StoredResume:
        with self._lock:
            try:
                breadcrumb("resume.upload: extracting text", file=filename)
                if len(data) > MAX_RESUME_BYTES:
                    raise FileTooLarge(f"{filename}: larger than {MAX_RESUME_BYTES // 1_048_576} MB. "
                                       "Your previous resume was kept.")
                text = extract_text(data, filename, cancel=cancel, progress=progress)
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                breadcrumb("resume.upload: parsing", file=filename)
                parsed = parse_resume(text)
                if not (parsed.skills or parsed.experience or parsed.education or parsed.projects or parsed.summary):
                    raise ResumeParseError(f"{filename}: nothing resume-like (no skills, experience, education or "
                                           "projects) could be recognised. Your previous resume was kept.")
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                breadcrumb("resume.upload: saving", file=filename)
                if profile_id is not None:
                    self.store.replace_profile(profile_id, filename, data, text, parsed)
                elif new_profile:
                    self.store.add_profile(filename, data, text, parsed, label)
                else:
                    self.store.save_new(filename, data, text, parsed, label)
            except (ResumeParseError, FileTooLarge) as e:
                log_event("retrieval", "resume upload failed", level=30, error=str(e), file=filename)
                raise
            except Cancelled:
                log_event("retrieval", "resume upload cancelled", file=filename)
                raise
            except Exception as e:
                log_event("retrieval", "resume upload crashed", level=40, exc_info=True)
                raise ResumeParseError(f"{filename}: could not be processed ({type(e).__name__}). "
                                       "Your previous resume was kept.") from e
            log_event("retrieval", "resume stored", file=filename, skills=len(parsed.skills),
                      experience=len(parsed.experience), projects=len(parsed.projects))
        breadcrumb("resume.upload: notifying")
        self._fire()
        breadcrumb("resume.upload: done")
        return self.get()

    # ------------------------------------------------------------------------------------------ reading
    def get(self, profile_id: int | None = None) -> StoredResume | None:
        with self._lock:
            return self.store.load(profile_id)

    def has_resume(self) -> bool:
        return self.get() is not None

    def profiles(self) -> list[StoredResume]:
        with self._lock:
            return self.store.list_profiles()

    def versions(self, profile_id: int | None = None):
        cur = self.get(profile_id)
        return self.store.versions(cur.id) if cur else []

    # ------------------------------------------------------------------------------------------ profiles
    def activate(self, profile_id: int) -> StoredResume:
        with self._lock:
            try:
                self.store.activate(profile_id)
            except KeyError:
                raise ResumeParseError("That resume profile no longer exists.") from None
        self._fire()
        return self.get()

    def rename(self, profile_id: int, label: str) -> None:
        with self._lock:
            self.store.set_label(profile_id, label)
        self._fire()

    def delete(self, profile_id: int | None = None) -> None:
        """Delete a profile (default: the active one). If other profiles remain, the newest becomes active."""
        with self._lock:
            target = self.get(profile_id)
            if target is None:
                return
            self.store.delete(target.id)
            rest = self.store.list_profiles()
            if rest and not any(r.active for r in rest):
                self.store.activate(rest[-1].id)
            if self.settings is not None and not rest:
                self.settings.set("resume.path", "")
        self._fire()

    def delete_all(self) -> None:
        with self._lock:
            self.store.delete(None)
            if self.settings is not None:
                self.settings.set("resume.path", "")
        self._fire()

    # ------------------------------------------------------------------------------------------ re-parse / edit
    def reparse(self, profile_id: int | None = None, keep_edits: bool = True, cancel=None) -> StoredResume:
        with self._lock:
            cur = self.store.load(profile_id)
            orig = self.store.load_original(cur.id) if cur else None
            if not cur or not orig:
                raise ResumeParseError("There is no stored resume to re-parse.")
            breadcrumb("resume.reparse: parsing")
            try:
                text = extract_text(orig[1], orig[0], cancel=cancel)
                self.store.update_parsed(cur.id, text, parse_resume(text), keep_edits=keep_edits)
            except (ResumeParseError, Cancelled):
                raise
            except Exception as e:
                raise ResumeParseError(f"Re-parse failed ({type(e).__name__}); the existing profile was kept.") from e
        self._fire()
        return self.get(profile_id)

    def edit_fields(self, edits: dict, profile_id: int | None = None) -> StoredResume:
        """Manually correct parsed fields. `edits` maps field name -> new value (merged into earlier edits)."""
        bad = [k for k in edits if k not in EDITABLE]
        if bad:
            raise ResumeParseError(f"These fields cannot be edited: {', '.join(bad)}.")
        with self._lock:
            cur = self.store.load(profile_id)
            if not cur:
                raise ResumeParseError("There is no stored resume to edit.")
            merged = dict(cur.edits)
            merged.update(edits)
            self.store.set_edits(cur.id, merged)
        self._fire()
        return self.get(profile_id)

    def reset_fields(self, names=None, profile_id: int | None = None) -> StoredResume:
        """Drop manual edits (all, or the named fields) and go back to what the parser found."""
        with self._lock:
            cur = self.store.load(profile_id)
            if not cur:
                raise ResumeParseError("There is no stored resume.")
            keep = {} if names is None else {k: v for k, v in cur.edits.items() if k not in set(names)}
            self.store.set_edits(cur.id, keep)
        self._fire()
        return self.get(profile_id)

    def restore_version(self, version_id: int) -> StoredResume:
        with self._lock:
            try:
                pid = self.store.restore(version_id)
            except KeyError:
                raise ResumeParseError("That version no longer exists.") from None
        self._fire()
        return self.get(pid)

    # ------------------------------------------------------------------------------------------ presentation
    def summary(self) -> dict:
        r = self.get()
        if not r:
            return {"loaded": False}
        p = r.parsed
        return {"loaded": True, "filename": r.filename, "label": r.label, "name": p.name, "years": p.years_of_experience,
                "skills": len(p.skills), "experience": len(p.experience), "projects": len(p.projects),
                "education": len(p.education), "certifications": len(p.certifications), "warnings": list(p.warnings),
                "edited": r.edited, "profiles": len(self.profiles())}

    def view_text(self) -> str:
        r = self.get()
        if not r:
            return "No resume uploaded."
        p = r.parsed
        out = [f"File: {r.filename}", f"Name: {p.name or '—'}", f"Years of experience: {p.years_of_experience or '—'}"]
        if p.summary:
            out += ["", "PROFESSIONAL SUMMARY", p.summary]
        if p.skills:
            out += ["", "SKILLS", ", ".join(p.skills)]
        for title, vals in (("PROGRAMMING LANGUAGES", p.programming_languages), ("FRAMEWORKS", p.frameworks),
                            ("TOOLS", p.tools)):
            if vals:
                out += ["", title, ", ".join(vals)]
        if p.experience:
            out += ["", "WORK EXPERIENCE"]
            for e in p.experience:
                out.append(f"• {e['title'] or '?'} — {e['company'] or '?'} ({e['start']} – {e['end']})")
                out += [f"    - {b}" for b in e["bullets"]]
        if p.projects:
            out += ["", "PROJECTS"]
            for pr in p.projects:
                out.append(f"• {pr['name']}" + (f" — {pr['description']}" if pr["description"] else ""))
                out += [f"    - {b}" for b in pr["bullets"]]
                if pr["technologies"]:
                    out.append("    Technologies: " + ", ".join(pr["technologies"]))
        if p.education:
            out += ["", "EDUCATION"] + [f"• {e['degree']} — {e['institution']} {e['dates']}".strip() for e in p.education]
        if p.certifications:
            out += ["", "CERTIFICATIONS"] + [f"• {c}" for c in p.certifications]
        if p.achievements:
            out += ["", "ACHIEVEMENTS"] + [f"• {a}" for a in p.achievements]
        if p.warnings:
            out += ["", "PARSER NOTES"] + [f"! {w}" for w in p.warnings]
        return "\n".join(out)
