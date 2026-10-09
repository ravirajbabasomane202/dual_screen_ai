"""Persistence of resume profiles with replace-only-on-success semantics, manual edits and version history."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

from database.repositories import ResumeRepository, ResumeVersionRow
from resume.parser import PARSER_VERSION, ParsedResume

# fields the user may correct by hand (derived fields — companies, titles, technologies, years — are recomputed)
EDITABLE = ("name", "summary", "skills", "programming_languages", "frameworks", "tools", "experience", "projects",
            "education", "certifications", "achievements")


def apply_edits(parsed: dict, edits: dict | None) -> dict:
    """Overlay manual corrections on a parse result and recompute the derived fields."""
    out = copy.deepcopy(parsed or {})
    if not edits:
        return out
    for k, v in edits.items():
        if k in EDITABLE:
            out[k] = copy.deepcopy(v)
    if any(k in edits for k in ("programming_languages", "frameworks", "tools")):
        out["technologies"] = list(dict.fromkeys(list(out.get("programming_languages", [])) + list(out.get("frameworks", []))
                                                 + list(out.get("tools", []))))
    if "experience" in edits:
        exp = out.get("experience", [])
        out["companies"] = list(dict.fromkeys(e.get("company") for e in exp if e.get("company")))
        out["job_titles"] = list(dict.fromkeys(e.get("title") for e in exp if e.get("title")))
        out["responsibilities"] = [b for e in exp for b in e.get("responsibilities", [])]
    return out


@dataclass
class StoredResume:
    id: int
    filename: str
    file_hash: str
    uploaded_at: float
    parsed_at: float | None
    parsed: ParsedResume                       # parser output WITH the user's manual edits applied
    raw_text: str
    parser_version: int
    label: str = ""
    active: bool = True
    edits: dict = field(default_factory=dict)
    parsed_base: ParsedResume | None = None    # parser output WITHOUT the manual edits (for the editor / "reset field")

    @property
    def edited(self) -> bool:
        return bool(self.edits)


class ResumeStore:
    def __init__(self, db):
        self.repo = ResumeRepository(db)

    # ---- building
    @staticmethod
    def _stored(r) -> StoredResume:
        return StoredResume(r.id, r.filename, r.file_hash, r.uploaded_at, r.parsed_at,
                            ParsedResume.from_dict(apply_edits(r.parsed, r.edits)), r.raw_text, r.parser_version,
                            r.label, r.active, dict(r.edits), ParsedResume.from_dict(r.parsed))

    @staticmethod
    def _fields(filename: str, data: bytes, raw_text: str, parsed: ParsedResume) -> dict:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return dict(filename=filename, ext=ext, file_hash=hashlib.sha256(data).hexdigest(), original=data,
                    raw_text=raw_text, parsed=parsed.to_dict(), parser_version=PARSER_VERSION)

    # ---- saving
    def save_new(self, filename: str, data: bytes, raw_text: str, parsed: ParsedResume, label: str = "") -> int:
        """Replace the ACTIVE profile (or create the first one). Previous content stays in the version history."""
        return self.repo.replace_active(label=label, **self._fields(filename, data, raw_text, parsed))

    def add_profile(self, filename: str, data: bytes, raw_text: str, parsed: ParsedResume, label: str = "") -> int:
        return self.repo.create_profile(label=label, **self._fields(filename, data, raw_text, parsed))

    def replace_profile(self, profile_id: int, filename: str, data: bytes, raw_text: str, parsed: ParsedResume) -> int:
        return self.repo.replace_profile(profile_id, **self._fields(filename, data, raw_text, parsed))

    # ---- reading
    def load(self, profile_id: int | None = None) -> StoredResume | None:
        r = self.repo.get(profile_id) if profile_id is not None else self.repo.get_active()
        return self._stored(r) if r else None

    def list_profiles(self) -> list[StoredResume]:
        return [self._stored(r) for r in self.repo.list_profiles()]

    def load_original(self, profile_id: int | None = None) -> tuple[str, bytes] | None:
        r = self.repo.get(profile_id, with_original=True) if profile_id is not None else self.repo.get_active(with_original=True)
        return (r.filename, r.original) if r and r.original else None

    # ---- changing
    def update_parsed(self, row_id: int, raw_text: str, parsed: ParsedResume, keep_edits: bool = True) -> None:
        self.repo.update_parsed(row_id, raw_text, parsed.to_dict(), PARSER_VERSION, keep_edits)

    def set_edits(self, row_id: int, edits: dict) -> None:
        self.repo.set_edits(row_id, {k: v for k, v in edits.items() if k in EDITABLE})

    def set_label(self, row_id: int, label: str) -> None:
        self.repo.set_label(row_id, label.strip())

    def activate(self, row_id: int) -> None:
        self.repo.activate(row_id)

    def delete(self, profile_id: int | None = None) -> None:
        if profile_id is None:
            self.repo.delete_all()
        else:
            self.repo.delete(profile_id)

    # ---- history
    def versions(self, row_id: int) -> list[ResumeVersionRow]:
        return self.repo.list_versions(row_id)

    def restore(self, version_id: int) -> int:
        return self.repo.restore_version(version_id)
