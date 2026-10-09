"""Per-role 'interview target': company, role and a pasted job description, used as extra context for answers such as
"Why do you want to join us?". Stored locally in its own table; editable and deletable."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass


@dataclass
class InterviewProfile:
    company: str = ""
    role: str = ""
    job_description: str = ""
    notes: str = ""

    @property
    def empty(self) -> bool:
        return not any((self.company.strip(), self.role.strip(), self.job_description.strip(), self.notes.strip()))

    def to_text(self, jd_chars: int = 1200) -> str:
        """Compact block for the prompt. Company/role always fit; the JD is trimmed at a sentence/line boundary."""
        if self.empty:
            return ""
        lines = ["INTERVIEW TARGET (the company and role the candidate is interviewing for):"]
        if self.company.strip():
            lines.append(f"Company: {self.company.strip()}")
        if self.role.strip():
            lines.append(f"Role: {self.role.strip()}")
        jd = " ".join(self.job_description.split())
        if jd:
            if len(jd) > jd_chars:
                cut = jd[:jd_chars]
                end = max(cut.rfind(". "), cut.rfind("; "))
                jd = (cut[:end + 1] if end > jd_chars * 0.6 else cut).rstrip() + " …"
            lines.append("Job description: " + jd)
        if self.notes.strip():
            lines.append("Candidate notes: " + " ".join(self.notes.split())[:400])
        return "\n".join(lines)


class InterviewProfileStore:
    def __init__(self, db):
        self.db = db

    def load(self) -> InterviewProfile:
        r = self.db.query_one("SELECT data FROM interview_profile WHERE id=1")
        if not r:
            return InterviewProfile()
        d = json.loads(r[0])
        return InterviewProfile(**{k: str(d.get(k, "")) for k in asdict(InterviewProfile())})

    def save(self, p: InterviewProfile) -> None:
        if p.empty:
            self.delete()
            return
        self.db.execute("REPLACE INTO interview_profile(id,ts,data) VALUES(1,?,?)", (time.time(), json.dumps(asdict(p))))

    def delete(self) -> None:
        self.db.execute("DELETE FROM interview_profile")
