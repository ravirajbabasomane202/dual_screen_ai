"""Resume chunking + retrieval + entity matching (used by the question classifier and the prompt builder)."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from applog import log_event, redact
from knowledge.hybrid import HashingEmbedder, HybridSearcher, SearchItem, tokenize
from resume.parser import ParsedResume


@dataclass
class ResumeChunk:
    id: str
    kind: str          # summary|skills|experience|project|education|certification|achievement|raw
    title: str
    text: str
    score: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.kind.capitalize()}: {self.title}" if self.title else self.kind.capitalize()


def build_chunks(p: ParsedResume, raw_text: str = "") -> list[ResumeChunk]:
    out: list[ResumeChunk] = []
    if p.summary:
        out.append(ResumeChunk("summary", "summary", "Professional summary", p.summary))
    cats = [("Programming languages", p.programming_languages), ("Frameworks", p.frameworks), ("Tools", p.tools)]
    lines = [f"{n}: {', '.join(v)}" for n, v in cats if v]
    rest = [s for s in p.skills if s not in p.technologies]
    if rest:
        lines.append("Other skills: " + ", ".join(rest))
    if lines:
        out.append(ResumeChunk("skills", "skills", "Skills", "\n".join(lines)))
    for i, e in enumerate(p.experience):
        span = f"{e['start']} – {e['end']}"
        body = f"{e['title']} at {e['company']} ({span})" + "".join(f"\n- {b}" for b in e["bullets"])
        if e["technologies"]:
            body += "\nTechnologies: " + ", ".join(e["technologies"])
        out.append(ResumeChunk(f"exp{i}", "experience", f"{e['title']} at {e['company']} ({span})".strip(), body))
    for i, pr in enumerate(p.projects):
        body = f"Project: {pr['name']}" + (f"\n{pr['description']}" if pr["description"] else "") \
            + "".join(f"\n- {b}" for b in pr["bullets"])
        if pr["technologies"]:
            body += "\nTechnologies: " + ", ".join(pr["technologies"])
        out.append(ResumeChunk(f"proj{i}", "project", pr["name"], body))
    for i, e in enumerate(p.education):
        out.append(ResumeChunk(f"edu{i}", "education", e["degree"] or e["institution"],
                               " — ".join(x for x in (e["degree"], e["institution"], e["dates"], e["details"]) if x)))
    if p.certifications:
        out.append(ResumeChunk("certs", "certification", "Certifications", "\n".join(f"- {c}" for c in p.certifications)))
    if p.achievements:
        out.append(ResumeChunk("ach", "achievement", "Achievements", "\n".join(f"- {a}" for a in p.achievements[:12])))
    # raw fallback: text not already represented, so oddly formatted resumes stay retrievable
    covered = set(re.findall(r"[a-z0-9+#]+", " ".join(c.text for c in out).lower()))
    words = raw_text.split()
    for i in range(0, len(words), 150):
        piece = " ".join(words[i:i + 150])
        toks = re.findall(r"[a-z0-9+#]+", piece.lower())
        if len(toks) >= 12 and sum(t in covered for t in toks) / len(toks) < 0.8:
            out.append(ResumeChunk(f"raw{i // 150}", "raw", "", piece))
    return out


_DEICTIC = re.compile(r"\b(this|that|the|your)\s+(project|role|position|job|company|team|work|internship|experience|system|"
                      r"application|app|product|model)\b|\byour (previous|last|current|recent|present|former)\b|\bhere\b", re.I)


class ResumeRetriever:
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingEmbedder()
        self._stored = None
        self._chunks: dict[str, ResumeChunk] = {}
        self._searcher: HybridSearcher | None = None
        self._entities: set[str] = set()

    def attach(self, manager) -> "ResumeRetriever":
        self.set_resume(manager.get())
        manager.on_change(self.set_resume, thread="inline")      # data model: updated before upload() returns
        return self

    def set_resume(self, stored) -> None:
        # built into locals and swapped in at the end, so a search running on the engine thread never sees a half-built index
        if not stored:
            self._stored, self._chunks, self._searcher, self._entities = None, {}, None, set()
            return
        p = stored.parsed
        chunks = build_chunks(p, stored.raw_text)
        chunk_map = {c.id: c for c in chunks}
        searcher = HybridSearcher(self.embedder).build(
            [SearchItem(c.id, c.text, boost_text=c.title) for c in chunks])
        ents = set(p.companies) | set(p.job_titles) | set(p.skills) | {x["name"] for x in p.projects}
        entities = {e.lower() for e in ents if e and len(e) > 1}
        self._stored, self._chunks, self._searcher, self._entities = stored, chunk_map, searcher, entities

    @property
    def loaded(self) -> bool:
        return self._stored is not None

    def search(self, query: str, top_k: int = 3, min_score: float = 0.25) -> list[ResumeChunk]:
        if not self._searcher:
            return []
        hits = self._searcher.search(query, top_k, min_score)
        out = [replace(self._chunks[h.id], score=h.score) for h in hits]
        log_event("retrieval", "resume search", query=redact(query), ids=[c.id for c in out])
        return out

    def entities(self) -> set[str]:
        return set(self._entities)

    def relevance(self, text: str) -> tuple[float, list[str]]:
        if not self._stored:
            return 0.0, []
        low = " " + re.sub(r"[^\w+#.\s/-]", " ", text.lower()) + " "
        matched = [e for e in self._entities if len(e) > 1 and re.search(r"(?<![\w+#])" + re.escape(e) + r"(?![\w+#])", low)]
        # generic one-word titles / very common skills alone are weak evidence
        matched = sorted(matched, key=len, reverse=True)
        if matched:
            return round(min(0.95, 0.6 + 0.1 * (len(matched) - 1)), 2), matched[:6]
        if _DEICTIC.search(text):
            return 0.4, []
        return 0.0, []

    def candidate_profile(self) -> dict:
        if not self._stored:
            return {}
        p = self._stored.parsed
        cur = p.experience[0] if p.experience else {}
        return {"name": p.name, "current_title": cur.get("title", ""), "current_company": cur.get("company", ""),
                "years_of_experience": p.years_of_experience, "top_skills": p.skills[:12],
                "projects": [x["name"] for x in p.projects]}

    def project_names(self) -> list[str]:
        return [x["name"] for x in self._stored.parsed.projects] if self._stored else []

    def profile_brief(self, max_chars: int = 700) -> str:
        c = self.candidate_profile()
        if not c:
            return ""
        parts = []
        if c["name"]:
            parts.append(f"Candidate: {c['name']}")
        role = " at ".join(x for x in (c["current_title"], c["current_company"]) if x)
        if role:
            parts.append(f"Current/most recent role: {role}")
        if c["years_of_experience"]:
            parts.append(f"Experience: ~{c['years_of_experience']:g} years")
        if c["top_skills"]:
            parts.append("Key skills: " + ", ".join(c["top_skills"]))
        if c["projects"]:
            parts.append("Projects: " + "; ".join(c["projects"][:6]))
        s = ". ".join(parts)
        return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"
