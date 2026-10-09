"""Interview-level memory: facts that stay relevant however long the transcript grows.

Structured (never the raw transcript):
    candidate_profile, resume_context, current_topic, current_project, important_facts, recent_questions,
    recent_answers, technologies
"""
from __future__ import annotations

import re
from collections import OrderedDict

from resume.lexicon import find_terms

_CLAIM = re.compile(r"\byou(?:'ve| have)?\s+(?:mentioned|said|stated|told us|told me|noted|explained|described|claimed|"
                    r"built|developed|designed|implemented|deployed|created|worked|led|used|wrote)\b[^.?!]*", re.I)
_CONSTRAINT = re.compile(r"\b(?:in|within|under|about|around)\s+(?:\d+|a|one|two|three|five|ten)\s+(?:minutes?|mins?|sentences?|words?|seconds?)\b|"
                         r"\b(?:briefly|in brief|keep it short|in detail|step by step)\b", re.I)
_ACRONYM = re.compile(r"\b(?:[A-Z]{2,6}[a-z]?\d*|[A-Z][a-z]+[A-Z][A-Za-z]+)\b")
_COMMON_CAPS = {"I", "OK", "AND", "THE", "WHAT", "WHY", "HOW", "YOU", "CAN", "DID", "USA", "UK", "CV", "HR", "AI", "ML", "OKAY"}


def _third_person(clause: str) -> str:
    c = re.sub(r"\byou(?:'ve| have)\b", "the candidate has", clause, flags=re.I)
    c = re.sub(r"\byou(?:'re| are)\b", "the candidate is", c, flags=re.I)
    c = re.sub(r"\byour\b", "the candidate's", c, flags=re.I)
    c = re.sub(r"\byou\b", "the candidate", c, flags=re.I)
    return " ".join(c.split()).strip(" ,;")


class PersistentContext:
    def __init__(self, max_facts: int = 12):
        self.max_facts = max_facts
        self.clear(keep_profile=False)

    # ----------------------------------------------------------------------------- state
    def clear(self, keep_profile: bool = True) -> None:
        if not keep_profile or not hasattr(self, "candidate_profile"):
            self.candidate_profile: dict = {}
            self.resume_context: str = ""
            self.project_names: list[str] = []
        self.current_topic = ""
        self.current_project = ""
        self.technologies: "OrderedDict[str, None]" = OrderedDict()
        self.important_facts: list[str] = []
        self.recent_questions: list[str] = []
        self.recent_answers: list[str] = []
        self.constraints: list[str] = []

    def set_profile(self, profile: dict, brief: str = "", project_names: list[str] | None = None) -> None:
        self.candidate_profile = dict(profile or {})
        self.resume_context = brief or ""
        self.project_names = list(project_names or [])

    # ----------------------------------------------------------------------------- learning
    def observe_question(self, text: str, known_entities: set[str] | None = None) -> dict:
        """Extract context from an interviewer utterance; returns what was learned (for debug)."""
        learned = {"facts": [], "technologies": [], "project": "", "constraints": []}
        for m in _CLAIM.finditer(text):
            fact = _third_person(m.group(0))
            if len(fact.split()) >= 4 and fact not in self.important_facts:
                self.important_facts.append(fact[:260]); learned["facts"].append(fact)
        self.important_facts = self.important_facts[-self.max_facts:]
        terms = find_terms(text)
        techs = terms["programming_languages"] + terms["frameworks"] + terms["tools"]
        for w in _ACRONYM.findall(text):
            if w not in _COMMON_CAPS and w.upper() not in _COMMON_CAPS and (w.isupper() or any(c.isupper() for c in w[1:])):
                techs.append(w)
        for t in techs:
            self.technologies[t] = None; self.technologies.move_to_end(t); learned["technologies"].append(t)
        while len(self.technologies) > 12:
            self.technologies.popitem(last=False)
        low = text.lower()
        for name in self.project_names:
            if name and name.lower() in low:
                self.current_project = name; learned["project"] = name
                break
        for m in _CONSTRAINT.finditer(text):
            c = m.group(0).lower()
            if c not in self.constraints:
                self.constraints.append(c); learned["constraints"].append(c)
        self.constraints = self.constraints[-4:]
        topic = self._topic(text)
        if topic:
            self.current_topic = topic
        self.recent_questions = (self.recent_questions + [text[:300]])[-6:]
        return learned

    @staticmethod
    def _topic(text: str) -> str:
        m = re.search(r"\b(?:about|regarding|on|with|using|of)\s+(?:your\s+|the\s+|our\s+)?([A-Za-z][\w+#./ -]{2,50}?)(?:[?.,]|\s+(?:and|in|at|for|that|which|how|why|what)\b|$)", text)
        return m.group(1).strip() if m else ""

    def observe_answer(self, answer: str) -> None:
        self.recent_answers = (self.recent_answers + [answer[:300]])[-4:]

    # ----------------------------------------------------------------------------- rendering
    def memory_text(self) -> str:
        lines = []
        if self.resume_context:
            lines.append(self.resume_context)
        if self.current_project:
            lines.append(f"Project currently being discussed: {self.current_project}")
        if self.current_topic and self.current_topic.lower() != self.current_project.lower():
            lines.append(f"Current topic: {self.current_topic}")
        if self.technologies:
            lines.append("Technologies mentioned in this interview: " + ", ".join(list(self.technologies)[-8:]))
        if self.constraints:
            lines.append("Interviewer constraints: " + "; ".join(self.constraints))
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {"candidate_profile": self.candidate_profile, "resume_context": self.resume_context,
                "current_topic": self.current_topic, "current_project": self.current_project,
                "important_facts": list(self.important_facts), "recent_questions": list(self.recent_questions),
                "recent_answers": list(self.recent_answers), "technologies": list(self.technologies)}

    def to_dict(self) -> dict:       # profile/resume are rebuilt from the resume store, so not persisted here
        s = self.snapshot()
        for k in ("candidate_profile", "resume_context"):
            s.pop(k)
        s["constraints"] = list(self.constraints)
        return s

    def load_dict(self, d: dict | None) -> None:
        d = d or {}
        self.current_topic = d.get("current_topic", "")
        self.current_project = d.get("current_project", "")
        self.important_facts = list(d.get("important_facts", []))[-self.max_facts:]
        self.recent_questions = list(d.get("recent_questions", []))[-6:]
        self.recent_answers = list(d.get("recent_answers", []))[-4:]
        self.technologies = OrderedDict((t, None) for t in d.get("technologies", []))
        self.constraints = list(d.get("constraints", []))
