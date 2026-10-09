"""Question-detection feedback: 'This was a question' / 'Not a question' buttons on transcript lines.

Corrections are stored locally and used twice: (1) as exact / near-exact overrides the next time the same sentence is
heard, and (2) to nudge `question.min_confidence` (bounded steps, only with several consistent signals)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

QUESTION, NOT_QUESTION = "question", "not_question"
STEP, MIN_SIGNALS, FLOOR, CEIL = 0.05, 3, 0.05, 0.60
_NORM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.lower()).split())


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x and y else 0.0


@dataclass
class FeedbackRecord:
    id: int
    ts: float
    text: str
    label: str
    detected: str
    confidence: float
    action: str


class FeedbackStore:
    def __init__(self, db):
        self.db = db

    # ---- question detection
    def record(self, text: str, label: str, detected: str = "", confidence: float = 0.0, action: str = "") -> int:
        if label not in (QUESTION, NOT_QUESTION):
            raise ValueError(label)
        norm = normalize(text)
        if not norm:
            raise ValueError("empty text")
        with self.db.transaction() as c:
            c.execute("DELETE FROM question_feedback WHERE norm=?", (norm,))     # the latest opinion wins
            cur = c.execute("INSERT INTO question_feedback(ts,text,norm,label,detected,confidence,action) VALUES(?,?,?,?,?,?,?)",
                            (time.time(), text, norm, label, detected, float(confidence), action))
            return cur.lastrowid

    def override_for(self, text: str) -> str | None:
        norm = normalize(text)
        if not norm:
            return None
        r = self.db.query_one("SELECT label FROM question_feedback WHERE norm=?", (norm,))
        if r:
            return r[0]
        if len(norm.split()) >= 4:                                  # near-duplicates of a corrected sentence
            for n, label in self.db.query("SELECT norm,label FROM question_feedback ORDER BY id DESC LIMIT 300"):
                if _jaccard(norm, n) >= 0.9:
                    return label
        return None

    def recent(self, limit: int = 100) -> list[FeedbackRecord]:
        return [FeedbackRecord(*r) for r in self.db.query(
            "SELECT id,ts,text,label,detected,confidence,action FROM question_feedback ORDER BY id DESC LIMIT ?", (limit,))]

    def counts(self) -> dict:
        rows = dict(self.db.query("SELECT label,COUNT(*) FROM question_feedback GROUP BY label"))
        return {QUESTION: rows.get(QUESTION, 0), NOT_QUESTION: rows.get(NOT_QUESTION, 0)}

    def clear(self) -> None:
        self.db.execute("DELETE FROM question_feedback")

    # ---- threshold tuning
    def _last_tuned(self) -> int:
        r = self.db.query_one("SELECT value FROM settings WHERE key='_feedback.last_tuned'")
        return int(r[0]) if r else 0

    def suggest_min_confidence(self, current: float) -> float | None:
        """New value for question.min_confidence, or None when the evidence is not consistent enough yet.

        missed questions   (user: question, we ignored it)      -> lower the floor
        false alarms       (user: not a question, we answered)  -> raise the floor"""
        since = self._last_tuned()
        rows = self.db.query("SELECT label,action,confidence FROM question_feedback WHERE id>?", (since,))
        missed = [c for l, a, c in rows if l == QUESTION and str(a).startswith("ignored")]
        false_alarm = [c for l, a, c in rows if l == NOT_QUESTION and a in ("answer", "confirm")]
        if len(missed) >= MIN_SIGNALS and len(missed) > len(false_alarm):
            return round(max(FLOOR, min(current - STEP, min(missed) - 0.01)), 2)
        if len(false_alarm) >= MIN_SIGNALS and len(false_alarm) > len(missed):
            return round(min(CEIL, max(current + STEP, max(false_alarm) + 0.01)), 2)
        return None

    def mark_tuned(self) -> None:
        last = self.db.query_one("SELECT COALESCE(MAX(id),0) FROM question_feedback")[0]
        self.db.execute("REPLACE INTO settings(key,value) VALUES('_feedback.last_tuned',?)", (str(last),))

    # ---- answer feedback (thumbs) and pinned answers
    def rate_answer(self, question: str, answer: str, rating: int, grounding: str = "", note: str = "") -> None:
        self.db.execute("INSERT INTO answer_feedback(ts,question,answer,rating,grounding,note) VALUES(?,?,?,?,?,?)",
                        (time.time(), question, answer, 1 if rating > 0 else -1, grounding, note))

    def needs_improvement(self, limit: int = 100) -> list[tuple]:
        return self.db.query("SELECT ts,question,answer,grounding,note FROM answer_feedback WHERE rating<0 "
                             "ORDER BY id DESC LIMIT ?", (limit,))

    def pin(self, question: str, answer: str) -> None:
        self.db.execute("INSERT INTO pinned_answers(ts,question,answer) VALUES(?,?,?)", (time.time(), question, answer))

    def pinned(self) -> list[tuple]:
        return self.db.query("SELECT id,ts,question,answer FROM pinned_answers ORDER BY id DESC")

    def unpin(self, pin_id: int) -> None:
        self.db.execute("DELETE FROM pinned_answers WHERE id=?", (pin_id,))
