"""Answer validation: reject replies that describe the question instead of answering it.

Two entry points:
  * `summary_prefix(text)`   – cheap check on the first words of a still-streaming answer
  * `validate_answer(...)`   – full check once the answer is complete
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_ROLE = r"(?:speaker|interviewer|interviewee|person|host|panel(?:ist)?s?|recruiter|hiring manager|candidate|user|they|he|she)"
_VERB = (r"(?:is|are|was|were|has|have|wants?|would\s+like|seems?|appears?|asks?|asked|asking|wondering|inquir\w+|requests?|"
         r"requesting|curious|trying|looking|interested|expects?)")
SUSPICIOUS = [
    (re.compile(rf"^\W*(?:so,?\s+|here,?\s+|it\s+(?:seems|looks|sounds)\s+(?:like|as\s+if)\s+)?(?:the\s+)?{_ROLE}\s+{_VERB}\b", re.I),
     "describes what the speaker/interviewer wants"),
    (re.compile(r"^\W*(?:the|this|that|their|his|her)\s+(?:question|query|prompt|request|inquiry)\s+"
                r"(?:is|asks?|about|refers?|concerns?|seeks?|wants?|relates?|pertains?|appears?|seems?)\b", re.I),
     "describes the question"),
    (re.compile(r"^\W*(?:you(?:'re|\s+are)\s+asking|you\s+(?:want|would\s+like)\s+to\s+know|you\s+asked)\b", re.I),
     "echoes the question back"),
    (re.compile(r"^\W*this\s+(?:is\s+)?(?:a\s+)?(?:question|query|prompt)\b", re.I), "labels the question"),
    (re.compile(r"^\W*(?:question|q|interviewer|speaker)\s*[:\-–]\s", re.I), "starts with a question label"),
    (re.compile(r"\b(?:is|are)\s+asking\s+(?:how|what|why|whether|if|about|for|when|where|which|who)\b", re.I),
     "contains 'is asking …'"),
    (re.compile(r"\bwants?\s+to\s+know\s+(?:how|what|why|whether|if|about|when|where|which|who)\b", re.I),
     "contains 'wants to know …'"),
]
REFUSAL = re.compile(r"^\W*(?:as\s+an\s+ai\b|i(?:'m|\s+am)\s+(?:just\s+)?an?\s+(?:ai|language\s+model)|i\s+(?:cannot|can't)\s+(?:answer|help|assist))", re.I)
_STOP = frozenset("a an the of to in on at by for from with and or but is are was were be been am do does did have has had i me my we us "
                  "our you your he she it its they them their this that these those there here what which who whom whose when where why "
                  "how can could would should will shall may might must not no about into".split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#]+", s.lower()) if w not in _STOP and len(w) > 1}


def summary_prefix(text: str) -> str | None:
    """Reason string when the beginning of an answer already looks like a question summary, else None."""
    head = text[:200]
    for rx, why in SUSPICIOUS:
        if rx.search(head):
            return why
    return None


@dataclass
class ValidationResult:
    valid: bool = True
    reasons: list[str] = field(default_factory=list)
    kind: str = "ok"                    # ok | summary | restatement | empty | refusal
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _first_sentence(t: str) -> str:
    m = re.split(r"(?<=[.!?])\s+|\n", t.strip(), maxsplit=1)
    return m[0] if m else t


def validate_answer(question: str, answer: str, *, intent: str = "", allow_analysis: bool = False,
                    grounding: str = "", sources_text: str = "") -> ValidationResult:
    a = (answer or "").strip()
    res = ValidationResult()
    if not a:
        return ValidationResult(False, ["empty answer"], "empty")
    if not allow_analysis:
        why = summary_prefix(a)
        if why:
            return ValidationResult(False, [why], "summary")
        for sent in re.split(r"(?<=[.!?])\s+", a)[:3]:
            for rx, w in SUSPICIOUS[5:]:
                if rx.search(sent):
                    return ValidationResult(False, [w], "summary")
    if REFUSAL.search(a):
        return ValidationResult(False, ["generic AI refusal instead of an interview answer"], "refusal")
    qa, aa = _content(question), _content(_first_sentence(a))
    if len(aa) >= 4 and qa:
        containment = len(aa & qa) / len(aa)
        jaccard = len(aa & qa) / len(aa | qa)
        if containment >= 0.85 and jaccard >= 0.6:
            return ValidationResult(False, [f"first sentence repeats the question (overlap {containment:.0%})"], "restatement")
    if a.rstrip().endswith("?") and len(a.split()) <= 25 and "." not in a:
        return ValidationResult(False, ["answer is itself a question"], "restatement")
    if grounding in ("resume", "knowledge_base") and sources_text:
        known = set(re.findall(r"\d[\d,.]*", sources_text + " " + question))
        unsupported = [n for n in re.findall(r"\d[\d,.]*", a) if n.strip(",.") and n not in known]
        if unsupported:
            res.warnings.append("numbers not found in the cited source: " + ", ".join(list(dict.fromkeys(unsupported))[:5]))
    if grounding in ("resume", "knowledge_base") and not sources_text:
        res.warnings.append(f"answer claims '{grounding}' grounding but no such source was provided")
    return res
