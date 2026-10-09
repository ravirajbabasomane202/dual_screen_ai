"""Interview-aware utterance classifier.

Types: QUESTION | INTERVIEW_PROMPT | STATEMENT | SMALL_TALK | NOISE, each with a confidence score and a reason.
Signals combined (cheap → no LLM call needed for most utterances):
    punctuation, wh/aux structure, imperative interview-prompt patterns, conversational openers
    ("I'd like to know…", "Let's start with…"), resume relevance, and conversation context (follow-ups).
The LLM is only consulted by the engine for the ambiguous middle band.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from question.detector import IMPERATIVE_RE, score_sentence


class QType(str, Enum):
    QUESTION = "QUESTION"
    INTERVIEW_PROMPT = "INTERVIEW_PROMPT"
    STATEMENT = "STATEMENT"
    SMALL_TALK = "SMALL_TALK"
    NOISE = "NOISE"


@dataclass
class Classification:
    type: QType
    confidence: float
    text: str
    reason: str
    signals: list[str] = field(default_factory=list)
    resume_relevance: float = 0.0
    resume_terms: list[str] = field(default_factory=list)
    question: str = ""                  # the sentence that carries the ask

    @property
    def is_actionable(self) -> bool:
        return self.type in (QType.QUESTION, QType.INTERVIEW_PROMPT)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


# --- interview prompt patterns: (regex, confidence, reason) ---------------------------------------------------
def _P(s: str) -> re.Pattern:
    return re.compile(s, re.I)

_YOU = r"(?:you|yourself|your\b)"
_LEAD = r"(?:(?:so|okay|ok|alright|now|well|and|then|please|just|first|firstly|to begin with|to start)[, ]+)*"
PROMPTS: list[tuple[re.Pattern, float, str]] = [
    (_P(_LEAD + r"(?:introduce|present)\s+yourself\b"), 0.97, "self-introduction prompt"),
    (_P(_LEAD + r"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:introduce|tell\s+(?:us|me)\s+about)\s+yourself\b"), 0.97, "self-introduction prompt"),
    (_P(_LEAD + r"tell\s+(?:us|me)\s+(?:a\s+(?:little|bit)\s+)?(?:more\s+)?about\s+(?:yourself|you\b|your\b|the\b|this\b|that\b|these\b|those\b|my\b|our\b|a\s+time|an?\s+(?:project|situation|challenge|instance|experience))"), 0.96, "'tell us about …' interview prompt"),
    (_P(_LEAD + r"tell\s+(?:us|me)\s+(?:more\s+)?(?:something|anything)\s+about\b"), 0.9, "'tell us something about …' prompt"),
    (_P(_LEAD + r"(?:give|share|provide)\s+(?:us|me)\s+(?:a\s+|an\s+|your\s+)?(?:brief\s+|short\s+|quick\s+|little\s+|detailed\s+|complete\s+)*(?:description|introduction|intro|overview|summary|background|picture|idea|walk[- ]?through|example|account)\b"), 0.95, "request for a description/overview"),
    (_P(_LEAD + r"(?:walk|take|run|guide)\s+(?:us|me)\s+through\b"), 0.95, "'walk us through …' prompt"),
    (_P(_LEAD + r"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:walk|take|run|guide)\s+(?:us|me)\s+through\b"), 0.96, "'walk us through …' prompt"),
    (_P(_LEAD + r"(?:describe|explain|discuss|elaborate\s+(?:on|upon)|summari[sz]e|outline|detail|highlight|share)\s+(?:to\s+(?:us|me)\s+)?(?:more\s+(?:about|on)\s+)?(?:about\s+)?(?:" + _YOU + r"|what\s+you|how\s+you|the\s+(?:role|project|work|challenges?)\s+you|(?:a|an|the)\s+(?:time|situation|project|challenge|instance|experience))"), 0.93, "describe/explain-your-… prompt"),
    (_P(_LEAD + r"(?:talk|speak)\s+(?:to\s+(?:us|me)\s+)?(?:a\s+(?:little|bit)\s+)?(?:more\s+)?(?:about|on)\s+(?:" + _YOU + r"|what\s+you|how\s+you|a\s+time|the\s+(?:most|biggest|toughest|hardest))"), 0.93, "'talk about your …' prompt"),
    (_P(_LEAD + r"(?:i'?d|i\s+would|we'?d|we\s+would)\s+(?:really\s+)?(?:like|love|want)\s+(?:to\s+(?:know|hear|understand|learn|see|get|talk|discuss|go\s+over)|you\s+to\s+(?:explain|describe|tell|walk|talk|share|elaborate|give|show|discuss))"), 0.92, "conversational request ('I'd like to know/hear…')"),
    (_P(_LEAD + r"(?:i\s+(?:want|wanna|need)|we\s+(?:want|need))\s+(?:you\s+)?to\s+(?:know|hear|understand|explain|describe|tell|walk|talk|share|elaborate|give|show)"), 0.85, "conversational request ('I want you to …')"),
    (_P(_LEAD + r"(?:i'?m|i\s+am|we'?re|we\s+are)\s+(?:very\s+|really\s+)?(?:curious|interested|keen|eager)\s+(?:to\s+(?:know|hear|understand|learn)|about|in\s+(?:hearing|knowing|your|how|what))"), 0.88, "conversational request ('I'm curious about…')"),
    (_P(_LEAD + r"let'?s\s+(?:start|begin|kick\s+(?:it\s+)?off|open)\s+(?:with|by|off\s+with)\b[^.?!]*\b(?:" + _YOU + r"|experience|background|resume|cv|projects?|role|skills?|education|career|introduction|yourself)\b"), 0.92, "interview opener ('Let's start with …')"),
    (_P(_LEAD + r"let'?s\s+(?:talk|discuss|go\s+(?:over|through)|dive\s+(?:in(?:to)?|deeper)|dig\s+(?:in(?:to)?|deeper)|explore|cover|focus\s+on|hear)\b[^.?!]*\b(?:" + _YOU + r"|experience|background|resume|cv|projects?|role|skills?|education|career|machine\s+learning|\w+ing)\b"), 0.9, "interview topic prompt ('Let's talk about …')"),
    (_P(_LEAD + r"(?:i\s+see|i\s+noticed|i\s+saw)\s+(?:on|in|from)\s+your\s+(?:resume|cv)\b[^.?!]*"), 0.78, "comment about the resume (invites a response)"),
    (_P(_LEAD + r"(?:name|list|give\s+(?:us|me)|mention|state|share)\s+(?:a\s+few\s+|some\s+|the\s+|your\s+|three\s+|two\s+|\d+\s+)?(?:of\s+)?(?:your\s+)?(?:strengths?|weakness(?:es)?|skills?|achievements?|projects?|technologies|languages|tools|hobbies|goals|expectations?)\b"), 0.93, "request to list/name something about the candidate"),
    (_P(_LEAD + r"(?:describe|explain|tell\s+(?:us|me)\s+about)\s+(?:your\s+)?(?:strengths?|weakness(?:es)?|background|experience|role|responsibilit\w+|career|education|journey|motivation|goals?)\b"), 0.94, "background prompt"),
    (_P(_LEAD + r"(?:convince|persuade)\s+(?:us|me)\b"), 0.9, "persuasion prompt"),
    (_P(_LEAD + r"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:describe|explain|elaborate|expand|share|give|show|demonstrate|clarify|summari[sz]e|outline|discuss|talk|tell|walk|go)\b"), 0.9, "polite request to explain/describe"),
    (_P(_LEAD + r"(?:describe|explain|elaborate|summari[sz]e|outline|discuss|define|clarify|compare|contrast|differentiate|illustrate|justify)\s+(?:\w+\s+){0,3}\w+"), 0.0, ""),   # handled by base imperative score
    (_P(_LEAD + r"tell\s+(?:us|me)\s+more\b"), 0.85, "follow-up prompt ('Tell me more …')"),
    (_P(_LEAD + r"(?:go|expand)\s+(?:on\s+that|deeper|further)\b|\belaborate\b"), 0.8, "follow-up prompt"),
]
PROMPTS = [p for p in PROMPTS if p[1] > 0]

SMALL_TALK_RE = _P(
    r"^\W*(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks|thank\s+you|thank\s+you\s+(?:so\s+much|very\s+much|everyone|all)|"
    r"nice\s+to\s+(?:meet|see)\s+you|welcome|great|perfect|awesome|sounds\s+good|got\s+it|i\s+see|okay|ok|cool|alright|right|sure|yes|no|yeah|"
    r"bye|goodbye|see\s+you|take\s+care|have\s+a\s+(?:good|great|nice)|no\s+problem|you'?re\s+welcome|excellent|wonderful|fantastic|that'?s\s+(?:great|good|fine))\b[\W\w]{0,40})$"
    r"|how\s+are\s+you(?:\s+(?:doing|today|going|feeling|this\s+\w+))*\W*$|how'?s\s+(?:it\s+going|your\s+day|everything)\W*$|"
    r"can\s+you\s+(?:hear|see)\s+me\b|(?:can|do)\s+you\s+(?:hear|see)\s+(?:me|my\s+screen)\b|you'?re\s+on\s+mute|i\s+think\s+you'?re\s+muted|"
    r"let\s+me\s+(?:share|just|check|pull)\b|give\s+me\s+(?:a\s+)?(?:second|moment|minute|sec)|one\s+(?:second|moment|sec)|sorry\s+(?:i'?m|for\s+being)\s+late|"
    r"(?:is\s+)?(?:the\s+)?(?:audio|sound|connection|video)\s+(?:ok|okay|fine|good|clear)|next\s+question|moving\s+on|let'?s\s+move\s+on|we(?:'ll|\s+will)\s+(?:get\s+back|be\s+in\s+touch)")
_FILLER_ONLY = re.compile(r"^(?:\W|\b(?:um+|uh+|er+|ah+|hmm+|mm+|hm+|oh|ok|okay|so|and|well|like|yeah|the|a)\b)*$", re.I)
_DEICTIC_FOLLOW = re.compile(r"\b(this|that|it|its|those|these|there|here|them|the same)\b", re.I)
_ELLIPTICAL = re.compile(r"^\W*(?:and|so|then|also|what\s+about|how\s+about|any|what's|whats)\b", re.I)
_EXPLICIT_Q = re.compile(r"^\W*(?:(?:so|and|okay|ok|now|then|well)\W+)*(?:what|why|how|when|where|which|who|whom|whose)\b", re.I)


def _norm(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    t = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|vs|e\.g|i\.e|etc|approx|St|Jr|Sr)\.", lambda m: m.group(0).replace(".", "§"), text)
    parts = [s.replace("§", ".").strip() for s in re.split(r"(?<=[.?!])\s+", t) if s.strip()]
    return parts or [text.strip()]


class QuestionClassifier:
    """`resume`: object with relevance(text)->(score, terms); `memory`: ContextManager (for follow-up context)."""

    def __init__(self, resume=None, memory=None):
        self.resume, self.memory = resume, memory

    # -------------------------------------------------------------------------------------------------
    def _prompt_score(self, sentence: str) -> tuple[float, str]:
        best = (0.0, "")
        for rx, conf, why in PROMPTS:
            if rx.search(sentence) and conf > best[0]:
                best = (conf, why)
        return best

    def classify(self, text: str) -> Classification:
        text = _norm(text)
        toks = re.findall(r"[A-Za-z0-9']+", text)
        short_prompt = bool(toks) and len(toks) <= 3 and self._prompt_score(text)[0] > 0
        if not toks or _FILLER_ONLY.match(text) or (len(toks) <= 2 and not short_prompt and not any(
                t.lower() in {"why", "how", "what", "when", "where", "which", "who"} for t in toks)
                and not SMALL_TALK_RE.match(text) and "?" not in text):
            return Classification(QType.NOISE, 0.9, text, "too short / filler only", ["noise"])

        rel, terms = (self.resume.relevance(text) if self.resume is not None else (0.0, []))
        sentences = split_sentences(text)
        best_rule, best_sent, best_sig = 0.0, sentences[-1], []
        best_prompt, prompt_why, prompt_sent = 0.0, "", ""
        for s in sentences:
            sc, sig = score_sentence(s)
            if sc > best_rule:
                best_rule, best_sent, best_sig = sc, s, sig
            ps, why = self._prompt_score(s)
            if ps > best_prompt:
                best_prompt, prompt_why, prompt_sent = ps, why, s

        small = [s for s in sentences if SMALL_TALK_RE.match(s)]
        greeting = bool(re.search(r"how\s+are\s+you|how'?s\s+(?:it|your)|(?:hear|see)\s+(?:me|my)", text, re.I))
        if small and len(small) == len(sentences) and best_prompt < 0.8 and (best_rule < 0.8 or greeting):
            return Classification(QType.SMALL_TALK, 0.9, text, "greeting / logistics / acknowledgement", ["small_talk"],
                                  rel, terms)
        # a small-talk sentence followed by a real ask ("Thanks. Tell us about yourself.") is handled below by max()

        sig = list(best_sig)
        conf, qtype, reason, ask = best_rule, QType.QUESTION, "", best_sent
        if best_prompt > conf:
            conf, qtype, reason, ask = best_prompt, QType.INTERVIEW_PROMPT, prompt_why, prompt_sent
            sig.append("interview_prompt")
        elif best_prompt >= 0.85 and best_rule >= 0.5:
            conf = min(0.99, max(best_rule, best_prompt) + 0.03)
        if qtype is QType.QUESTION:
            if "imperative" in sig and "question_mark" not in sig and not any(x in sig for x in ("wh_start", "aux_inversion")):
                qtype = QType.INTERVIEW_PROMPT
                reason = "imperative request"
            else:
                reason = "explicit question structure (" + ", ".join(sig) + ")" if sig else "weak question cues"

        # --- context: follow-ups ("And the biggest challenge?", "What about deployment?")
        active = self.memory is not None and self.memory.has_active_topic()
        if active and conf < 0.85:
            if _ELLIPTICAL.match(text) and ("?" in text or len(toks) <= 8):
                conf = max(conf, 0.68); sig.append("elliptical_follow_up"); qtype = QType.QUESTION
                reason = "short follow-up in an active conversation"
            elif _DEICTIC_FOLLOW.search(text) and (_EXPLICIT_Q.match(text) or "?" in text) and len(toks) <= 14:
                conf = min(0.99, conf + 0.12); sig.append("deictic_follow_up")
                reason = (reason + "; " if reason else "") + "refers back to the current topic"

        # --- resume relevance boosts borderline utterances; a bare statement is never promoted by relevance alone
        if rel >= 0.4 and 0.3 <= conf < 0.9:
            conf = min(0.99, conf + 0.15 * rel / 0.95 + 0.05); sig.append("resume_relevance")
            reason = (reason + "; " if reason else "") + f"mentions resume content ({', '.join(terms[:3]) or 'this project/role'})"
        # a statement the interviewer makes about the resume that addresses "you" reads as an invitation to respond
        if rel >= 0.6 and conf < 0.6 and re.search(r"\byou(?:r)?\b", text, re.I) and re.search(
                r"\b(project|role|company|experience|worked|built|developed|led|used|chose|choose|decision|challenge)\b", text, re.I) and "?" in text:
            conf = max(conf, 0.65); sig.append("resume_question_mark")

        conf = round(min(conf, 0.99), 3)
        if conf < 0.30:
            kind = QType.STATEMENT
            return Classification(kind, round(1 - conf if conf < 0.2 else 0.6, 2), text, "no question or interview-prompt structure",
                                  sig, rel, terms, best_sent)
        if conf < 0.60 and qtype is QType.QUESTION:
            reason = (reason or "weak cues") + " — ambiguous"
        return Classification(qtype, conf, text, reason, sig, rel, terms, ask)
