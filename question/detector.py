"""Rule-based question detection with a 0..1 confidence score."""
import re
from dataclasses import dataclass, field

WH = {"what", "why", "how", "when", "where", "which", "who", "whom", "whose"}
AUX = {"can", "could", "would", "will", "should", "is", "are", "am", "do", "does", "did", "was", "were",
       "have", "has", "had", "may", "might", "shall"}
QUANT = {"many", "much", "long", "often", "far", "old", "big", "tall", "fast", "large", "small", "high", "wide", "deep"}
LEAD_FILLERS = {"so", "okay", "ok", "alright", "right", "now", "well", "and", "but", "um", "uh", "like", "then",
                "ah", "hey", "also", "basically", "actually", "yeah"}
_IMPERATIVES = ("explain|define|describe|compare|contrast|tell me|give me|give an example|name|list|summarize|"
                "clarify|elaborate|discuss|differentiate|illustrate|justify|outline")
IMPERATIVE_RE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|will you\s+|can someone\s+|i want you to\s+|"
    r"i'?d like you to\s+)?(?:" + _IMPERATIVES + r")\b", re.I)
CUE_RE = re.compile(
    r"\b(difference(?:s)? between|what'?s the difference|what is the difference|"
    r"how (?:do|does|did|can|would|should) (?:you|we|i|it|they|one)|what do you think|"
    r"can anyone|who can|who knows|does (?:that|this) make sense)\b", re.I)
MID_WH_RE = re.compile(r",\s*(what|why|how|when|where|which|who)\b", re.I)
TAG_RE = re.compile(r",\s*(right|correct|isn'?t it|aren'?t they|don'?t you think)\s*\??$", re.I)
MAX_UTTERANCE_CHARS = 6000   # sanity bound only; ~1500 tokens
_PRONOUNS = {"i", "we", "you", "they", "he", "she"}


@dataclass
class DetectionResult:
    text: str                       # full utterance (given to the LLM)
    question: str                   # the sentence that looks like the question
    confidence: float
    signals: list[str] = field(default_factory=list)
    classification: object = None   # question.classifier.Classification (type, reason, resume relevance, ...)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def score_sentence(sentence: str) -> tuple[float, list[str]]:
    raw = sentence.strip()
    toks = _words(raw)
    if not toks:
        return 0.0, []
    i = 0
    while i < len(toks) - 1 and toks[i] in LEAD_FILLERS:
        i += 1
    lead = " ".join(toks[i:])
    first = toks[i].split("'")[0]
    nxt = toks[i + 1].split("'")[0] if i + 1 < len(toks) else ""
    score, sig = 0.0, []

    has_q = "?" in raw
    if has_q:
        score += 0.35; sig.append("question_mark")
    wh_first = first in WH
    imperative = bool(IMPERATIVE_RE.match(lead))
    if wh_first:
        score += 0.5; sig.append("wh_start")
        if nxt in AUX or toks[i].endswith("'s"):
            score += 0.2; sig.append("wh_aux")
        elif nxt in QUANT and len(toks) - i >= 4:        # "how many / how much / how long ..."
            score += 0.35; sig.append("wh_quantity")
        elif any(t.split("'")[0] in AUX for t in toks[i + 2:i + 8]) and nxt not in _PRONOUNS:
            score += 0.3; sig.append("wh_aux_later")      # "which database would you use"
        if nxt in _PRONOUNS:
            score -= 0.3; sig.append("embedded_clause")
    elif first in AUX and len(toks) - i >= 3:
        score += 0.4; sig.append("aux_inversion")
    if imperative:
        score += 0.5; sig.append("imperative")
        if any(t.split("'")[0] in WH for t in toks[i + 1:]):
            score += 0.35; sig.append("imperative_wh")
    if not wh_first and MID_WH_RE.search(raw):
        score += 0.5; sig.append("wh_after_comma")
    if CUE_RE.search(raw):
        score += 0.25; sig.append("cue_phrase")
    if TAG_RE.search(raw):
        score += 0.2; sig.append("tag_question")
    if len(toks) <= 3 and not (wh_first or imperative):
        score = min(score, 0.2)
    return round(max(0.0, min(score, 0.99)), 3), sig


class QuestionDetector:
    """Backward-compatible entry point: `detect(text) -> DetectionResult` whose `confidence` is the probability that
    the utterance is something to answer (explicit question OR interview prompt). Non-actionable types
    (statement / small talk / noise) always score below the semantic-check floor."""

    def __init__(self, resume=None, memory=None):
        from question.classifier import QuestionClassifier
        self.classifier = QuestionClassifier(resume, memory)

    def detect(self, text: str) -> DetectionResult:
        text = " ".join(text.split())
        c = self.classifier.classify(text)
        conf = c.confidence if c.is_actionable else min(0.2, round(1 - c.confidence, 3) if c.confidence > 0.5 else c.confidence)
        if not c.is_actionable and c.type.value == "STATEMENT":
            # keep the rule score for statements so borderline wording can still reach the semantic check
            conf = min(0.29, max(score_sentence(c.question or text)[0], 0.0))
        # long, multi-clause questions are kept whole (token budgeting happens later, never character truncation)
        return DetectionResult(text=text[-MAX_UTTERANCE_CHARS:], question=c.question or text, confidence=conf,
                               signals=c.signals, classification=c)
