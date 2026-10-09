"""Raw transcript -> clean transcript -> boundary detection -> context extraction -> classification.

The full utterance is always preserved for the LLM (no character truncation): `ask` marks the interrogative part,
`context_clauses` the preceding clauses that carry facts ("You mentioned that you built …").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from question.classifier import Classification, QuestionClassifier, split_sentences

_FILLER = re.compile(r"\b(?:um+|uh+|er+|erm|ah+|hmm+)\b[,.]?\s*", re.I)
_STUTTER = re.compile(r"\b(\w+)(\s+\1\b){1,}", re.I)
_CONTEXT_CUE = re.compile(r"\b(you\s+(?:mentioned|said|stated|told|noted|built|developed|worked|used|described|explained)|"
                          r"in your (?:previous|last|earlier) (?:answer|response)|earlier you|as you said|on your resume)\b", re.I)


def clean_transcript(text: str) -> str:
    """Light, meaning-preserving cleanup: whitespace, hesitation sounds, stuttered repeats, spacing before punctuation."""
    t = " ".join(text.replace(" ", " ").split())
    t = _FILLER.sub("", t)
    t = _STUTTER.sub(r"\1", t)
    t = re.sub(r"\s+([?.!,;:])", r"\1", t)
    t = re.sub(r"\bi\b", "I", t)
    return t.strip()


@dataclass
class ProcessedQuestion:
    raw: str
    clean: str
    classification: Classification
    ask: str                                          # interrogative sentence(s)
    context_clauses: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:                            # what goes to the LLM: the complete cleaned utterance
        return self.clean


class QuestionPipeline:
    def __init__(self, classifier: QuestionClassifier):
        self.classifier = classifier

    def process(self, raw: str) -> ProcessedQuestion:
        clean = clean_transcript(raw)
        sentences = split_sentences(clean)
        cls = self.classifier.classify(clean)
        ask = cls.question or sentences[-1]
        ctx = [s for s in sentences if s != ask and _CONTEXT_CUE.search(s)]
        if ask in sentences:                           # keep every sentence from the ask to the end (multi-sentence asks)
            tail = sentences[sentences.index(ask):]
            ask = " ".join(tail) if len(tail) <= 3 else ask
        return ProcessedQuestion(raw, clean, cls, ask, ctx, sentences)
