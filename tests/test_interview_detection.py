import pytest

from question.classifier import QType, QuestionClassifier
from question.confidence import Action, decide
from question.detector import QuestionDetector
from question.pipeline import QuestionPipeline, clean_transcript

C = QuestionClassifier()

ACTIONABLE = [
    "Tell us about yourself.", "Introduce yourself.", "Introduce yourself to the board members.",
    "Give us a brief description about yourself.", "Give me a brief description about yourself.",
    "Walk us through your resume.", "Walk me through your resume.", "Tell us about your experience.",
    "Take us through your previous projects.", "Describe your background.", "Tell me about your current role.",
    "Give me a brief introduction.", "Explain your experience with AI.", "Talk about your most challenging project.",
    "What is your experience with Python?", "Why should we hire you?", "What are your strengths?",
    "Tell me about your previous project.", "Why did you choose Python?", "Can you explain your project?",
    "Let's talk about your previous role.", "I'd like to know more about your background.",
    "Let's start with your experience.", "Can you walk us through what you built?",
    "I'd like you to explain how this works.", "Let's talk about your experience with machine learning.",
    "How many Red Swift cars are currently on the road", "Why did you leave your previous company?",
    "What technologies did you use?", "Tell me about this project.", "Tell me more about your experience here.",
    "Thanks. Tell us about yourself.",
]
IGNORED = ["Thank you everyone.", "Today we will talk about overfitting.", "Okay let's move on to the next slide.",
           "Classification and regression.", "How are you doing today?", "Can you hear me?", "um", "Nice to meet you."]


@pytest.mark.parametrize("text", ACTIONABLE)
def test_interview_questions_and_prompts_are_detected(text):
    r = C.classify(text)
    assert r.is_actionable and r.confidence >= 0.8, r
    assert r.type in (QType.QUESTION, QType.INTERVIEW_PROMPT)
    assert decide(QuestionDetector().detect(text).confidence) is Action.AUTO


@pytest.mark.parametrize("text", IGNORED)
def test_non_questions_are_not_answered(text):
    assert not C.classify(text).is_actionable
    assert decide(QuestionDetector().detect(text).confidence) is Action.IGNORE


def test_non_grammatical_prompts_are_typed_as_interview_prompt():
    for t in ("Tell us about yourself.", "Introduce yourself.", "Walk us through your resume."):
        assert C.classify(t).type is QType.INTERVIEW_PROMPT
    assert C.classify("What are your strengths?").type is QType.QUESTION


def test_classification_has_reason_and_serialises():
    d = C.classify("Tell us about yourself").to_dict()
    assert d["type"] == "INTERVIEW_PROMPT" and d["confidence"] > 0.9 and d["reason"] and d["text"]


def test_long_question_is_not_truncated():
    q = ("You mentioned in your previous answer that you built a Python-based computer vision system using OpenCV and YOLO, "
         "and you said that the system was deployed on edge devices. Can you explain why you selected YOLO instead of another "
         "detection model and what challenges you faced during deployment?")
    det = QuestionDetector().detect(q)
    assert det.text == q and len(q) > 250 and det.confidence >= 0.85


def test_pipeline_extracts_context_clauses_and_keeps_full_text():
    q = ("You mentioned that you built a vision system using OpenCV. Can you explain why you selected YOLO and what "
         "challenges you faced?")
    p = QuestionPipeline(C).process("um " + q)
    assert p.text == q and p.context_clauses and "OpenCV" in p.context_clauses[0] and p.ask.startswith("Can you explain")


def test_clean_transcript_is_meaning_preserving():
    assert clean_transcript("so um what is the the difference between A and B ?") == "so what is the difference between A and B?"


class FakeResume:
    def relevance(self, text):
        return (0.7, ["acme corp"]) if "acme" in text.lower() else (0.0, [])


def test_resume_relevance_boosts_borderline_utterances():
    base = QuestionClassifier().classify("so at acme you did what")
    boosted = QuestionClassifier(resume=FakeResume()).classify("so at acme you did what")
    assert boosted.confidence >= base.confidence and boosted.resume_relevance == 0.7


class Memory:
    def has_active_topic(self): return True


def test_context_makes_elliptical_followups_questions():
    assert not QuestionClassifier().classify("And the biggest challenge?").confidence >= 0.6
    r = QuestionClassifier(memory=Memory()).classify("And the biggest challenge?")
    assert r.is_actionable and r.confidence >= 0.6
