import pytest
from question.confidence import Action, combine, decide
from question.detector import QuestionDetector

D = QuestionDetector()


@pytest.mark.parametrize("text", [
    "What is the difference between classification and regression?",
    "so can you tell me like what is the diff between classification and regression basically",
    "Classification and regression, what's the difference?",
    "Explain how gradient descent works",
    "How does backpropagation compute the gradients?",
])
def test_clear_questions_are_auto(text):
    assert decide(D.detect(text).confidence) is Action.AUTO, D.detect(text)


@pytest.mark.parametrize("text", [
    "Classification and regression.",
    "Today we will talk about overfitting.",
    "Okay let's move on to the next slide.",
    "Thank you everyone.",
    "What I mean is that the model memorises the training data.",
])
def test_statements_are_ignored(text):
    assert decide(D.detect(text).confidence) is Action.IGNORE, D.detect(text)


def test_ambiguous_goes_to_semantic_check_zone():
    r = D.detect("why do models overfit")
    assert 0.3 <= r.confidence < 0.85


def test_question_sentence_is_found_inside_longer_speech():
    r = D.detect("Imagine a model that memorises everything. Why is that bad?")
    assert r.question == "Why is that bad?" and r.confidence >= 0.85
    assert "memorises" in r.text   # full utterance is kept for the LLM


def test_thresholds():
    assert decide(0.9) is Action.AUTO
    assert decide(0.7) is Action.CONFIRM
    assert decide(0.59) is Action.IGNORE
    assert decide(0.7, auto=0.6) is Action.AUTO


def test_combine():
    assert combine(0.7, True, 0.95) >= 0.85
    assert combine(0.7, False, 0.9) < 0.3
