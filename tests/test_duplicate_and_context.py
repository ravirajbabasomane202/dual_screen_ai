from context.context_manager import ContextManager
from question.cleaner import basic_clean
from question.duplicate_filter import DuplicateFilter, normalize, similarity


class Clock:
    t = 0.0
    def __call__(self): return self.t


def test_normalize_and_similarity():
    assert normalize("So, um, what IS overfitting?") == "what is overfitting"
    assert similarity("What is overfitting?", "what is overfitting basically") >= 0.8
    assert similarity("What is overfitting?", "What is regression?") < 0.8


def test_duplicate_within_timeout_only():
    clk = Clock()
    f = DuplicateFilter(timeout=30, clock=clk)
    assert not f.is_duplicate("What is supervised learning?")
    f.register("What is supervised learning?")
    clk.t = 10
    assert f.is_duplicate("what is supervised learning")
    assert not f.is_duplicate("What are its examples?")
    clk.t = 31
    assert not f.is_duplicate("What is supervised learning?")


def test_embedding_duplicate():
    vecs = {"a b c": [1, 0], "x y z": [0.99, 0.1], "other": [0, 1]}
    f = DuplicateFilter(embed_fn=lambda t: vecs[t], clock=Clock())
    f.register("a b c")
    assert f.is_duplicate("x y z")
    assert not f.is_duplicate("other")


def test_basic_clean():
    assert basic_clean("so um can you tell me like what is the diff between a and b basically") \
        == "Can you tell me what is the diff between a and b?"


def test_context_window_and_toggle():
    c = ContextManager(max_turns=2)
    for i in range(3):
        c.add(f"q{i}", f"a{i}")
    assert [q for q, _ in c.pairs()] == ["q1", "q2"]
    c.enabled = False
    assert c.pairs() == []
    c.enabled = True
    c.clear()
    assert c.pairs() == []
