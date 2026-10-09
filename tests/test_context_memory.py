from context.context_manager import ContextManager
from context.token_manager import ContextBudgeter, estimate_tokens, trim_to_tokens
from llm.prompt import SYSTEM_PROMPT, build_messages

LONG_Q = ("You mentioned in your previous answer that you built a Python-based computer vision system using OpenCV and YOLO, "
          "and you said that the system was deployed on edge devices. Can you explain why you selected YOLO instead of another "
          "detection model and what challenges you faced during deployment?")


class Chunk:
    def __init__(self, label, text): self.label, self.text = label, text


def test_long_question_survives_a_tiny_context_window():
    cm = ContextManager()
    cm.observe(LONG_Q)
    for i in range(12):
        cm.add(f"Old unrelated question {i} about databases?", "Old unrelated answer. " * 60)
    kb = [Chunk(f"doc{i}.pdf", "Filler knowledge text. " * 80) for i in range(6)]
    resume = [Chunk("Project: Smart Shelf Vision", "Built an edge computer vision system with OpenCV and YOLO.")]
    plan = ContextBudgeter(window=1400, reserve_output=350).plan(
        question=LONG_Q, system_tokens=estimate_tokens(SYSTEM_PROMPT), history=cm.pairs(), kb=kb, resume=resume,
        facts=cm.facts(), memory_text=cm.memory_text())
    assert plan.total_tokens <= 1400 * 0.92
    assert plan.dropped.get("history") and not plan.over_budget
    msgs = build_messages(LONG_Q, "Short", plan.history, resume_chunks=plan.resume, kb_chunks=plan.kb, facts=plan.facts,
                          memory_text=plan.memory_text)
    last = msgs[-1]["content"]
    assert LONG_Q in last                                     # whole question, verbatim
    assert "Smart Shelf Vision" in last                       # relevant resume entry kept
    assert any("YOLO" in f for f in plan.facts) or "YOLO" in plan.memory_text   # important fact kept
    assert len(plan.history) < 12                             # old history removed first


def test_question_is_never_truncated_even_when_over_budget():
    plan = ContextBudgeter(window=300, reserve_output=100).plan(question=LONG_Q * 3, system_tokens=200, history=[("a", "b")],
                                                                kb=[Chunk("x", "y" * 500)])
    assert plan.over_budget and plan.history == [] and plan.kb == []


def test_everything_fits_when_budget_allows():
    plan = ContextBudgeter(8192, 600).plan(question="Why YOLO?", system_tokens=500, history=[("q", "a")],
                                           kb=[Chunk("k", "text")], resume=[Chunk("r", "text")], facts=["f"])
    assert plan.dropped == {} and len(plan.history) == 1 and len(plan.kb) == 1


def test_relevant_history_is_kept_over_irrelevant():
    hist = [("What is YOLO in computer vision?", "YOLO is a real-time object detector."),
            ("What is your favourite colour?", "Blue " * 100), ("Where do you live?", "City " * 100)]
    plan = ContextBudgeter(900, 300).plan(question="Why did you pick YOLO for detection?", system_tokens=300, history=hist)
    assert any("YOLO" in q for q, _ in plan.history)


def test_persistent_memory_extracts_facts_projects_and_technologies():
    cm = ContextManager()
    cm.persistent.set_profile({"name": "Jane"}, "Candidate: Jane", ["Smart Shelf Vision"])
    learned = cm.observe("Tell me about Smart Shelf Vision. You mentioned that you used YOLO on edge devices.")
    assert cm.persistent.current_project == "Smart Shelf Vision" and "YOLO" in learned["technologies"]
    assert any("YOLO" in f and "candidate" in f for f in cm.facts())
    snap = cm.snapshot()
    assert set(snap) >= {"candidate_profile", "resume_context", "current_topic", "current_project", "important_facts",
                         "recent_questions", "recent_answers"}


def test_clear_keeps_profile_but_drops_conversation():
    cm = ContextManager()
    cm.persistent.set_profile({"name": "Jane"}, "Candidate: Jane", ["P"])
    cm.observe("You mentioned that you built P with YOLO and Python."); cm.add("q", "a")
    cm.clear(keep_profile=True)
    assert cm.pairs() == [] and cm.facts() == [] and cm.persistent.candidate_profile == {"name": "Jane"}


def test_persistence_roundtrip_and_disabled_context():
    cm = ContextManager()
    cm.observe("You said that you deployed the model on a Jetson Nano board.")
    d = cm.persistent.to_dict()
    cm2 = ContextManager(); cm2.persistent.load_dict(d)
    assert cm2.facts() == cm.facts()
    cm.enabled = False
    assert cm.pairs() == [] and cm.facts() == []


def test_answers_are_not_cut_at_300_chars_any_more():
    cm = ContextManager(answer_chars=600)
    cm.add("q", "word " * 100)
    assert len(cm.pairs()[0][1]) > 300


def test_trim_to_tokens_prefers_sentence_boundaries():
    t = trim_to_tokens("First sentence here. Second sentence here. Third sentence is quite a bit longer than the rest.", 12)
    assert t.startswith("First sentence here.") and t.endswith("…")
