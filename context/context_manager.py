"""Conversation memory: short-term turns + persistent interview facts behind one backward-compatible facade."""
from context.persistent import PersistentContext
from context.short_term import ShortTermContext


class ContextManager:
    def __init__(self, max_turns: int = 4, enabled: bool = True, answer_chars: int = 600, max_facts: int = 12):
        self.enabled = enabled
        self.short = ShortTermContext(max_turns, answer_chars)
        self.persistent = PersistentContext(max_facts)
        self.target_text = ""          # "interview target" (company / role / job description), set from the profile page

    # --- original API ---------------------------------------------------------------------------------
    def add(self, question: str, answer: str) -> None:
        self.short.add(question, answer)
        self.persistent.observe_answer(answer)

    def pairs(self) -> list[tuple[str, str]]:
        return self.short.pairs() if self.enabled else []

    def load(self, pairs) -> None:
        self.short.clear()
        for q, a in pairs:
            self.short.add(q, a)

    def clear(self, keep_profile: bool = True) -> None:
        self.short.clear()
        self.persistent.clear(keep_profile=keep_profile)

    # --- structured memory ----------------------------------------------------------------------------
    def observe(self, question: str, known_entities=None) -> dict:
        return self.persistent.observe_question(question, known_entities) if self.enabled else {}

    def facts(self) -> list[str]:
        return list(self.persistent.important_facts) if self.enabled else []

    def memory_text(self) -> str:
        base = self.persistent.memory_text() if self.enabled else self.persistent.resume_context
        return (base + "\n" + self.target_text).strip() if self.target_text else base

    def snapshot(self) -> dict:
        return self.persistent.snapshot()

    def has_active_topic(self) -> bool:
        return bool(self.persistent.current_project or self.persistent.current_topic or self.short.pairs())

    def apply_settings(self, settings) -> None:
        self.short.set_max_turns(int(settings.get("context.short_term_turns", 4)))
        self.short.answer_chars = int(settings.get("context.answer_chars", 600))
        self.persistent.max_facts = int(settings.get("context.persistent_facts", 12))
