"""Recent conversation turns (bounded, newest last)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Turn:
    question: str
    answer: str
    ts: float = field(default_factory=time.time)


class ShortTermContext:
    def __init__(self, max_turns: int = 4, answer_chars: int = 600):
        self.answer_chars = answer_chars
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, question: str, answer: str) -> None:
        a = answer if len(answer) <= self.answer_chars else answer[: self.answer_chars].rsplit(" ", 1)[0] + " …"
        self._turns.append(Turn(question, a))

    def pairs(self) -> list[tuple[str, str]]:
        return [(t.question, t.answer) for t in self._turns]

    def last_ts(self) -> float | None:
        return self._turns[-1].ts if self._turns else None

    def set_max_turns(self, n: int) -> None:
        if n != self._turns.maxlen:
            self._turns = deque(self._turns, maxlen=max(1, n))

    def clear(self) -> None:
        self._turns.clear()
