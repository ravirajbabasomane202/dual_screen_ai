"""Confidence thresholds and combination of rule-based + LLM scores."""
from enum import Enum

SEMANTIC_FLOOR = 0.30   # below this the rules alone say "not a question": don't even ask the LLM


class Action(Enum):
    AUTO = "auto"           # process immediately
    CONFIRM = "confirm"     # show "possible question" and wait for the user
    IGNORE = "ignore"


def decide(conf: float, auto: float = 0.85, confirm: float = 0.60) -> Action:
    if conf >= auto:
        return Action.AUTO
    if conf >= confirm:
        return Action.CONFIRM
    return Action.IGNORE


def combine(rule_conf: float, llm_is_question: bool, llm_conf: float) -> float:
    if not llm_is_question:
        return round(min(rule_conf, 0.3) * 0.5, 3)
    return round(0.4 * rule_conf + 0.6 * llm_conf, 3)
