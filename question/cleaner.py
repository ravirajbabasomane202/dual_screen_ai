"""Question cleaning. The LLM does the real work (llm/service.py); this is the offline fallback
and the text normaliser used before display."""
import re

_FILLERS = re.compile(r"\b(um+|uh+|er+|ah+|you know|i mean|kind of|sort of|like|basically|actually|literally)\b[, ]*",
                      re.I)


def basic_clean(text: str, is_question: bool = True) -> str:
    t = _FILLERS.sub("", text)
    t = re.sub(r"^(?:so|okay|ok|alright|well|now|and)\b[, ]*", "", t.strip(), flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if is_question and not t.endswith("?"):
        t = t.rstrip(".!") + "?"
    return t
