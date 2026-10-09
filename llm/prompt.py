"""Prompts, JSON contract, tolerant parsing, and an incremental JSON-field streamer.

Output contract (keys in this order, so the model decides WHAT IS BEING ASKED before it writes the answer):
    {"question": "...", "intent": "...", "grounding": "resume|knowledge_base|conversation|general|unknown",
     "answer": "...", "confidence": 0.0}
The legacy contract ({"is_question", "confidence", "clean_question", "answer", "needs_clarification"}) is still parsed.
"""
import json
import re
from dataclasses import dataclass

from context.token_manager import estimate_tokens

SYSTEM_PROMPT = """You are the candidate in a live job interview, answering out loud. Text after "QUESTION:" is exactly what the interviewer just said (speech recognition: it may contain small errors). Write the answer the candidate should SAY, in the first person.

RULES
1. ANSWER the question. Never restate, paraphrase, summarise or describe it.
2. Never write phrases like "The speaker is asking", "The interviewer is asking / wants to know", "The question asks", "This question is about", "They want to know".
3. Fix obvious speech-recognition errors silently; keep the meaning. A long question has several parts: answer every part.
4. General or technical questions: answer correctly and concisely from general knowledge.
5. Questions about the candidate (background, experience, projects, "tell me about yourself"): use ONLY facts found in CANDIDATE PROFILE, CANDIDATE RESUME, INTERVIEW FACTS or earlier turns. NEVER invent employers, projects, numbers, dates or technologies. If a needed fact is missing, answer with what is known and speak generally and honestly about the rest.
6. Estimation questions (e.g. "how many red Swift cars are on the road?"): give a short reasoned estimate with a number. Do not refuse and do not restate.
7. Use REFERENCE NOTES when relevant, ignore them when not.
8. Follow the length rule below. Plain spoken English; at most a few short bullets.

OUTPUT: exactly one JSON object, keys in this order:
{"question": "<cleaned question>", "intent": "<what the interviewer wants to find out, max 15 words>", "grounding": "<resume|knowledge_base|conversation|general|unknown>", "answer": "<the spoken answer>", "confidence": <0.0-1.0>}

EXAMPLES
QUESTION: what is python
{"question": "What is Python?", "intent": "definition of Python", "grounding": "general", "answer": "Python is a high-level, interpreted programming language known for readable syntax. It is widely used for web backends, data science, automation and AI.", "confidence": 0.97}
QUESTION: how many red swift cars are on the road
{"question": "How many red Swift cars are currently on the road?", "intent": "a reasoned estimate of a number", "grounding": "general", "answer": "I'd estimate around 70,000. Roughly 3 million Swifts have been sold in India, about 6% of them are red, and perhaps 40% are still on the road: 3,000,000 x 6% x 40% is about 72,000. I would refine this with registration data by colour.", "confidence": 0.5}"""

CLASSIFY_PROMPT = """You decide whether a speech-recognition transcript from a job interview contains something the candidate must answer.
YES (is_question true): direct questions, and requests that are not grammatical questions, such as "Tell us about yourself", "Introduce yourself", "Walk me through your resume", "Describe your background", "Let's talk about your experience", "I'd like to know more about your project".
NO: greetings, thanks, logistics ("can you hear me"), statements by the interviewer, remarks like "isn't that interesting".
Return ONLY JSON: {"is_question": true/false, "confidence": 0.0-1.0, "clean_question": "grammar-corrected question or empty"}"""

REPAIR_NOTE = ("IMPORTANT: your previous reply {why}. Do NOT describe or repeat the question. Start the \"answer\" "
               "directly with the answer itself, in the first person, as the candidate speaking.")

LENGTH_RULES = {
    "Very Short": ("Answer in 1-2 short sentences (max 40 words). No examples.", 90),
    "Short": ("Answer in 2-4 short sentences (max 80 words). Add a one-line example only if it really helps.", 200),
    "Medium": ("Answer in about 120-150 words. Include one concrete example.", 380),
    "Detailed": ("Answer in up to 300 words with a clear structure and an example.", 750),
}
STYLE_RULES = {
    "Concise": "",
    "STAR": "STYLE: structure behavioural answers as STAR (Situation, Task, Action, Result), spoken naturally, "
            "without printing the four labels. Use only real facts from the candidate's profile.",
    "Technical deep-dive": "STYLE: technical deep-dive. Explain the mechanism, the trade-offs and one concrete example "
                           "or pitfall; use precise terms. Stay within the length rule.",
    "Spoken-natural": "STYLE: sound like a calm person talking, not writing: short sentences, contractions, no lists, "
                      "no headings.",
}
STYLE_PRESETS = tuple(STYLE_RULES)
GROUNDINGS = ("resume", "knowledge_base", "conversation", "general", "unknown")

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"}, "intent": {"type": "string"},
        "grounding": {"type": "string", "enum": list(GROUNDINGS)},
        "answer": {"type": "string"}, "confidence": {"type": "number"},
    },
    "required": ["question", "intent", "grounding", "answer", "confidence"],
}
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"is_question": {"type": "boolean"}, "confidence": {"type": "number"},
                   "clean_question": {"type": "string"}},
    "required": ["is_question", "confidence", "clean_question"],
}


@dataclass
class LLMResult:
    is_question: bool = False
    confidence: float = 0.0
    clean_question: str = ""
    answer: str = ""
    needs_clarification: bool = False
    intent: str = ""
    grounding: str = ""
    valid: bool = True                 # set by the answer validator
    validation: dict | None = None
    attempts: int = 1
    from_plain_text: bool = False


class ParseError(ValueError):
    pass


def _block(title: str, items) -> str:
    body = "\n".join(f"[{getattr(c, 'label', '') or 'note'}]\n{getattr(c, 'text', c)}" for c in items)
    return f"{title}:\n{body}\n"


def build_messages(transcript: str, length: str, history: list[tuple[str, str]], *, resume_chunks=None,
                   kb_chunks=None, facts=None, memory_text: str = "", repair: str = "", style: str = "") -> list[dict]:
    rule, _ = LENGTH_RULES.get(length, LENGTH_RULES["Short"])
    style_rule = STYLE_RULES.get(style, "")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nLENGTH RULE: " + rule + (("\n" + style_rule) if style_rule else "")}]
    for q, a in history:
        msgs.append({"role": "user", "content": f"QUESTION: {q}"})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"question": q, "intent": "", "grounding": "conversation", "answer": a, "confidence": 0.9})})
    parts = []
    if memory_text:
        parts.append(f"CANDIDATE PROFILE:\n{memory_text}\n")
    if facts:
        parts.append("INTERVIEW FACTS (said earlier in this interview):\n" + "\n".join(f"- {f}" for f in facts) + "\n")
    if resume_chunks:
        parts.append(_block("CANDIDATE RESUME (verified facts)", resume_chunks))
    if kb_chunks:
        parts.append(_block("REFERENCE NOTES", kb_chunks))
    if repair:
        parts.append(REPAIR_NOTE.format(why=repair) + "\n")
    parts.append(f"QUESTION: {transcript}")
    msgs.append({"role": "user", "content": "\n".join(parts)})
    return msgs


def build_classify_messages(transcript: str) -> list[dict]:
    return [{"role": "system", "content": CLASSIFY_PROMPT}, {"role": "user", "content": f"Transcript: {transcript}"}]


def max_tokens_for(length: str, question: str = "", override: int = 0) -> int:
    """Answer budget + JSON keys + the echoed question/intent (a long question must not truncate the answer)."""
    if override and override > 0:
        return int(override)
    base = LENGTH_RULES.get(length, LENGTH_RULES["Short"])[1]
    return base + 200 + int(estimate_tokens(question) * 1.2)


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def parse_llm_json(raw: str) -> LLMResult:
    text = _THINK.sub("", raw).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    data = None
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            data = None
    if data is not None and not isinstance(data, dict):
        data = None
    if data is None:                       # truncated / sloppy JSON: salvage fields by regex
        data = {}
        if m := re.search(r'"is_question"\s*:\s*(true|false)', text):
            data["is_question"] = m.group(1) == "true"
        if m := re.search(r'"confidence"\s*:\s*([0-9.]+)', text):
            try:
                data["confidence"] = float(m.group(1).rstrip("."))
            except ValueError:
                pass
        if m := re.search(r'"grounding"\s*:\s*"(\w+)"', text):
            data["grounding"] = m.group(1)
        for key in ("clean_question", "question", "intent", "answer"):
            s = JSONFieldStreamer(key)
            val = s.feed(text)
            if val:
                data[key] = val
        if not data:
            raise ParseError("No JSON found in LLM output")
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    g = str(data.get("grounding", "") or "").strip().lower()
    return LLMResult(
        is_question=bool(data["is_question"]) if "is_question" in data else True,
        confidence=max(0.0, min(conf, 1.0)),
        clean_question=str(data.get("clean_question") or data.get("question") or "").strip(),
        answer=str(data.get("answer", "") or "").strip(),
        needs_clarification=bool(data.get("needs_clarification", False)),
        intent=str(data.get("intent", "") or "").strip(),
        grounding=g if g in GROUNDINGS else "")


class JSONFieldStreamer:
    """Feed raw JSON chunks; get the decoded text of one string field as soon as it is available."""

    _ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}

    def __init__(self, field: str):
        self._pat = re.compile(r'"%s"\s*:\s*"' % re.escape(field))
        self._buf, self._pos, self.done, self.value = "", None, False, ""

    def feed(self, chunk: str) -> str:
        if self.done:
            return ""
        self._buf += chunk
        if self._pos is None:
            m = self._pat.search(self._buf)
            if not m:
                return ""
            self._pos = m.end()
        s, i, out = self._buf, self._pos, []
        while i < len(s):
            c = s[i]
            if c == '"':
                self.done, i = True, i + 1
                break
            if c == "\\":
                if i + 1 >= len(s):
                    break
                n = s[i + 1]
                if n == "u":
                    if i + 6 > len(s):
                        break
                    try:
                        out.append(chr(int(s[i + 2:i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                out.append(self._ESC.get(n, n))
                i += 2
                continue
            out.append(c)
            i += 1
        self._pos = i
        piece = "".join(out)
        self.value += piece
        return piece
