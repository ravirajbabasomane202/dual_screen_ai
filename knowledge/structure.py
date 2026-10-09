"""Turn raw extracted page text into structural blocks (heading / paragraph / list / table / qa).

Pure and dependency-free. Heuristics only: PDF text has no real structure, so every rule here is deliberately
conservative (when unsure, a line stays part of a paragraph).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Block", "structure_text", "clean_text"]


@dataclass
class Block:
    kind: str            # heading | paragraph | list | table | qa
    text: str
    level: int = 0       # headings only: 1 = top level
    question: str = ""   # qa only: the question text without its "Q:" / "1." marker


_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", " ": " ",
              " ": " ", " ": " ", " ": "\n", " ": "\n", "•": "•"}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f­​﻿]")

_BULLET_CHARS = "-*•▪◦‣●○■□–—·►➢→"
_BULLET = re.compile(
    r"^(?P<indent>\s*)(?P<marker>[" + re.escape(_BULLET_CHARS) + r"]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+(?P<rest>\S.*)$")
_LONE_MARKER = re.compile(r"^\s*(?:[" + re.escape(_BULLET_CHARS) + r"]|\(?\d{1,3}[.)])\s*$")
_MD_HEAD = re.compile(r"^\s{0,3}(#{1,6})\s+(\S.*?)\s*#*\s*$")
_MD_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_Q_MARK = re.compile(r"^\s*(?:Q(?:uestion)?\s*\d*|Question\s+\d+)\s*[:.)\-–]\s*(?P<q>\S.*)$", re.I)
_Q_NUM = re.compile(r"^\s*Question\s+\d+\s+(?P<q>\S.*)$", re.I)
_ANSWER = re.compile(r"^\s*(?:A|Ans|Answer)\s*[:.)\-–]\s*", re.I)
_NUM_HEAD = re.compile(r"^(?P<num>\d+(?:\.\d+)+)\.?\s+(?P<t>[^\W\d_].{0,90})$")
_NUM_HEAD1 = re.compile(r"^(?P<num>\d{1,2})\s+(?P<t>[A-Z][^.!?]{1,70})$")
_NUM_DOT1 = re.compile(r"^\s*(?P<num>\d{1,2})[.)]\s+(?P<t>[A-Z][^.!?]{1,60})$")
_CELL_SPLIT = re.compile(r"\t+|\s{2,}")
_SMALL = frozenset("a an the of and or to in on for with vs via as at by from is are into over per".split())
_TERMINAL = ".!?,;"


def clean_text(text: str) -> str:
    """Normalise ligatures / control characters / newlines."""
    if not text:
        return ""
    for k, v in _LIGATURES.items():
        if k in text:
            text = text.replace(k, v)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    return _CTRL.sub("", text)


def _words(s: str) -> list[str]:
    return s.split()


def _is_all_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _is_title_case(s: str) -> bool:
    ws = _words(s)
    if not ws or not ws[0][0].isupper() and not ws[0][0].isdigit():
        return False
    caps = 0
    for i, w in enumerate(ws):
        core = w.strip("()[]\"'")
        if not core:
            continue
        if core[0].isupper() or core[0].isdigit() or any(c.isupper() for c in core[1:]):
            caps += 1
        elif core.lower() in _SMALL and i > 0:
            continue
        else:
            return False
    return caps >= 1


def _cells(line: str) -> list[str]:
    s = line.strip()
    if "|" in s:
        parts = [c.strip() for c in s.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts
    return [c.strip() for c in _CELL_SPLIT.split(s) if c.strip()]


def _row_candidate(line: str) -> bool:
    s = line.strip()
    if not s or _BULLET.match(line):
        return False
    cells = _cells(s)
    if len(cells) < 2:
        return False
    if "|" not in s and len(s) > 160:
        return False
    return all(len(c) <= 80 for c in cells)


def _join_wrapped(lines: list[str]) -> str:
    """Join hard-wrapped lines of one paragraph; repair hyphenated line breaks."""
    out = ""
    for raw in lines:
        ln = " ".join(raw.split())
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[0].islower():
            out = out[:-1] + ln
        else:
            out += " " + ln
    return out


def _normalise_marker(line: str) -> str:
    m = _BULLET.match(line)
    if not m:
        return " ".join(line.split())
    marker, rest = m.group("marker"), " ".join(m.group("rest").split())
    if marker[0] in _BULLET_CHARS and not marker[0].isdigit() and len(marker) == 1:
        return "- " + rest
    return f"{marker} {rest}"


def _question_of(line: str) -> str | None:
    m = _Q_MARK.match(line) or _Q_NUM.match(line)
    if m:
        return " ".join(m.group("q").split())
    b = _BULLET.match(line)
    if b and len(line.strip()) <= 240 and b.group("rest").rstrip().endswith("?"):
        return " ".join(b.group("rest").split())
    return None


def _strict_heading(line: str) -> int:
    """Heading level for the unambiguous kinds only (markdown, '2.1 Title', ALL CAPS); else 0."""
    s = line.strip()
    if not s or _ANSWER.match(s):
        return 0
    m = _MD_HEAD.match(line)
    if m:
        return len(m.group(1))
    m = _NUM_HEAD.match(s)
    if m and not s.endswith(tuple(_TERMINAL)) and len(_words(s)) <= 12:
        return m.group("num").count(".") + 1
    if len(s) <= 60 and _is_all_caps(s) and not s.endswith(tuple(_TERMINAL)) and len(_words(s)) <= 8 \
            and not _BULLET.match(line):
        return 1
    return 0


def _heading_info(line: str, prev_open: str, next_line: str | None) -> tuple[int, str]:
    """(level, heading text) or (0, '') for a non-heading line in plain context."""
    s = line.strip()
    if not s or len(s) > 90 or _ANSWER.match(s):
        return 0, ""
    m = _MD_HEAD.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    lvl = _strict_heading(line)
    if lvl:
        return lvl, s
    if "http" in s or "@" in s or s[0].islower():
        return 0, ""
    # a line that merely continues an unfinished sentence is never a heading
    if prev_open and len(prev_open) >= 40 and not prev_open.rstrip().endswith(tuple(_TERMINAL + ":")):
        return 0, ""
    if s.endswith(":"):
        body = s[:-1].strip()
        if body and len(_words(body)) <= 8 and next_line is not None and len(body) >= 3:
            return 3, body
        return 0, ""
    if s.endswith(tuple(_TERMINAL)) or len(_words(s)) > 10:
        return 0, ""
    m = _NUM_HEAD1.match(s)
    if m and _is_title_case(m.group("t")):
        return 1, s
    if _is_title_case(s) and not s.isdigit() and len(s) >= 3:
        return 2, s
    return 0, ""


def structure_text(page_text: str) -> list[Block]:
    """Raw page text -> ordered blocks. Never raises on odd input."""
    try:
        return _structure(page_text)
    except Exception:  # pragma: no cover - defensive: structure must never kill a document
        flat = " ".join((page_text or "").split())
        return [Block("paragraph", flat)] if flat else []


def _structure(page_text: str) -> list[Block]:
    text = clean_text(page_text or "")
    lines = [ln.rstrip() for ln in text.split("\n")]
    # a bullet glyph alone on its line belongs to the next non-empty line
    merged: list[str] = []
    i = 0
    while i < len(lines):
        if _LONE_MARKER.match(lines[i]) and i + 1 < len(lines) and lines[i + 1].strip():
            merged.append(lines[i].strip() + " " + lines[i + 1].strip())
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    lines = merged
    n = len(lines)

    # ---- pass 1: classify; table runs need >= 2 consecutive candidate rows
    kind = ["text"] * n
    for idx, ln in enumerate(lines):
        if not ln.strip():
            kind[idx] = "blank"
        elif _MD_HEAD.match(ln):
            kind[idx] = "md"
        elif _question_of(ln) is not None:
            kind[idx] = "question"
        elif _BULLET.match(ln):
            kind[idx] = "bullet"
        elif "|" in ln and _MD_SEP.match(ln):
            kind[idx] = "sep"
        elif _row_candidate(ln):
            kind[idx] = "row"
    idx = 0
    while idx < n:
        if kind[idx] in ("row", "sep"):
            j = idx
            while j < n and kind[j] in ("row", "sep"):
                j += 1
            rows = sum(1 for k in range(idx, j) if kind[k] == "row")
            if rows < 2:
                for k in range(idx, j):
                    kind[k] = "text"
            idx = j
        else:
            idx += 1

    blocks: list[Block] = []
    para: list[str] = []
    items: list[str] = []
    item_indent = 0
    table: list[str] = []
    qa_lines: list[str] = []
    qa_q: str | None = None

    def flush_para() -> None:
        if para:
            t = _join_wrapped(para)
            if t:
                blocks.append(Block("paragraph", t))
            para.clear()

    def flush_list() -> None:
        if items:
            blocks.append(Block("list", "\n".join(items)))
            items.clear()

    def flush_table() -> None:
        if table:
            blocks.append(Block("table", "\n".join(table)))
            table.clear()

    def flush_qa() -> None:
        nonlocal qa_q
        if qa_q is None:
            return
        body: list[str] = []
        cur: list[str] = []

        def push() -> None:
            if cur:
                t = _join_wrapped(cur)
                if t:
                    body.append(t)
                cur.clear()

        for ln in qa_lines[1:]:
            if not ln.strip():
                push()
            elif _ANSWER.match(ln):
                push()
                cur.append(ln)
            elif _BULLET.match(ln):
                push()
                body.append(_normalise_marker(ln))
            else:
                cur.append(ln)
        push()
        first = " ".join(qa_lines[0].split())
        blocks.append(Block("qa", "\n".join([first] + body), question=qa_q))
        qa_lines.clear()
        qa_q = None

    def flush_all() -> None:
        flush_para(); flush_list(); flush_table(); flush_qa()

    def next_nonblank(i0: int) -> str | None:
        for k in range(i0 + 1, n):
            if lines[k].strip():
                return lines[k]
        return None

    skip_until = -1
    for i, ln in enumerate(lines):
        if i <= skip_until:
            continue
        k = kind[i]

        if qa_q is not None:
            if k == "question":
                flush_qa()
            elif k == "md" or (k not in ("blank",) and _strict_heading(ln)):
                flush_qa()
            else:
                qa_lines.append(ln.strip())
                continue

        if k != "row" and k != "sep":
            flush_table()

        if k == "blank":
            flush_para()
            if items:
                nxt = next_nonblank(i)
                if nxt is None or not _BULLET.match(nxt):
                    flush_list()
            continue
        if k == "md":
            flush_all()
            m = _MD_HEAD.match(ln)
            blocks.append(Block("heading", m.group(2).strip(), level=len(m.group(1))))
            continue
        if k == "question":
            flush_all()
            q = _question_of(ln) or ""
            qa_q = q
            qa_lines.append(ln.strip())
            # a long question may wrap onto following lines
            j = i
            while not qa_q.endswith("?") and j + 1 < n and len(qa_q) < 200 and kind[j + 1] == "text" \
                    and lines[j + 1].strip() and not _ANSWER.match(lines[j + 1]) \
                    and (lines[j + 1].strip()[0].islower() or lines[j + 1].strip().endswith("?")):
                j += 1
                qa_lines[0] += " " + lines[j].strip()
                qa_q += " " + lines[j].strip()
                if lines[j].strip().endswith("?"):
                    break
            skip_until = j
            continue
        if k == "sep":
            continue
        if k == "row":
            flush_para(); flush_list()
            table.append(" | ".join(_cells(ln)))
            continue
        if k == "bullet":
            m = _BULLET.match(ln)
            nxt = next_nonblank(i)
            # "1. Introduction" followed by prose is a heading, not a one-item list
            d = _NUM_DOT1.match(ln)
            if d and not items and not para and (nxt is None or not _BULLET.match(nxt)) and nxt is not None \
                    and _is_title_case(d.group("t")) and (i == 0 or kind[i - 1] != "bullet"):
                flush_para()
                blocks.append(Block("heading", ln.strip(), level=1))
                continue
            flush_para()
            if not items:
                item_indent = len(m.group("indent"))
            items.append(_normalise_marker(ln))
            continue

        # ---- plain text line
        if items:
            indent = len(ln) - len(ln.lstrip())
            prev = items[-1]
            lvl, _ = _heading_info(ln, "", next_nonblank(i))
            if ln.strip()[0].islower() or indent > item_indent + 1 or (len(prev) >= 60 and not lvl):
                items[-1] = _join_wrapped([prev, ln])
                continue
            flush_list()
        lvl, title = _heading_info(ln, para[-1] if para else "", next_nonblank(i))
        if lvl:
            flush_para()
            blocks.append(Block("heading", title, level=lvl))
        else:
            para.append(ln)
    flush_all()
    return blocks
