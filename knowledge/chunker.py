"""Split a parsed document into retrieval chunks that respect sections, Q&A pairs, lists and tables."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from context.token_manager import estimate_tokens
from knowledge.pdf_parser import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    document: str
    page: int
    section: str
    kind: str            # text | qa | list | table
    question: str
    text: str
    tokens: int


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _split_long(text: str, target: int) -> list[str]:
    out, cur = [], ""
    for s in _sentences(text) or [text]:
        if cur and estimate_tokens(cur + " " + s) > target:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def chunk_document(doc: ParsedDocument, target_tokens: int = 220, overlap_tokens: int = 40) -> list[Chunk]:
    raw: list[dict] = []               # {page, section, kind, question, text}
    stack: list[tuple[int, str]] = []  # (level, title)
    buf: list[str] = []
    buf_meta = {"page": 1, "section": ""}

    def section() -> str:
        return " > ".join(t for _, t in stack)

    def flush(overlap: bool = True) -> None:
        if buf:
            raw.append({**buf_meta, "kind": "text", "question": "", "text": "\n".join(buf)})
            buf.clear()

    def add_piece(kind, page, text, question=""):
        raw.append({"page": page, "section": section(), "kind": kind, "question": question, "text": text})

    for pg in doc.pages:
        for b in pg.blocks:
            if b.kind == "heading":
                flush()
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                stack.append((b.level, b.text))
                continue
            if b.kind == "qa":
                flush()
                if estimate_tokens(b.text) <= target_tokens * 2.5:
                    add_piece("qa", pg.number, b.text, b.question)
                else:
                    head, _, rest = b.text.partition("\n")
                    for part in _split_long(rest, target_tokens):
                        add_piece("qa", pg.number, head + "\n" + part, b.question)
                continue
            if b.kind in ("list", "table"):
                flush()
                lines, cur = b.text.split("\n"), []
                for ln in lines:
                    if cur and estimate_tokens("\n".join(cur + [ln])) > target_tokens:
                        add_piece(b.kind, pg.number, "\n".join(cur)); cur = []
                    cur.append(ln)
                if cur:
                    add_piece(b.kind, pg.number, "\n".join(cur))
                continue
            for piece in _split_long(b.text, target_tokens):    # paragraph
                cur_text = "\n".join(buf)
                if buf and (estimate_tokens(cur_text + "\n" + piece) > target_tokens
                            or buf_meta["section"] != section()):
                    tail = _sentences(buf[-1])[-1:] if (buf_meta["section"] == section() and overlap_tokens) else []
                    flush()
                    buf.extend(t for t in tail if estimate_tokens(t) <= overlap_tokens)
                if not buf:
                    buf_meta.update(page=pg.number, section=section())
                buf.append(piece)
    flush()

    chunks: list[Chunk] = []
    for i, r in enumerate(raw):
        text = r["text"].strip()
        if len(text.split()) < 8 and not (len(raw) == 1 or r["kind"] in ("qa", "table", "list") and len(text.split()) >= 3):
            continue
        cid = hashlib.sha1(f"{doc.sha256}|{i}|{text[:50]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(cid, doc.name, r["page"], r["section"], r["kind"], r["question"], text,
                            estimate_tokens(text)))
    return chunks
