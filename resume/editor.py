"""Text codecs for the 'Edit parsed fields' mode: structured resume fields <-> a small, human-editable text format.
Pure logic (no Qt) so it is unit-tested; the UI only shows these strings in text boxes.

    Experience / projects: blocks separated by a blank line
        Senior Engineer | Acme Corp | Jan 2020 | Present
        - Built a fraud detection system
        - Led a team of 5
    Education: one per line   Degree | Institution | Dates
    Lists (skills, tools, ...): comma- or line-separated
"""
from __future__ import annotations


def parse_list(text: str) -> list[str]:
    items = []
    for chunk in text.replace("\n", ",").replace(";", ",").split(","):
        c = chunk.strip(" \t•-*")
        if c and c.lower() not in (i.lower() for i in items):
            items.append(c)
    return items


def format_list(items) -> str:
    return ", ".join(str(i) for i in items or [])


def _blocks(text: str) -> list[list[str]]:
    out, cur = [], []
    for line in text.splitlines():
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _bullets(lines: list[str]) -> list[str]:
    return [l.strip().lstrip("-•* ").strip() for l in lines if l.strip().startswith(("-", "•", "*")) and l.strip().lstrip("-•* ")]


def format_experience(entries) -> str:
    blocks = []
    for e in entries or []:
        head = " | ".join([e.get("title", ""), e.get("company", ""), e.get("start", ""), e.get("end", "")])
        blocks.append("\n".join([head] + [f"- {b}" for b in e.get("bullets", [])]))
    return "\n\n".join(blocks)


def parse_experience(text: str, old=None) -> list[dict]:
    """`old` (the previous entries) lets untouched entries keep fields the text format cannot show (technologies...)."""
    old = old or []
    out = []
    for blk in _blocks(text):
        head = [p.strip() for p in blk[0].split("|")] + ["", "", "", ""]
        title, company, start, end = head[:4]
        bullets = _bullets(blk[1:])
        entry = {"title": title, "company": company, "start": start, "end": end, "location": "", "bullets": bullets,
                 "responsibilities": list(bullets), "achievements": [], "technologies": []}
        for o in old:
            if o.get("title") == title and o.get("company") == company:
                entry["location"] = o.get("location", "")
                entry["technologies"] = list(o.get("technologies", []))
                entry["achievements"] = [a for a in o.get("achievements", []) if a in bullets]
                break
        out.append(entry)
    return out


def format_projects(entries) -> str:
    blocks = []
    for p in entries or []:
        head = " | ".join([p.get("name", ""), p.get("description", ""), ", ".join(p.get("technologies", []))])
        blocks.append("\n".join([head] + [f"- {b}" for b in p.get("bullets", [])]))
    return "\n\n".join(blocks)


def parse_projects(text: str) -> list[dict]:
    out = []
    for blk in _blocks(text):
        head = [p.strip() for p in blk[0].split("|")] + ["", ""]
        out.append({"name": head[0], "description": head[1], "technologies": parse_list(head[2]), "bullets": _bullets(blk[1:])})
    return out


def format_education(entries) -> str:
    return "\n".join(" | ".join([e.get("degree", ""), e.get("institution", ""), e.get("dates", "")]) for e in entries or [])


def parse_education(text: str, old=None) -> list[dict]:
    old = old or []
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        p = [x.strip() for x in line.split("|")] + ["", "", ""]
        entry = {"degree": p[0], "institution": p[1], "dates": p[2], "details": ""}
        for o in old:
            if o.get("degree") == p[0] and o.get("institution") == p[1]:
                entry["details"] = o.get("details", "")
        out.append(entry)
    return out


def format_lines(items) -> str:
    return "\n".join(items or [])


def parse_lines(text: str) -> list[str]:
    return [l.strip(" \t•-*") for l in text.splitlines() if l.strip(" \t•-*")]


def collect_edits(parsed, fields: dict[str, str]) -> dict:
    """Compare the editor's text boxes with the current parsed values; return only the fields that really changed."""
    cur = {"name": parsed.name, "summary": parsed.summary, "skills": parsed.skills,
           "programming_languages": parsed.programming_languages, "frameworks": parsed.frameworks, "tools": parsed.tools,
           "experience": parsed.experience, "projects": parsed.projects, "education": parsed.education,
           "certifications": parsed.certifications, "achievements": parsed.achievements}
    new = {
        "name": fields.get("name", "").strip(), "summary": fields.get("summary", "").strip(),
        "skills": parse_list(fields.get("skills", "")), "programming_languages": parse_list(fields.get("programming_languages", "")),
        "frameworks": parse_list(fields.get("frameworks", "")), "tools": parse_list(fields.get("tools", "")),
        "experience": parse_experience(fields.get("experience", ""), parsed.experience),
        "projects": parse_projects(fields.get("projects", "")),
        "education": parse_education(fields.get("education", ""), parsed.education),
        "certifications": parse_lines(fields.get("certifications", "")),
        "achievements": parse_lines(fields.get("achievements", "")),
    }
    out = {}
    for k, v in new.items():
        if k not in fields:
            continue
        if k in ("experience", "projects", "education"):
            same = {"experience": format_experience, "projects": format_projects, "education": format_education}[k](cur[k]) \
                   == {"experience": format_experience, "projects": format_projects, "education": format_education}[k](v)
        else:
            same = v == cur[k]
        if not same:
            out[k] = v
    return out
