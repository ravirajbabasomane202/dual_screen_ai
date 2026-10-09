"""Minimal valid text-only PDF writer for tests: make_pdf(pages) where pages = list of list-of-lines."""


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    objs: list[bytes] = []                       # index i -> object number i+1
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, lines in enumerate(pages):
        page_no, content_no = 3 + 2 * i, 4 + 2 * i
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
        for ln in lines:
            ops.append(f"({_esc(ln)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", "replace")
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_no} 0 R "
                     f"/Resources << /Font << /F1 {3 + 2 * n} 0 R >> >> >>").encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


SAMPLE_RESUME = """Alex Morgan
SUMMARY
Software engineer with 5+ years of experience building machine learning products.
SKILLS
Languages: Python, SQL
Frameworks: PyTorch, FastAPI
Tools: Docker, AWS, Git
EXPERIENCE
Machine Learning Engineer | Northwind Labs | Jan 2021 - Present
- Built a fraud detection pipeline with XGBoost that reduced chargebacks by 25%
- Deployed models with FastAPI and Docker on AWS
PROJECTS
Smart Shelf Vision - computer vision inventory tracker
- Built with Python, OpenCV and YOLO on edge devices
EDUCATION
B.Tech in Computer Science, State University, 2015 - 2019
"""

SAMPLE_KB_PAGES = [[
    "Python Interview Answers", "",
    "Q: What are Python decorators?",
    "A: A decorator is a function that wraps another function to extend its behaviour without modifying it.",
    "It is applied with the @decorator syntax above the function definition.", "",
    "Q: What is the difference between a list and a tuple?",
    "A: Lists are mutable while tuples are immutable, so tuples can be used as dictionary keys.",
]]
