import io

import pytest

from app.errors import ResumeParseError
from database.sqlite import Database
from resume import ResumeManager, ResumeRetriever, parse_resume
from resume.retrieval import build_chunks
from tests.pdfgen import make_pdf

CLASSIC = """JANE DOE
jane.doe@example.com | +1 (555) 123-4567 | linkedin.com/in/janedoe

SUMMARY
Software engineer with 6+ years building machine learning systems.

TECHNICAL SKILLS
Languages: Python, Java, SQL
Frameworks: Django, PyTorch, FastAPI
Tools: Docker, AWS, Git

PROFESSIONAL EXPERIENCE
Senior Software Engineer | Acme Corp | Jan 2020 - Present
- Built a fraud detection system with XGBoost, reducing losses by 30%
- Led a team of 5 engineers
Software Engineer, Beta Labs (Jun 2017 - Dec 2019)
- Developed REST APIs with Flask

PROJECTS
AI Interview Assistant - local LLM desktop app
- Built with Python, PySide6 and Ollama
Tech: Python, Ollama

EDUCATION
B.Tech in Computer Science, IIT Bombay, 2013 - 2017

CERTIFICATIONS
- AWS Certified Solutions Architect
"""

MESSY = """John   Q   Smith
----------------------------
== Profile ==
Data scientist, 3 years of experience.
== Technical Skills: Python, R, TensorFlow, scikit-learn, Tableau, Kafka ==
EXPERIENCE:
Data Scientist at Globex Inc. | 03/2021 – 05/2024
• Trained churn models in TensorFlow improving retention by 12%
Intern - Initech (Jun 2020 - Aug 2020)
* Wrote ETL jobs in SQL
ACADEMIC PROJECTS
Traffic Sign Recognition
- CNN built with Keras and OpenCV
Education: M.Sc. Data Science, University of Leeds, 2019 - 2021
"""


def test_classic_resume_is_fully_structured():
    r = parse_resume(CLASSIC)
    assert r.name == "Jane Doe" and "Python" in r.programming_languages and "Django" in r.frameworks and "Docker" in r.tools
    assert [e["company"] for e in r.experience] == ["Acme Corp", "Beta Labs"]
    assert r.experience[0]["title"] == "Senior Software Engineer" and r.experience[0]["end"].lower() == "present"
    assert r.projects[0]["name"] == "AI Interview Assistant" and "Ollama" in r.projects[0]["technologies"]
    assert r.education[0]["institution"] == "IIT Bombay" and r.certifications == ["AWS Certified Solutions Architect"]
    assert r.years_of_experience >= 8
    assert any("30%" in a for a in r.achievements)


def test_unusual_format_still_parses():
    r = parse_resume(MESSY)
    assert r.name == "John Q Smith"
    assert {"Python", "TensorFlow", "Tableau"} <= set(r.skills)
    assert "Globex Inc." in r.companies and "Data Scientist" in r.job_titles
    assert r.projects and r.projects[0]["name"] == "Traffic Sign Recognition"
    assert r.education and "Leeds" in r.education[0]["institution"]
    assert 3 <= r.years_of_experience <= 3.5


def test_no_contact_details_are_kept_in_parsed_data():
    r = parse_resume(CLASSIC)
    blob = str(r.to_dict()) + " ".join(c.text for c in build_chunks(r, "")) + ResumeRetriever().profile_brief()
    assert "@" not in blob and "555" not in blob and "linkedin" not in blob.lower()


def test_garbage_never_crashes():
    for t in ["", "\n\n", "!!!???", "a" * 5000, "Experience\n\n- - -\n2019-2020", "Skills:\nProjects:\nEducation"]:
        parse_resume(t)


def _docx_bytes(text):
    import docx
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    b = io.BytesIO(); d.save(b); return b.getvalue()


@pytest.fixture
def mgr(tmp_path):
    return ResumeManager(Database(tmp_path / "r.db"))


def test_upload_formats_store_view_persist(tmp_path):
    pdf = make_pdf([CLASSIC.split("\n")])
    for name, data in (("cv.pdf", pdf), ("cv.docx", _docx_bytes(CLASSIC)), ("cv.txt", CLASSIC.encode())):
        db = Database(tmp_path / f"{name}.db")
        m = ResumeManager(db)
        stored = m.upload_bytes(name, data)
        assert stored.parsed.name == "Jane Doe" and len(stored.parsed.experience) == 2, name
        m2 = ResumeManager(Database(tmp_path / f"{name}.db"))             # persistence across launches
        assert m2.has_resume() and m2.get().parsed.companies == stored.parsed.companies
        assert "WORK EXPERIENCE" in m2.view_text() and "Acme Corp" in m2.view_text()


def test_upload_from_path_sets_setting(tmp_path):
    from settings.settings_manager import SettingsManager
    db = Database(":memory:")
    s = SettingsManager(db)
    f = tmp_path / "cv.txt"; f.write_text(CLASSIC)
    ResumeManager(db, s).upload(f)
    assert s.get("resume.path") == str(f)


def test_replace_reparse_delete(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    mgr.upload_bytes("b.txt", MESSY.encode())                              # replace
    assert mgr.get().parsed.name == "John Q Smith" and mgr.get().filename == "b.txt"
    again = mgr.reparse()                                                  # from the stored original bytes
    assert again.parsed.name == "John Q Smith"
    mgr.delete()
    assert not mgr.has_resume() and mgr.summary() == {"loaded": False}
    with pytest.raises(ResumeParseError):
        mgr.reparse()


def test_failed_upload_keeps_previous_profile(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    for name, data in (("bad.pdf", b"not a pdf"), ("empty.txt", b""), ("x.exe", b"MZ" * 50), ("img.pdf", make_pdf([[""]])),
                       ("nothing.txt", b"lorem ipsum dolor sit amet consectetur adipiscing elit sed do")):
        with pytest.raises(ResumeParseError):
            mgr.upload_bytes(name, data)
        assert mgr.get().filename == "a.txt" and mgr.get().parsed.name == "Jane Doe"


def test_listeners_fire(mgr):
    seen = []
    mgr.on_change(lambda s: seen.append(s.filename if s else None))
    mgr.upload_bytes("a.txt", CLASSIC.encode()); mgr.delete()
    assert seen == ["a.txt", None]


def test_resume_retrieval_and_relevance(mgr):
    mgr.upload_bytes("a.txt", CLASSIC.encode())
    rr = ResumeRetriever().attach(mgr)
    top = rr.search("Tell me about your AI project")[0]
    assert top.kind == "project" and "AI Interview Assistant" in top.title
    assert rr.search("what did you do at Acme Corp")[0].kind == "experience"
    assert rr.search("what is the weather on mars") == []
    score, terms = rr.relevance("What did you do at Acme Corp?")
    assert score >= 0.6 and "acme corp" in terms
    assert rr.relevance("Tell me about this project")[0] == 0.4 and rr.relevance("What is 2+2?")[0] == 0.0
    assert "Jane Doe" in rr.profile_brief() and rr.project_names() == ["AI Interview Assistant"]
    mgr.delete()
    assert rr.search("python") == [] and rr.relevance("Acme Corp") == (0.0, [])


def test_resume_answer_is_grounded_in_retrieved_entry(mgr):
    """Question -> resume retrieval -> prompt carries the project -> personalised answer accepted by the validator."""
    import json
    from threading import Event
    from context.context_manager import ContextManager
    from knowledge import Retriever
    from llm.base import LLMProvider
    from llm.service import Extras, LLMService
    from settings.settings_manager import SettingsManager

    mgr.upload_bytes("a.txt", CLASSIC.encode())
    rr = ResumeRetriever().attach(mgr)
    r = Retriever(None, rr).retrieve("Tell me about your AI project")
    sent = {}

    class P(LLMProvider):
        name, model = "p", "p"
        def is_available(self): return True
        def list_models(self): return ["p"]
        def stream(self, messages, **kw):
            sent["m"] = messages
            yield json.dumps({"question": "Tell me about your AI project", "intent": "describe project", "grounding": "resume",
                              "answer": "I built the AI Interview Assistant, a local LLM desktop app using Python, PySide6 and Ollama.",
                              "confidence": 0.9})
    s = SettingsManager(Database(":memory:"))
    res = LLMService(P(), s, ContextManager()).answer_stream("Tell me about your AI project", lambda q: None, lambda d: None,
                                                              Event(), extras=Extras(r.resume, r.kb, r.ids))
    assert "AI Interview Assistant" in sent["m"][-1]["content"] and res.valid and res.grounding == "resume"
