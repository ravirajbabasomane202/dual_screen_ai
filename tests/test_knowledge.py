import time

import pytest

from database.sqlite import Database
from knowledge import KnowledgeIndex, Retriever
from knowledge import index as kb_index_mod
from knowledge.chunker import chunk_document
from knowledge.hybrid import HashingEmbedder, HybridSearcher, SearchItem, tokenize
from knowledge.pdf_parser import parse_document
from knowledge.structure import structure_text
from tests.pdfgen import make_pdf

PY_QA = [["Python Interview Answers", "",
          "Q: What are Python decorators?",
          "A: A decorator is a function that wraps another function to extend its behaviour without modifying it.",
          "Apply it with the @decorator syntax above the function definition.", "",
          "Q: What is the difference between a list and a tuple?",
          "A: Lists are mutable while tuples are immutable, so tuples can be dictionary keys."]]
TECH = [["Technical Questions", "", "Kubernetes", "",
         "Pods are the smallest deployable units in Kubernetes and hold one or more containers.",
         "A Deployment manages replica sets and rolling updates of pods.", "",
         "Key Concepts", "- Service: stable network endpoint for pods", "- Ingress: HTTP routing into the cluster", "",
         "Language      Typing      Speed", "Python        dynamic     medium", "C++           static      fast"],
        ["Memory Management", "", "C++ uses RAII so resources are released when objects go out of scope.",
         "FastAPI is an async Python web framework built on Starlette and Pydantic."]]
TIPS = [["Interview Tips", "", "Always structure answers with the STAR method: Situation, Task, Action and Result.",
         "Practise explaining trade-offs when you describe a design decision."]]
MULTI = [[f"ACME Handbook Confidential", f"Chapter {i}", "", f"Section text for chapter {i} discusses zebra{i} migration patterns in detail.", f"{i}"]
         for i in range(1, 5)]


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "kb"; f.mkdir()
    (f / "python_answers.pdf").write_bytes(make_pdf(PY_QA))
    (f / "technical_questions.pdf").write_bytes(make_pdf(TECH))
    (f / "interview.pdf").write_bytes(make_pdf(TIPS))
    (f / "handbook.pdf").write_bytes(make_pdf(MULTI))
    (f / "notes.txt").write_text("Redis is an in-memory data store used for caching and queues.\n")
    return f


def test_structure_detects_headings_lists_tables_qa():
    blocks = structure_text("\n".join(TECH[0] + PY_QA[0]))
    kinds = {b.kind for b in blocks}
    assert {"heading", "paragraph", "list", "table", "qa"} <= kinds
    qa = [b for b in blocks if b.kind == "qa"][0]
    assert qa.question == "What are Python decorators?" and "wraps another function" in qa.text


def test_pdf_parse_and_chunk_metadata(folder):
    doc = parse_document(folder / "python_answers.pdf")
    chunks = chunk_document(doc)
    qa = [c for c in chunks if c.kind == "qa"]
    assert doc.status == "ok" and len(qa) == 2 and qa[0].page == 1 and qa[0].chunk_id and qa[0].document == "python_answers.pdf"
    assert "Decorators" in qa[0].text or "decorator" in qa[0].text.lower() and qa[0].question.startswith("What are")
    assert qa[0].section == "Python Interview Answers"
    assert "wraps another function" in qa[0].text                       # question and answer stay together


def test_repeated_headers_and_page_numbers_removed(folder):
    doc = parse_document(folder / "handbook.pdf")
    text = " ".join(b.text for p in doc.pages for b in p.blocks)
    assert "Confidential" not in text and "zebra3" in text


def test_sync_search_relevance(folder):
    idx = KnowledgeIndex(Database(":memory:"))
    rep = idx.sync_folder(folder)
    assert rep.added == 5 and not rep.failed and rep.total_chunks >= 6
    top = lambda q: idx.search(q)[0]
    assert top("what are python decorators").document == "python_answers.pdf" and top("what are python decorators").kind == "qa"
    assert top("explain kubernetes pods").document == "technical_questions.pdf"
    assert "RAII" in top("C++ RAII memory").text
    assert "FastAPI" in top("what is FastAPI").text
    assert top("how to use the STAR method").document == "interview.pdf"
    assert top("what is redis caching").document == "notes.txt"
    assert idx.search("completely unrelated quantum gastronomy of penguins") == []
    hit = top("python decorators")
    assert hit.label.startswith("python_answers.pdf p.1") and 0 < hit.score <= 1


def test_no_reindexing_when_nothing_changed(folder, monkeypatch):
    idx = KnowledgeIndex(Database(":memory:"))
    idx.sync_folder(folder)
    calls = {"parse": 0, "embed": 0}
    real_parse, real_embed = kb_index_mod.pdf_parser.parse_document, idx.embedder.embed
    monkeypatch.setattr(kb_index_mod.pdf_parser, "parse_document", lambda p: (calls.__setitem__("parse", calls["parse"] + 1), real_parse(p))[1])
    monkeypatch.setattr(idx.embedder, "embed", lambda t: (calls.__setitem__("embed", calls["embed"] + 1), real_embed(t))[1])
    rep = idx.sync_folder(folder)
    assert rep.unchanged == 5 and rep.added == rep.updated == 0 and calls == {"parse": 0, "embed": 0}


def test_only_changed_document_is_reindexed_and_removed_ones_purged(folder, monkeypatch):
    idx = KnowledgeIndex(Database(":memory:"))
    idx.sync_folder(folder)
    (folder / "interview.pdf").write_bytes(make_pdf([["Interview Tips", "", "Bring a notebook and ask about team culture, tooling and deployment practices."]]))
    (folder / "notes.txt").unlink()
    parsed = []
    real = kb_index_mod.pdf_parser.parse_document
    monkeypatch.setattr(kb_index_mod.pdf_parser, "parse_document", lambda p: (parsed.append(p.name), real(p))[1])
    rep = idx.sync_folder(folder)
    assert parsed == ["interview.pdf"] and rep.updated == 1 and rep.removed == 1
    assert idx.search("what is redis caching") == [] or idx.search("what is redis caching")[0].document != "notes.txt"
    assert "culture" in idx.search("team culture and tooling")[0].text


def test_bad_files_do_not_abort_the_folder(folder):
    (folder / "corrupt.pdf").write_bytes(b"%PDF-1.4 garbage")
    (folder / "scanned.pdf").write_bytes(make_pdf([[""]]))
    idx = KnowledgeIndex(Database(":memory:"))
    rep = idx.sync_folder(folder)
    assert rep.added >= 5 and [n for n, _ in rep.failed] == ["corrupt.pdf"] and "scanned.pdf" in rep.no_text
    again = idx.sync_folder(folder)                              # failures are remembered, not retried each time
    assert again.added == again.updated == 0
    assert idx.search("python decorators")


def test_encrypted_pdf_is_reported(folder):
    pypdf = pytest.importorskip("pypdf")
    import io
    w = pypdf.PdfWriter()
    r = pypdf.PdfReader(io.BytesIO(make_pdf(TIPS)))
    w.add_page(r.pages[0]); w.encrypt("secret")
    with open(folder / "locked.pdf", "wb") as f:
        w.write(f)
    rep = KnowledgeIndex(Database(":memory:")).sync_folder(folder)
    assert any(n == "locked.pdf" and "password" in why for n, why in rep.failed)


def test_index_persists_across_instances_without_reparsing(folder, tmp_path, monkeypatch):
    path = tmp_path / "kb.db"
    KnowledgeIndex(Database(path)).sync_folder(folder)
    idx2 = KnowledgeIndex(Database(path))
    monkeypatch.setattr(kb_index_mod.pdf_parser, "parse_document", lambda p: pytest.fail("re-parsed"))
    assert idx2.sync_folder(folder).unchanged == 5
    assert idx2.search("python decorators")[0].document == "python_answers.pdf"
    assert idx2.stats()["documents"] == 5 and idx2.stats()["chunks"] >= 6
    idx2.clear()
    assert idx2.search("python decorators") == [] and idx2.stats()["chunks"] == 0


def test_force_reindex_and_embedder_change_reuses_text(folder, monkeypatch):
    idx = KnowledgeIndex(Database(":memory:"))
    idx.sync_folder(folder)
    assert idx.sync_folder(folder, force=True).updated == 5

    class Other(HashingEmbedder):
        def __init__(self): super().__init__(256); self.signature = "other-256"
    idx.embedder = Other(); idx._searcher = None
    monkeypatch.setattr(kb_index_mod.pdf_parser, "parse_document", lambda p: pytest.fail("re-parsed"))
    idx.sync_folder(folder)                                      # only the vectors are rebuilt
    assert {c.embedder for c in idx.repo.all_chunks()} == {"other-256"} and idx.search("python decorators")


def test_embedding_failure_falls_back_to_hashing(folder):
    from knowledge.hybrid import EmbeddingError

    class Broken:
        signature, dim = "ollama:x", 0
        def embed(self, t): raise EmbeddingError("down")
    idx = KnowledgeIndex(Database(":memory:"), embedder=Broken())
    rep = idx.sync_folder(folder)
    assert not rep.failed and rep.warnings and idx.search("python decorators")


def test_hybrid_exact_terms_and_speed():
    assert tokenize("C++ and C# with Node.js CI/CD") == ["cpp", "csharp", "nodejs", "cicd"]
    items = [SearchItem(f"c{i}", f"chunk {i} " + " ".join(f"word{(i * 7 + j) % 400}" for j in range(60))) for i in range(2000)]
    items.append(SearchItem("needle", "The GraphQL federation gateway merges subgraph schemas."))
    s = HybridSearcher().build(items)
    t = time.perf_counter()
    hits = s.search("graphql federation", 4)
    assert hits[0].id == "needle" and (time.perf_counter() - t) < 0.05


def test_retriever_combines_resume_and_kb_and_honours_settings(folder):
    from settings.settings_manager import SettingsManager

    class FakeResume:
        def search(self, q, k, min_score):
            class C: id, label, text, score = "proj0", "Project: X", "x", 0.9
            return [C()]
    db = Database(":memory:")
    idx = KnowledgeIndex(db); idx.sync_folder(folder)
    s = SettingsManager(db)
    r = Retriever(idx, FakeResume(), s)
    out = r.retrieve("what are python decorators")
    assert out.resume and out.kb and out.ids[0] == "proj0" and not out.empty and out.seconds < 0.05
    s.set("retrieval.use_resume", False); s.set("retrieval.top_k", 1)
    out = r.retrieve("what are python decorators")
    assert not out.resume and len(out.kb) == 1
    s.set("retrieval.use_knowledge_base", False)
    assert Retriever(idx, None, s).retrieve("python decorators").empty
    assert Retriever().retrieve("anything").empty


def test_rag_answer_uses_retrieved_chunk(folder):
    """PDF folder -> index -> search -> retrieved chunk in the prompt -> grounded answer accepted."""
    import json
    from threading import Event
    from context.context_manager import ContextManager
    from database.sqlite import Database as DB
    from llm.base import LLMProvider
    from llm.service import Extras, LLMService
    from settings.settings_manager import SettingsManager
    db = DB(":memory:")
    idx = KnowledgeIndex(db); idx.sync_folder(folder)
    r = Retriever(idx, None).retrieve("What are Python decorators?")
    seen = {}

    class P(LLMProvider):
        name, model = "p", "p"
        def is_available(self): return True
        def list_models(self): return ["p"]
        def stream(self, messages, **kw):
            seen["prompt"] = messages[-1]["content"]
            yield json.dumps({"question": "What are Python decorators?", "intent": "definition", "grounding": "knowledge_base",
                              "answer": "A decorator is a function that wraps another function to extend its behaviour.", "confidence": 0.9})
    res = LLMService(P(), SettingsManager(db), ContextManager()).answer_stream(
        "What are Python decorators?", lambda q: None, lambda d: None, Event(), extras=Extras(r.resume, r.kb, r.ids))
    assert "REFERENCE NOTES" in seen["prompt"] and "wraps another function" in seen["prompt"] and res.valid
