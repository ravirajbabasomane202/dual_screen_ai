"""Application services that sit next to the engine: resume, knowledge base, retrieval."""
from __future__ import annotations

from context.interview_profile import InterviewProfileStore
from database.repositories import ContextRepository
from knowledge import KnowledgeIndex, Retriever
from knowledge.hybrid import make_embedder
from knowledge.watcher import FolderWatcher
from question.feedback import FeedbackStore
from resume import ResumeManager, ResumeRetriever


class Services:
    def __init__(self, db, settings, context):
        self.db, self.settings, self.context = db, settings, context
        self.resume = ResumeManager(db, settings)
        self.resume_retriever = ResumeRetriever()
        self.kb = KnowledgeIndex(db, settings=settings)
        self.retriever = Retriever(self.kb, self.resume_retriever, settings, context)
        self.context_repo = ContextRepository(db)
        self.feedback = FeedbackStore(db)
        self.profile_store = InterviewProfileStore(db)
        self.watcher = FolderWatcher(self.kb, self.kb_folders, interval=float(settings.get("knowledge.watch_interval", 30)))
        self.resume_retriever.attach(self.resume)      # must update before the profile hook below reads it
        self.resume.on_change(self._on_resume, thread="inline")      # data-model hook: no widgets here
        self._on_resume(self.resume.get())

    def kb_folders(self) -> list[str]:
        """The primary knowledge-base folder plus any extra folders the user added."""
        primary = str(self.settings.get("knowledge.folder", "") or "")
        extra = [str(f) for f in (self.settings.get("knowledge.extra_folders", []) or [])]
        return [f for f in dict.fromkeys([primary] + extra) if f]

    def _on_resume(self, stored) -> None:
        rr = self.resume_retriever
        if stored:
            self.context.persistent.set_profile(rr.candidate_profile(), rr.profile_brief(), rr.project_names())
        else:
            self.context.persistent.set_profile({}, "", [])

    def use_provider(self, provider) -> None:
        """Pick embeddings once the LLM provider exists (Ollama embedding model if configured, else offline hashing)."""
        emb = make_embedder(self.settings, provider)
        if emb.signature != self.kb.embedder.signature:
            self.kb.embedder = emb
            self.kb._searcher = None

    # interview memory persistence (only used when the user opted in to storing history) -----------------------
    def save_memory(self) -> None:
        self.context_repo.save(self.context.persistent.to_dict())

    def load_memory(self) -> None:
        self.context.persistent.load_dict(self.context_repo.load())
