"""LLM provider interface (Ollama today, llama.cpp server today, anything local tomorrow)."""
from abc import ABC, abstractmethod
from threading import Event
from typing import Iterator


class LLMProvider(ABC):
    name = "llm"
    model = ""
    model_path = ""                 # for providers that load a local file
    num_ctx = 0                     # context window requested from the runtime (0 = runtime default)
    last_stats: dict = {}           # filled by stream(): prompt_tokens, completion_tokens, finish_reason, ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...

    @abstractmethod
    def stream(self, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 300,
               json_schema: dict | None = None, cancel: Event | None = None) -> Iterator[str]:
        """Yield text chunks. Raise LLMNotRunning / LLMModelMissing / LLMTimeout."""

    def generate(self, messages: list[dict], **kw) -> str:
        return "".join(self.stream(messages, **kw))

    def warmup(self) -> None:
        pass

    def context_window(self) -> int | None:
        """Actual context size of the loaded model if the runtime reports it."""
        return self.num_ctx or None

    def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError

    def has_model(self, name: str) -> bool:
        have = self.list_models()
        return name in have or f"{name}:latest" in have
