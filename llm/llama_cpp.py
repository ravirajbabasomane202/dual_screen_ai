"""llama.cpp provider — uses `llama-server` (OpenAI-compatible endpoint on localhost). No cloud."""
import json
from threading import Event
from typing import Iterator

import requests

from app.errors import LLMContextOverflow, LLMNotRunning, LLMTimeout
from llm.base import LLMProvider


class LlamaCppProvider(LLMProvider):
    name = "llama.cpp"

    def __init__(self, host: str = "http://localhost:8080", model: str = "default", timeout: float = 60,
                 num_ctx: int = 0):
        self.host, self.model, self.timeout, self.num_ctx = host.rstrip("/"), model, timeout, num_ctx
        self.last_stats = {}
        self._server_ctx: int | None = None

    def context_window(self) -> int | None:
        """n_ctx the server was started with (llama-server's `-c`); per-request context cannot be changed."""
        if self._server_ctx is None:
            try:
                props = requests.get(f"{self.host}/props", timeout=2).json()
                self._server_ctx = int(props.get("default_generation_settings", {}).get("n_ctx") or 0) or None
            except Exception:
                return self.num_ctx or None
        return self._server_ctx

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.host}/health", timeout=2).ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/v1/models", timeout=3)
            return [m["id"] for m in r.json().get("data", [])]
        except requests.RequestException as e:
            raise LLMNotRunning("llama-server is not running.\n\nStart it (llama-server -m <model.gguf>) and try again.") from e

    def has_model(self, name: str) -> bool:
        return self.is_available()  # llama-server serves exactly the model it was started with

    def stream(self, messages, *, temperature=0.2, max_tokens=300, json_schema=None,
               cancel: Event | None = None) -> Iterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True, "temperature": temperature,
                   "max_tokens": max_tokens, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        if json_schema:
            payload["response_format"] = {"type": "json_object", "schema": json_schema}
        try:
            r = requests.post(f"{self.host}/v1/chat/completions", json=payload, stream=True,
                              timeout=(3.05, self.timeout))
        except requests.ConnectionError as e:
            raise LLMNotRunning("llama-server is not running.\n\nStart it and try again.") from e
        except requests.Timeout as e:
            raise LLMTimeout() from e
        self.last_stats = {}
        if r.status_code >= 400:
            body = (r.text or "")[:500]
            r.close()
            if "context" in body.lower() and ("exceed" in body.lower() or "too large" in body.lower() or "n_ctx" in body.lower()):
                raise LLMContextOverflow(f"llama-server: context size exceeded: {body}")
            raise RuntimeError(f"llama-server HTTP {r.status_code}: {body}")
        try:
            for line in r.iter_lines():
                if cancel is not None and cancel.is_set():
                    break
                if not line or not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if body == b"[DONE]":
                    break
                obj = json.loads(body)
                if obj.get("usage"):
                    self.last_stats.update(prompt_tokens=obj["usage"].get("prompt_tokens"),
                                           completion_tokens=obj["usage"].get("completion_tokens"))
                choice = (obj.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    self.last_stats["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta", {}).get("content")
                if delta:
                    yield delta
        except requests.Timeout as e:
            raise LLMTimeout() from e
        finally:
            r.close()
