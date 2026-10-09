"""Ollama provider — talks to http://localhost:11434 only."""
import json
from threading import Event
from typing import Iterator

import requests

import logging

from app.errors import LLMContextOverflow, LLMModelMissing, LLMNotRunning, LLMTimeout
from llm.base import LLMProvider


log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3:8b", timeout: float = 60,
                 num_ctx: int = 8192):
        self.host, self.model, self.timeout, self.num_ctx = host.rstrip("/"), model, timeout, num_ctx
        self.last_stats = {}
        self._think_ok = True            # flips to False if the model rejects the "think" option

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException as e:
            raise LLMNotRunning() from e

    def warmup(self) -> None:
        try:  # loads the model into VRAM/RAM so the first real answer is fast
            requests.post(f"{self.host}/api/generate", json={"model": self.model, "keep_alive": "30m"}, timeout=120)
        except requests.RequestException as e:
            log.warning("Ollama warm-up failed: %s", e)

    def embed(self, text: str, model: str) -> list[float]:
        r = requests.post(f"{self.host}/api/embed", json={"model": model, "input": text}, timeout=10)
        r.raise_for_status()
        return r.json()["embeddings"][0]

    def stream(self, messages, *, temperature=0.2, max_tokens=300, json_schema=None,
               cancel: Event | None = None) -> Iterator[str]:
        payload = {
            "model": self.model, "messages": messages, "stream": True, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx or 4096},
        }
        if self._think_ok:
            payload["think"] = False
        if json_schema:
            payload["format"] = json_schema
        self.last_stats = {"num_ctx": payload["options"]["num_ctx"]}
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, stream=True, timeout=(3.05, self.timeout))
        except requests.ConnectionError as e:
            raise LLMNotRunning() from e
        except requests.Timeout as e:
            raise LLMTimeout() from e
        if r.status_code >= 400:
            body = _body(r)
            r.close()
            if r.status_code == 404:
                raise LLMModelMissing(f"Model '{self.model}' is not installed.\n\nRun: ollama pull {self.model}")
            if r.status_code == 400 and "think" in body.lower() and self._think_ok:
                self._think_ok = False                      # non-thinking model: retry once without the option
                log.warning("model rejects 'think'; retrying without it (%s)", body[:200])
                yield from self.stream(messages, temperature=temperature, max_tokens=max_tokens,
                                       json_schema=json_schema, cancel=cancel)
                return
            if _is_overflow(body):
                raise LLMContextOverflow(f"Context length exceeded for '{self.model}' (num_ctx={payload['options']['num_ctx']}): {body[:300]}")
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {body[:500]}")
        try:
            for line in r.iter_lines():
                if cancel is not None and cancel.is_set():
                    break
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    err = str(data["error"])
                    if "not found" in err:
                        raise LLMModelMissing(err)
                    if _is_overflow(err):
                        raise LLMContextOverflow(err)
                    raise RuntimeError(f"Ollama error: {err}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    self.last_stats.update(prompt_tokens=data.get("prompt_eval_count"),
                                           completion_tokens=data.get("eval_count"),
                                           finish_reason=data.get("done_reason"),
                                           total_seconds=(data.get("total_duration") or 0) / 1e9,
                                           load_seconds=(data.get("load_duration") or 0) / 1e9)
                    pt = data.get("prompt_eval_count") or 0
                    if pt and pt >= payload["options"]["num_ctx"] - 8:
                        self.last_stats["context_truncated"] = True      # Ollama silently drops the START of the prompt
                        log.warning("prompt filled the whole context window (%s/%s tokens): the system prompt may have been cut off",
                                    pt, payload["options"]["num_ctx"])
                    break
        except requests.Timeout as e:
            raise LLMTimeout() from e
        finally:
            r.close()


def _body(r) -> str:
    try:
        return r.text or ""
    except Exception:
        return ""


def _is_overflow(text: str) -> bool:
    t = text.lower()
    return "context length" in t or "context window" in t or "exceeds the" in t and "context" in t or "too many tokens" in t
