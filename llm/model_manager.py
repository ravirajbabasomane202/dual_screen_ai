"""Provider factory, model resolution, and Ollama model download with progress."""
import json

import requests

from app.errors import LLMNotRunning
from llm.base import LLMProvider


def create_provider(settings, resolved_model: str) -> LLMProvider:
    timeout = float(settings.get("ai.timeout"))
    host = settings.get("ai.host")
    num_ctx = int(settings.get("ai.context_window", 8192) or 0)
    if settings.get("ai.provider") == "llama.cpp":
        from llm.llama_cpp import LlamaCppProvider
        return LlamaCppProvider(host if "8080" in host else "http://localhost:8080", resolved_model, timeout, num_ctx)
    from llm.ollama import OllamaProvider
    return OllamaProvider(host, resolved_model, timeout, num_ctx)


def pull_ollama_model(host: str, model: str, progress_cb=None) -> None:
    """Download a model through the local Ollama daemon (internet needed once)."""
    try:
        r = requests.post(f"{host.rstrip('/')}/api/pull", json={"model": model, "stream": True}, stream=True, timeout=(3, 600))
    except requests.ConnectionError as e:
        raise LLMNotRunning() from e
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        if "error" in d:
            raise RuntimeError(d["error"])
        if progress_cb and d.get("total"):
            progress_cb(int(100 * d.get("completed", 0) / d["total"]), d.get("status", ""))
    if progress_cb:
        progress_cb(100, "success")


def detect_local_server() -> tuple[str, str] | None:
    """Look for a running local LLM server. Returns (provider, host) or None."""
    for provider, host in (("ollama", "http://localhost:11434"), ("llama.cpp", "http://localhost:8080")):
        try:
            requests.get(host, timeout=1.5)
            return provider, host
        except Exception:
            continue
    return None
