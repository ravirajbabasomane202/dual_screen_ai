import pytest
import requests
from app.errors import LLMModelMissing, LLMNotRunning, WhisperModelMissing
from llm.ollama import OllamaProvider


def test_ollama_not_running_gives_friendly_message():
    p = OllamaProvider("http://127.0.0.1:9", "qwen3:4b")    # nothing listens on port 9
    assert p.is_available() is False
    with pytest.raises(LLMNotRunning) as e:
        list(p.stream([{"role": "user", "content": "hi"}]))
    assert "Ollama is not running" in e.value.user_message


def test_missing_whisper_model_is_reported_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setattr("app.paths.app_dir", lambda: tmp_path)
    from speech.faster_whisper import FasterWhisperEngine
    with pytest.raises(WhisperModelMissing):
        FasterWhisperEngine("tiny", "cpu").load()


def test_hallucination_filter():
    from speech.stt_manager import filter_hallucination
    assert filter_hallucination("Thank you.", 1.0) == ""
    assert filter_hallucination("Thank you.", 5.0) == "Thank you."
    assert filter_hallucination("the the the the the the the", 3.0) == ""
    assert filter_hallucination("What is overfitting?", 1.0) == "What is overfitting?"
