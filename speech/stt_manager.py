"""STT engine interface, factory, model download, and hallucination filtering."""
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
# Whisper often invents these on near-silence / music.
_HALLUCINATIONS = {
    "you", "thank you", "thanks for watching", "thank you for watching", "bye", "okay", "uh", "um",
    "subtitles by the amara.org community", "please subscribe",
}


class STTEngine(ABC):
    name = "stt"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        """audio: float32 mono 16 kHz. Returns text ('' if nothing)."""


def filter_hallucination(text: str, audio_seconds: float) -> str:
    t = re.sub(r"[^\w\s']", "", text.lower()).strip()
    if t in _HALLUCINATIONS and audio_seconds < 2.5:
        return ""
    words = t.split()
    if len(words) >= 6 and len(set(words)) <= 2:   # "the the the the..." loops
        return ""
    return text.strip()


def whisper_dir(model: str) -> Path:
    from app.paths import models_dir
    return models_dir() / "whisper" / model


def is_whisper_downloaded(model: str) -> bool:
    d = whisper_dir(model)
    return (d / "model.bin").exists() and (d / "config.json").exists()


def download_whisper_model(model: str, progress_cb=None) -> Path:
    """Download the CTranslate2 Whisper model once; afterwards everything is offline."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class _Bar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)      # newer huggingface_hub passes name=...; plain tqdm rejects it
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            r = super().update(n)
            if progress_cb and self.total:
                progress_cb(max(0, min(100, int(100 * self.n / self.total))))
            return r

    dest = whisper_dir(model)
    repo = f"Systran/faster-whisper-{model}"
    try:
        snapshot_download(repo, local_dir=str(dest), tqdm_class=_Bar)
    except (TypeError, ValueError):       # progress bar API mismatch with this huggingface_hub version: download without it
        snapshot_download(repo, local_dir=str(dest))
    if not is_whisper_downloaded(model):
        raise RuntimeError("Whisper download incomplete.")
    return dest


def create_engine(settings, resolved_model: str, resolved_device: str) -> STTEngine:
    lang = settings.get("speech.language")
    if settings.get("speech.engine") == "whisper.cpp":
        from speech.whisper_cpp import WhisperCppEngine
        return WhisperCppEngine(resolved_model, lang)
    from speech.faster_whisper import FasterWhisperEngine
    return FasterWhisperEngine(resolved_model, resolved_device, lang)
