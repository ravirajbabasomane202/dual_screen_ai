"""whisper.cpp engine via pywhispercpp (drop-in replacement for faster-whisper).

NOTE: the stock pywhispercpp wheel is CPU-only; build it with GGML_VULKAN/CUDA for GPU.
"""
import numpy as np

from speech.stt_manager import STTEngine


class WhisperCppEngine(STTEngine):
    name = "whisper.cpp"

    def __init__(self, model: str, language: str = "en"):
        self.model_name = model
        self.language = "auto" if language in ("", "auto") else language
        self._model = None

    def load(self) -> None:
        from app.paths import models_dir
        try:
            from pywhispercpp.model import Model
        except ImportError as e:
            raise RuntimeError("whisper.cpp engine needs: pip install pywhispercpp") from e
        self._model = Model(self.model_name, models_dir=str(models_dir() / "ggml"),
                            print_realtime=False, print_progress=False, language=self.language)

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        segs = self._model.transcribe(audio.astype(np.float32))
        return " ".join(s.text.strip() for s in segs).strip()
