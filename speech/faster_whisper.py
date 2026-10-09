"""faster-whisper (CTranslate2) engine — fully local."""
import numpy as np

from app.errors import InsufficientMemory, WhisperModelMissing
from speech.stt_manager import STTEngine, is_whisper_downloaded, whisper_dir


class FasterWhisperEngine(STTEngine):
    name = "faster-whisper"

    def __init__(self, model: str, device: str = "cpu", language: str = "en"):
        self.model_name, self.device = model, device
        self.language = None if language in ("", "auto") else language
        self._model = None

    def load(self) -> None:
        if not is_whisper_downloaded(self.model_name):
            raise WhisperModelMissing(f"Whisper model '{self.model_name}' is not downloaded.\n\n"
                                      "Open Setup and download it (one-time).")
        from faster_whisper import WhisperModel
        compute = "float16" if self.device == "cuda" else "int8"
        try:
            self._model = WhisperModel(str(whisper_dir(self.model_name)), device=self.device, compute_type=compute)
        except MemoryError as e:
            raise InsufficientMemory() from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise InsufficientMemory("Not enough GPU memory (VRAM) for this Whisper model.\n\n"
                                         "Pick a smaller model or set Compute Device to CPU.") from e
            if self.device == "cuda":   # missing cuBLAS/cuDNN etc.: fall back to CPU
                self.device = "cpu"
                self._model = WhisperModel(str(whisper_dir(self.model_name)), device="cpu", compute_type="int8")
            else:
                raise

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        segments, _ = self._model.transcribe(
            audio, language=self.language, beam_size=1 if partial else 3, best_of=1,
            temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
            vad_filter=False, no_speech_threshold=0.6)
        return " ".join(s.text.strip() for s in segments).strip()
