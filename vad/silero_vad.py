"""Silero VAD (ONNX, no PyTorch needed) + a speech segmenter state machine.

States: SILENCE -> SPEECH START -> SPEECH CONTINUING (partials) -> SPEECH END (final)
"""
import glob
import os
from collections import deque
from typing import Callable

import numpy as np

SR = 16000
FRAME = 512
FRAME_S = FRAME / SR


class SileroVAD:
    """Frame-level speech probability using the Silero VAD v5/v6 ONNX model."""

    CONTEXT = 64

    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort
        path = model_path or self._find_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._hc = "h" in {i.name for i in self._sess.get_inputs()}   # faster-whisper's bundled layout
        self.reset()

    @staticmethod
    def _find_model() -> str:
        import importlib.util
        candidates = []
        for pkg, sub in (("faster_whisper", "assets"), ("silero_vad", "data")):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                base = list(spec.submodule_search_locations)[0]
                candidates += glob.glob(os.path.join(base, sub, "silero_vad*.onnx"))
        if not candidates:
            raise FileNotFoundError("Silero VAD ONNX model not found (install faster-whisper).")
        return sorted(candidates)[-1]

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._ctx = np.zeros((1, self.CONTEXT), np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._ctx, frame.reshape(1, -1).astype(np.float32)], axis=1)
        if self._hc:
            out, self._h, self._c = self._sess.run(None, {"input": x, "h": self._h, "c": self._c})
        else:
            out, self._state = self._sess.run(
                None, {"input": x, "state": self._state, "sr": np.array(SR, dtype=np.int64)})
        self._ctx = x[:, -self.CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class SpeechSegmenter:
    """Turns a stream of 32 ms frames into speech_start / partial / final events."""

    def __init__(self, prob_fn: Callable[[np.ndarray], float], on_event: Callable[..., None],
                 sensitivity: float = 0.5, min_silence_ms: int = 600, min_speech_ms: int = 300,
                 preroll_ms: int = 320, partial_every_s: float = 1.5, max_segment_s: float = 25.0,
                 tail_ms: int = 200, reset_fn: Callable[[], None] | None = None):
        self.prob_fn, self.on_event, self.reset_fn = prob_fn, on_event, reset_fn
        self.sensitivity = sensitivity
        self.min_silence = int(min_silence_ms / 1000 / FRAME_S)
        self.min_speech = int(min_speech_ms / 1000 / FRAME_S)
        self.partial_every = int(partial_every_s / FRAME_S)
        self.max_frames = int(max_segment_s / FRAME_S)
        self.tail = int(tail_ms / 1000 / FRAME_S)
        self._preroll: deque = deque(maxlen=max(1, int(preroll_ms / 1000 / FRAME_S)))
        self._reset()

    def _reset(self) -> None:
        self.in_speech, self._buf, self._silence, self._since_partial, self._pre_len = False, [], 0, 0, 0

    @property
    def threshold(self) -> float:
        return min(0.9, max(0.2, 0.8 - 0.6 * self.sensitivity))

    def feed(self, frame: np.ndarray) -> None:
        p = self.prob_fn(frame)
        if not self.in_speech:
            self._preroll.append(frame)
            if p >= self.threshold:
                self.in_speech, self._buf = True, list(self._preroll)
                self._pre_len = len(self._buf) - 1
                self._silence = self._since_partial = 0
                self.on_event("speech_start")
            return
        self._buf.append(frame)
        self._silence = self._silence + 1 if p < max(0.1, self.threshold - 0.15) else 0
        self._since_partial += 1
        if self._silence >= self.min_silence or len(self._buf) >= self.max_frames:
            self._finish()
        elif self._since_partial >= self.partial_every:
            self._since_partial = 0
            self.on_event("partial", audio=np.concatenate(self._buf))

    def _finish(self) -> None:
        speech_frames = len(self._buf) - self._pre_len - self._silence
        if speech_frames >= self.min_speech:
            keep = len(self._buf) - max(0, self._silence - self.tail)
            self.on_event("final", audio=np.concatenate(self._buf[:keep]))
        else:
            self.on_event("discard")
        self._preroll.clear()
        self._reset()
        if self.reset_fn:
            self.reset_fn()
