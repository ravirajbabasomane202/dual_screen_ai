"""Shared audio helpers: all sources deliver 16 kHz mono float32."""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

TARGET_RATE = 16000
FRAME = 512  # 32 ms @ 16 kHz, the frame size Silero VAD expects


class AudioSource(ABC):
    name = "source"

    def __init__(self):
        self.on_audio: Callable[[np.ndarray], None] | None = None

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def is_alive(self) -> bool:
        return True


class StreamResampler:
    """Stateful resampler (soxr if available, linear interpolation otherwise)."""

    def __init__(self, in_rate: int, out_rate: int = TARGET_RATE):
        self.in_rate, self.out_rate, self._rs = in_rate, out_rate, None
        if in_rate != out_rate:
            try:
                import soxr
                self._rs = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32")
            except ImportError:
                pass

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.in_rate == self.out_rate or len(x) == 0:
            return x
        if self._rs is not None:
            return self._rs.resample_chunk(x)
        n = int(len(x) * self.out_rate / self.in_rate)
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def to_mono(x: np.ndarray, channels: int) -> np.ndarray:
    if channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32, copy=False)


class FrameChunker:
    """Re-frames arbitrary-length chunks into fixed FRAME-sample frames."""

    def __init__(self, size: int = FRAME):
        self.size, self._buf = size, np.zeros(0, np.float32)

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, x])
        n = len(self._buf) // self.size
        frames = [self._buf[i * self.size:(i + 1) * self.size] for i in range(n)]
        self._buf = self._buf[n * self.size:]
        return frames
