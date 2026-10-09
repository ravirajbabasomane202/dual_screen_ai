"""AudioCapture abstraction: system (WASAPI loopback) | microphone | mixed -> 512-sample 16 kHz frames."""
import logging
import threading
import time
from typing import Callable

import numpy as np

from audio.common import FRAME, FrameChunker, AudioSource

log = logging.getLogger(__name__)


class AudioManager:
    def __init__(self, settings, on_frame: Callable[[np.ndarray], None],
                 on_error: Callable[[Exception], None] | None = None, sources: list[AudioSource] | None = None):
        self.s, self.on_frame, self.on_error = settings, on_frame, on_error
        self._sources = sources
        self._chunkers: dict[str, FrameChunker] = {}
        self._queues: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # source construction ------------------------------------------------
    def _build_sources(self) -> list[AudioSource]:
        out: list[AudioSource] = []
        if self.s.get("audio.system_enabled"):
            from audio.wasapi_capture import WasapiLoopbackCapture
            out.append(WasapiLoopbackCapture(self.s.get("audio.system_device") or ""))
        if self.s.get("audio.mic_enabled"):
            from audio.microphone import MicrophoneCapture
            out.append(MicrophoneCapture(self.s.get("audio.mic_device") or ""))
        return out

    # lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        sources = self._sources if self._sources is not None else self._build_sources()
        self._active = sources
        if not sources:
            from app.errors import NoAudioDevice
            raise NoAudioDevice("No audio source enabled. Enable system audio or the microphone in Settings.")
        mixed = len(sources) > 1
        for src in sources:
            self._chunkers[src.name] = FrameChunker()
            self._queues[src.name] = np.zeros(0, np.float32)
            src.on_audio = (lambda x, n=src.name: self._on_mixed(n, x)) if mixed else self._on_single
            src.start()
        if mixed:
            self._spawn(self._mixer_loop)
        self._spawn(self._watchdog)

    def stop(self) -> None:
        self._stop.set()
        for src in getattr(self, "_active", []):
            src.on_audio = None
            src.stop()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()

    def _spawn(self, fn):
        t = threading.Thread(target=fn, daemon=True, name=fn.__name__)
        t.start()
        self._threads.append(t)

    # single source ------------------------------------------------------
    def _gain(self) -> float:
        return float(self.s.get("audio.volume", 1.0))

    def _on_single(self, x: np.ndarray) -> None:
        src = self._active[0]
        for f in self._chunkers[src.name].push(x * self._gain()):
            self.on_frame(f)

    # mixed: time-driven so a silent loopback (which sends nothing) can't stall the mic
    def _on_mixed(self, name: str, x: np.ndarray) -> None:
        with self._lock:
            q = np.concatenate([self._queues[name], x])
            self._queues[name] = q[-FRAME * 200:]  # cap ~6 s

    def _mixer_loop(self) -> None:
        period, nxt = FRAME / 16000, time.monotonic()
        while not self._stop.is_set():
            nxt += period
            mix = np.zeros(FRAME, np.float32)
            with self._lock:
                for name, q in self._queues.items():
                    take = q[:FRAME]
                    mix[: len(take)] += take
                    self._queues[name] = q[FRAME:]
            self.on_frame(np.clip(mix * self._gain(), -1.0, 1.0))
            time.sleep(max(0.0, nxt - time.monotonic()))

    # reconnect if the default output device changed / stream died
    def _watchdog(self) -> None:
        while not self._stop.wait(2.0):
            for src in self._active:
                if not src.is_alive():
                    log.warning("%s stream died, restarting", src.name)
                    try:
                        src.stop()
                        src.start()
                    except Exception as e:
                        if self.on_error:
                            self.on_error(e)
                        self._stop.wait(5.0)
