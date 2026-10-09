"""Optional microphone capture via sounddevice."""
import logging

import numpy as np

from app.errors import AudioStreamFailure, NoAudioDevice
from audio.common import AudioSource, StreamResampler, to_mono

log = logging.getLogger(__name__)


class MicrophoneCapture(AudioSource):
    name = "microphone"

    def __init__(self, device_name: str = ""):
        super().__init__()
        self.device_name = device_name
        self._stream = self._rs = None

    @staticmethod
    def list_devices() -> list[dict]:
        import sounddevice as sd
        return [dict(index=i, name=d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

    def start(self) -> None:
        import sounddevice as sd
        idx = None
        if self.device_name:
            idx = next((d["index"] for d in self.list_devices() if self.device_name.lower() in d["name"].lower()), None)
        try:
            info = sd.query_devices(idx, "input")
        except Exception as e:
            raise NoAudioDevice("No microphone found.") from e
        rate, ch = int(info["default_samplerate"]), min(2, int(info["max_input_channels"]))
        self._rs = StreamResampler(rate)

        def cb(indata, frames, t, status):
            x = self._rs.process(to_mono(indata.reshape(-1).copy(), ch))
            if self.on_audio and len(x):
                self.on_audio(x)

        try:
            self._stream = sd.InputStream(device=idx, samplerate=rate, channels=ch, dtype="float32",
                                          blocksize=int(rate * 0.03), callback=cb)
            self._stream.start()
        except Exception as e:
            raise AudioStreamFailure(f"Could not open the microphone: {e}") from e

    def is_alive(self) -> bool:
        return self._stream is not None and self._stream.active

    def stop(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
