"""Windows WASAPI loopback capture (what you hear from the speakers) via PyAudioWPatch."""
import logging

import numpy as np

from app.errors import AudioStreamFailure, NoAudioDevice, WasapiUnavailable
from audio.common import AudioSource, StreamResampler, to_mono

log = logging.getLogger(__name__)


class WasapiLoopbackCapture(AudioSource):
    name = "system"

    def __init__(self, device_name: str = ""):
        super().__init__()
        self.device_name = device_name
        self._p = self._stream = self._rs = None
        self._channels = 2

    @staticmethod
    def list_devices() -> list[dict]:
        try:
            import pyaudiowpatch as pa
        except ImportError as e:
            raise WasapiUnavailable() from e
        p = pa.PyAudio()
        try:
            return [dict(index=d["index"], name=d["name"], rate=int(d["defaultSampleRate"]),
                         channels=d["maxInputChannels"]) for d in p.get_loopback_device_info_generator()]
        except OSError as e:
            raise WasapiUnavailable() from e
        finally:
            p.terminate()

    def _pick(self, pa, p) -> dict:
        try:
            wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        except OSError as e:
            raise WasapiUnavailable() from e
        loopbacks = list(p.get_loopback_device_info_generator())
        if not loopbacks:
            raise NoAudioDevice()
        if self.device_name:
            for d in loopbacks:
                if self.device_name.lower() in d["name"].lower():
                    return d
        speaker = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speaker.get("isLoopbackDevice"):
            return speaker
        for d in loopbacks:
            if speaker["name"] in d["name"]:
                return d
        return loopbacks[0]

    def start(self) -> None:
        try:
            import pyaudiowpatch as pa
        except ImportError as e:
            raise WasapiUnavailable() from e
        self._p = pa.PyAudio()
        try:
            dev = self._pick(pa, self._p)
            rate, self._channels = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
            self._rs = StreamResampler(rate)
            self._stream = self._p.open(
                format=pa.paInt16, channels=self._channels, rate=rate, input=True,
                input_device_index=dev["index"], frames_per_buffer=int(rate * 0.03),
                stream_callback=self._callback)
            self._pa = pa
            log.info("WASAPI loopback started on %s (%d Hz, %d ch)", dev["name"], rate, self._channels)
        except (OSError, ValueError) as e:
            self.stop()
            raise AudioStreamFailure(f"Could not open the system-audio stream: {e}") from e

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            x = self._rs.process(to_mono(x, self._channels))
            if self.on_audio and len(x):
                self.on_audio(x)
        except Exception:
            log.exception("loopback callback failed")
        return (None, self._pa.paContinue)

    def is_alive(self) -> bool:
        try:
            return self._stream is not None and self._stream.is_active()
        except Exception:
            return False

    def stop(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._p is not None:
                self._p.terminate()
        except Exception:
            pass
        self._stream = self._p = None
