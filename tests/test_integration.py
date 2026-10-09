"""End-to-end with REAL components: WAV -> Silero VAD -> faster-whisper -> detector -> Ollama (Qwen3) -> UI.

Skips automatically unless all of these exist:
  * tests/data/question.wav   (16-bit PCM mono/stereo speech of a question, e.g. recorded or TTS-generated)
  * the Whisper model downloaded (done by the first-run Setup wizard)
  * Ollama running with the configured model
Run:  pytest -m integration -s
"""
import time
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration
WAV = Path(__file__).parent / "data" / "question.wav"


def _load_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        ch, rate, raw = f.getnchannels(), f.getframerate(), f.readframes(f.getnframes())
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    from audio.common import StreamResampler, to_mono
    return StreamResampler(rate).process(to_mono(x, ch))


def test_wav_to_streamed_answer_and_ui():
    pytest.importorskip("faster_whisper")
    if not WAV.exists():
        pytest.skip("tests/data/question.wav not provided")
    from database.sqlite import Database
    from settings.settings_manager import SettingsManager
    from speech.stt_manager import is_whisper_downloaded
    s = SettingsManager(Database(":memory:"))
    model = "base" if s.get("speech.model") == "auto" else s.get("speech.model")
    if not is_whisper_downloaded(model):
        pytest.skip(f"Whisper '{model}' not downloaded")
    from llm.model_manager import create_provider
    llm_model = "qwen3:4b" if s.get("ai.model") == "auto" else s.get("ai.model")
    prov = create_provider(s, llm_model)
    if not prov.is_available() or not prov.has_model(llm_model):
        pytest.skip("Ollama not running or model missing")

    from app.engine import MeetingEngine
    from speech.faster_whisper import FasterWhisperEngine
    from vad.silero_vad import SileroVAD
    stt = FasterWhisperEngine(model, "cpu", "en"); stt.load()
    events = []
    eng = MeetingEngine(s, lambda n, **p: events.append((n, p)), stt=stt, vad=SileroVAD(), provider=prov, hardware=object())
    eng._audio = type("A", (), {"start": lambda self: None, "stop": lambda self: None})()
    eng._injected["audio"] = True
    eng.start()
    time.sleep(1)
    audio = np.concatenate([_load_wav(WAV), np.zeros(16000 * 2, np.float32)])   # trailing silence ends the utterance
    for i in range(0, len(audio) - 512, 512):
        eng.feed_frame(audio[i:i + 512])
    deadline = time.time() + 120
    while time.time() < deadline and not any(n == "answer_done" for n, _ in events):
        time.sleep(0.25)
    eng.stop()
    names = [n for n, _ in events]
    assert "transcript" in names, events
    assert "answer_done" in names, [e for e in events if e[0] in ("error", "transcript", "pending")]
    done = dict(events)["answer_done"]
    print("\nQUESTION:", done["question"], "\nANSWER:", done["answer"], f"\nfirst token {done['first_token']:.1f}s, total {done['seconds']:.1f}s")
    assert done["answer"]
