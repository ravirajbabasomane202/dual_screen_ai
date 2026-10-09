import numpy as np
import pytest
from audio.audio_manager import AudioManager
from audio.common import FRAME, AudioSource, FrameChunker, StreamResampler, to_mono
from vad.silero_vad import SpeechSegmenter


def test_to_mono_and_chunker():
    stereo = np.array([1, 3, 2, 4], np.float32)
    assert to_mono(stereo, 2).tolist() == [2.0, 3.0]
    c = FrameChunker()
    frames = c.push(np.zeros(FRAME * 2 + 100, np.float32)) + c.push(np.zeros(FRAME - 100, np.float32))
    assert len(frames) == 3 and all(len(f) == FRAME for f in frames)


def test_resampler_48k_to_16k_keeps_tone():
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rs = StreamResampler(48000)
    y = np.concatenate([rs.process(x[i:i + 1440]) for i in range(0, len(x), 1440)])
    assert 15000 < len(y) <= 16000
    assert abs(np.sqrt(np.mean(y[1000:] ** 2)) - 0.707) < 0.05


class FakeSource(AudioSource):
    name = "fake"
    def start(self): self.started = True
    def stop(self): self.started = False


def test_audio_manager_single_source_frames():
    class S:  # minimal settings stub
        def get(self, k, d=None): return {"audio.volume": 1.0}.get(k, d)
    got, src = [], FakeSource()
    m = AudioManager(S(), got.append, sources=[src])
    m.start()
    src.on_audio(np.ones(FRAME * 2 + 10, np.float32))
    m.stop()
    assert len(got) == 2 and src.started is False


def _segmenter(probs, events):
    it = iter(probs)
    return SpeechSegmenter(lambda f: next(it), lambda k, audio=None: events.append((k, None if audio is None else len(audio))),
                           sensitivity=0.5, min_silence_ms=320, min_speech_ms=160, partial_every_s=0.32)


def test_segmenter_state_machine_start_partial_final():
    frame = np.zeros(FRAME, np.float32)
    probs = [0.0] * 3 + [0.9] * 30 + [0.0] * 12
    ev = []
    seg = _segmenter(probs, ev)
    for _ in probs:
        seg.feed(frame)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "speech_start" and "partial" in kinds and kinds[-1] == "final"
    assert kinds.count("final") == 1


def test_segmenter_discards_blips_and_never_emits_for_silence():
    frame = np.zeros(FRAME, np.float32)
    ev = []
    seg = _segmenter([0.0] * 20 + [0.9] + [0.0] * 20, ev)
    for _ in range(41):
        seg.feed(frame)
    assert [k for k, _ in ev] == ["speech_start", "discard"]


def test_segmenter_force_splits_long_speech():
    frame = np.zeros(FRAME, np.float32)
    ev = []
    seg = SpeechSegmenter(lambda f: 0.9, lambda k, audio=None: ev.append(k), max_segment_s=1.0, partial_every_s=10)
    for _ in range(100):
        seg.feed(frame)
    assert "final" in ev


def test_silero_onnx_runs():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_whisper")
    from vad.silero_vad import SileroVAD
    v = SileroVAD()
    silence = [v.prob(np.zeros(FRAME, np.float32)) for _ in range(10)]
    noise = np.random.default_rng(0).normal(0, 0.01, FRAME).astype(np.float32)
    assert max(silence) < 0.2 and 0.0 <= v.prob(noise) <= 1.0
