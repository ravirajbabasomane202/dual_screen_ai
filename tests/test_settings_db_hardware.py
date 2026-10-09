from database.sqlite import Database
from hardware.detector import GPUInfo, HardwareInfo, recommend
from settings.settings_manager import SettingsManager


def test_defaults_are_private_and_free():
    s = SettingsManager(Database(":memory:"))
    assert s.get("privacy.store_transcript") is False
    assert s.get("privacy.store_questions") is False
    assert s.get("privacy.store_audio") is False
    assert s.get("audio.system_enabled") is True and s.get("audio.mic_enabled") is False
    assert s.get("question.auto_threshold") == 0.85 and s.get("question.confirm_threshold") == 0.60
    assert s.get("question.duplicate_timeout") == 30 and s.get("ai.answer_length") == "Short"


def test_settings_persist_and_notify(tmp_path):
    path = tmp_path / "t.db"
    s = SettingsManager(Database(path))
    seen = []
    s.on_change(seen.append)
    s.set_many({"ai.model": "qwen3:4b", "display.font_size": 22})
    assert seen == [{"ai.model", "display.font_size"}]
    s2 = SettingsManager(Database(path))
    assert s2.get("ai.model") == "qwen3:4b" and s2.get("display.font_size") == 22


def test_history_and_context_storage():
    db = Database(":memory:")
    db.add_question("q1", "a1"); db.add_question("q2", "a2")
    assert db.recent_questions() == [("q1", "a1"), ("q2", "a2")]
    db.clear_questions()
    assert db.recent_questions() == []
    db.save_context([("q", "a")])
    assert db.load_context() == [("q", "a")]
    db.clear_context()
    assert db.load_context() == []


def test_recommendation_matches_spec_example():
    hw = HardwareInfo("Intel i5", 6, 16, [GPUInfo("NVIDIA GeForce RTX 3060", "nvidia", 12.0)], cuda=True)
    r = recommend(hw)
    assert (r.stt_model, r.llm_model, r.backend) == ("small", "qwen3:8b", "CUDA")


def test_recommendation_cpu_tiers():
    assert recommend(HardwareInfo(ram_gb=32, cpu_cores=12)).llm_model == "qwen3:8b"
    assert recommend(HardwareInfo(ram_gb=16, cpu_cores=4)).llm_model == "qwen3:4b"
    assert recommend(HardwareInfo(ram_gb=8)).stt_model == "tiny"
    r = recommend(HardwareInfo(ram_gb=16, gpus=[GPUInfo("AMD Radeon", "amd", 8)], vulkan=True))
    assert r.backend == "Vulkan" and r.compute_device == "cpu"
