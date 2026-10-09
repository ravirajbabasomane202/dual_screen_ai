"""Header health indicators: Mic · System audio · STT · LLM · Resume · Knowledge Base.

Pure function of the engine / settings / StatusBus, so it is testable without any widget. Each item says what to do
when clicked: open a settings page ("page:<name>") or run a quick test ("test:<name>")."""
from __future__ import annotations

from dataclasses import dataclass

from app.status import ERROR, IDLE, OK, RUNNING, WARN

GREEN, AMBER, RED, GREY = OK, WARN, ERROR, IDLE


@dataclass
class HealthItem:
    key: str
    label: str
    state: str          # ok | warn | error | idle
    message: str
    action: str = ""    # "page:Resume" | "test:llm" | ""
    icon: str = ""


def _bus_state(bus, stage, default=IDLE):
    ev = bus.snapshot().get(stage)
    return (ev.state, ev.message) if ev else (default, "")


def compute_health(engine, settings, bus) -> list[HealthItem]:
    items: list[HealthItem] = []
    st, msg = _bus_state(bus, "audio")
    sys_on = bool(settings.get("audio.system_enabled"))
    mic_on = bool(settings.get("audio.mic_enabled"))
    items.append(HealthItem("system_audio", "System audio", (st if st != RUNNING else WARN) if sys_on else IDLE,
                            msg or ("Waiting to start" if sys_on else "Disabled in settings"), "page:Audio", "speaker"))
    items.append(HealthItem("mic", "Microphone", (st if st != RUNNING else WARN) if mic_on else IDLE,
                            msg if mic_on and msg else ("Off (meeting audio only)" if not mic_on else "Waiting to start"),
                            "page:Audio", "mic"))
    st, msg = _bus_state(bus, "stt_model")
    items.append(HealthItem("stt", "Speech-to-text", {RUNNING: WARN}.get(st, st), msg or "Not loaded yet",
                            "page:Speech", "waveform"))
    st, msg = _bus_state(bus, "llm_model")
    items.append(HealthItem("llm", "Language model", {RUNNING: WARN}.get(st, st), msg or "Not checked yet",
                            "test:llm", "brain"))
    res = getattr(engine, "resume", None)
    cur = res.get() if res is not None else None
    items.append(HealthItem("resume", "Resume", OK if cur else IDLE,
                            (f"{cur.label or cur.filename} loaded" if cur else "No resume uploaded"), "page:Resume", "document"))
    kb = getattr(engine, "kb", None)
    if kb is None:
        items.append(HealthItem("kb", "Knowledge base", IDLE, "Unavailable", "page:Knowledge Base", "folder"))
    else:
        s = kb.stats()
        kst, kmsg = _bus_state(bus, "kb")
        if kst == ERROR:
            items.append(HealthItem("kb", "Knowledge base", ERROR, kmsg, "page:Knowledge Base", "folder"))
        elif s["documents"] == 0:
            items.append(HealthItem("kb", "Knowledge base", IDLE, "Nothing indexed yet", "page:Knowledge Base", "folder"))
        else:
            items.append(HealthItem("kb", "Knowledge base", OK, f"{s['documents']} document(s), {s['active_chunks']} chunk(s)",
                                    "page:Knowledge Base", "folder"))
    return items
