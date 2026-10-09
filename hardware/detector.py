"""Hardware detection and model recommendation."""
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    vendor: str            # nvidia | amd | intel | other
    vram_gb: float = 0.0


@dataclass
class HardwareInfo:
    cpu_name: str = "Unknown CPU"
    cpu_cores: int = 1
    ram_gb: float = 0.0
    gpus: list[GPUInfo] = field(default_factory=list)
    cuda: bool = False
    vulkan: bool = False

    @property
    def nvidia(self) -> GPUInfo | None:
        n = [g for g in self.gpus if g.vendor == "nvidia"]
        return max(n, key=lambda g: g.vram_gb) if n else None


@dataclass
class Recommendation:
    stt_model: str
    llm_model: str
    backend: str           # CUDA | Vulkan | CPU
    compute_device: str    # cuda | cpu  (for faster-whisper)
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout=6) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except Exception:
        return ""


def _cpu_name() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def _vulkan_present() -> bool:
    try:
        import ctypes
        ctypes.WinDLL("vulkan-1.dll")
        return True
    except Exception:
        return shutil.which("vulkaninfo") is not None


def detect() -> HardwareInfo:
    import psutil
    hw = HardwareInfo(cpu_name=_cpu_name(), cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
                      ram_gb=round(psutil.virtual_memory().total / 2**30, 1))
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in out.strip().splitlines():
        try:
            name, mem = [p.strip() for p in line.rsplit(",", 1)]
            hw.gpus.append(GPUInfo(name, "nvidia", round(float(mem) / 1024, 1)))
        except ValueError:
            pass
    if hw.gpus:
        hw.cuda = True
        try:  # confirm CTranslate2 can actually see CUDA, if it's installed
            import ctranslate2
            hw.cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            pass
    ps = _run(["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"], timeout=10)
    try:
        rows = json.loads(ps) if ps.strip() else []
        for r in rows if isinstance(rows, list) else [rows]:
            n = r.get("Name", "")
            low = n.lower()
            if "nvidia" in low or any(g.name == n for g in hw.gpus):
                continue
            vendor = "amd" if ("amd" in low or "radeon" in low) else "intel" if "intel" in low else "other"
            hw.gpus.append(GPUInfo(n, vendor, round((r.get("AdapterRAM") or 0) / 2**30, 1)))
    except Exception:
        pass
    hw.vulkan = _vulkan_present()
    return hw


def recommend(hw: HardwareInfo) -> Recommendation:
    nv = hw.nvidia
    if hw.cuda and nv:
        v = nv.vram_gb
        if v >= 20:
            return Recommendation("large-v3", "qwen3:14b", "CUDA", "cuda")
        if v >= 10:
            return Recommendation("small", "qwen3:8b", "CUDA", "cuda")
        if v >= 6:
            return Recommendation("base", "qwen3:4b", "CUDA", "cuda")
        return Recommendation("base", "qwen3:1.7b", "CUDA", "cuda", ["Low VRAM: using small models."])
    notes = []
    backend = "CPU"
    if hw.vulkan and any(g.vendor in ("amd", "intel") for g in hw.gpus):
        backend = "Vulkan"
        notes.append("Ollama can use your AMD/Intel GPU for the LLM; speech-to-text runs on CPU "
                     "(faster-whisper has no Vulkan support; whisper.cpp Vulkan builds do).")
    if hw.ram_gb >= 32 and hw.cpu_cores >= 8:
        rec = Recommendation("small", "qwen3:8b", backend, "cpu")
        notes.append("CPU-only 8B models are slow (a few tokens/sec). Consider qwen3:4b for faster answers.")
    elif hw.ram_gb >= 16:
        rec = Recommendation("base", "qwen3:4b", backend, "cpu")
    elif hw.ram_gb >= 8:
        rec = Recommendation("tiny", "qwen3:1.7b", backend, "cpu")
    else:
        rec = Recommendation("tiny", "qwen3:0.6b", backend, "cpu", ["Very low RAM: answers will be basic."])
    rec.notes = notes + rec.notes
    return rec
