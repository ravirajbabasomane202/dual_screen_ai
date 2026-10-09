# PyInstaller spec — run from the project root:  pyinstaller build/DualScreenAI.spec --noconfirm
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("faster_whisper")            # includes the bundled Silero VAD ONNX model
datas += [("../display/assets/icons", "display/assets/icons"),     # offline SVG icons
          ("../display/assets/fonts", "display/assets/fonts")]     # Inter + DejaVu Sans Mono (read via app.paths.resource_dir)
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("soxr")
hidden = [   # imported dynamically by name, so PyInstaller can't see them
    "audio.wasapi_capture", "audio.microphone", "speech.faster_whisper", "speech.whisper_cpp",
    "llm.ollama", "llm.llama_cpp", "pypdf", "docx", "diagnostics.runner", "applog", "app.services", "app.backup", "app.bundle", "app.health", "app.tasks", "app.crashguard", "app.status", "app.gui_thread", "display.pages.resume_page", "display.pages.knowledge_page", "knowledge.watcher", "PySide6.QtSvg", "pyaudiowpatch", "sounddevice", "keyboard", "soxr",
]

a = Analysis(["../main.py"], pathex=[".."], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=["torch", "tkinter", "matplotlib", "scipy", "pandas", "IPython"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DualScreenAI", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DualScreenAI")
