# Dual-Screen Interview AI Assistant

100% local, zero-API-cost assistant. Your meeting stays on **Monitor 1**; this app listens to the
system audio and shows the detected question, a cleaned-up version of it, and a streaming answer on **Monitor 2**.

```
WASAPI loopback → Silero VAD → faster-whisper → clean → QuestionClassifier → context + resume + PDF retrieval → local LLM → answer validator → Monitor 2
```

No audio, text or telemetry leaves the PC. The only network traffic is the one-time model download
(Whisper from Hugging Face, Qwen3 through Ollama) and optional updates. Everything else talks to `localhost`.

## Quick start (Windows 10/11)
1. Install [Python 3.10+](https://python.org) and [Ollama](https://ollama.com); make sure Ollama is running.
2. `pip install -r requirements.txt`
3. `python main.py` → the **first-run wizard** (progress dots; every step can be skipped): Welcome (hardware) → Audio (with a capture
   test) → Speech model → Language model (finds Ollama / llama.cpp, one-click test + download) → Resume (optional, drag-and-drop) →
   Knowledge base (optional) → Finish (pick Monitor 2).
4. Start your meeting on Monitor 1. Questions appear on Monitor 2.

NVIDIA GPU (optional, faster): `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*`.
AMD/Intel GPUs: Ollama uses them for the LLM automatically where supported; speech-to-text runs on CPU
(or switch Settings → Speech → Engine to `whisper.cpp` with a Vulkan build of `pywhispercpp`).

## The interface
* **Header**: health icons (system audio · microphone · speech-to-text · language model · resume · knowledge base — click one to open its
  settings page or run a quick test), pause, *Answer now*, practice mode, activity log (**Ctrl+L**), debug panel, mini mode,
  click-through lock, command palette (**Ctrl+K**), settings. **F1** lists every shortcut.
* **Pipeline strip**: Audio → VAD → STT → Detect → Context → Retrieve → LLM → Validate → Display. Colour *and* icon show the state;
  hover for the last message and timing, click to filter the activity log to that stage.
* **Answer card**: grounding badge (Resume · Knowledge base · Conversation · General · Unverified), clickable source chips, Copy ·
  Shorter · Longer · Regenerate · Pin · Why this answer? · 👍/👎, answer-style presets (Concise · STAR · Technical deep-dive ·
  Spoken-natural). Type a question in the box under the card to re-ask something the speech model missed.
* **Errors** never block: a toast plus an inline banner with the cause, what to do and a *Details* expander (the stack trace goes
  to `errors.log`). Settings → Debug & Logs → *Export diagnostics bundle* makes a redacted zip for bug reports.
* Themes: Dark, Light, High-contrast, optional accent colour, *Reduce motion* (Settings → Display & Theme).

## Knowledge base manager (Settings → Knowledge Base)
Add files (PDF / DOCX / TXT / MD, or drop them), add folders (checked at start, optional live watching), write **notes** and
**Q&A pairs** (Q&A pairs get the highest retrieval priority). The table shows type, pages, chunks, size, status, last indexed and
source; select rows for *Preview · Re-index · Rename · Disable / Enable · Open location · Delete* (delete has an **Undo** for 10
minutes). The *Chunks* tab turns single chunks off, the *Search playground* shows what a question would retrieve with fused / BM25 /
vector scores and sources. Long jobs show progress and a real **Cancel**; finished documents are kept.
Resume page: structured view, *Edit parsed fields* (kept across re-parse), parser warnings, version history, several named profiles.
Your data: Settings → Privacy → export / import everything as one zip, *Wipe all data*.

## How a question is handled
`question/classifier.py` labels every transcript segment `QUESTION`, `INTERVIEW_PROMPT`, `STATEMENT`, `SMALL_TALK` or `NOISE`
with a confidence and a reason, using punctuation, wh/aux structure, interview-prompt patterns ("Tell us about yourself",
"Walk me through your resume", "I'd like to know more about…", "Let's start with…"), resume relevance and conversation context
(follow-ups such as "And the biggest challenge?"). The LLM is consulted only for the ambiguous middle band.

| Confidence | Action |
|---|---|
| ≥ 85 % | answered automatically |
| 60–84 % | local LLM does a quick semantic check; still < 85 % → "Possible question" bar, press **Answer** / `Ctrl+Shift+A` |
| `question.min_confidence` (30 %) – 59 % | semantic check only; ignored if the combined score stays < 60 % |
| below | ignored, LLM never called |

Small talk / logistics ("Can you hear me?", "Thank you everyone") is never answered. Duplicates are filtered (30 s).

## Answering (strict contract)
The model must return `{"question","intent","grounding","answer","confidence"}` and write the answer the **candidate says aloud**.
`llm/validation.py` rejects answers that describe or restate the question ("The speaker is asking…", "They want to know…",
"This question is about…") — the first words are checked *while streaming*, so a bad answer is never shown — and the request
is regenerated with a repair note (`ai.max_retries`). `grounding` says whether the answer came from your **resume**, the
**knowledge base**, the **conversation**, **general** knowledge, or is **unknown**; the footer shows it.
Resume/knowledge answers are instructed never to invent employers, projects, numbers or technologies.

## Resume (Settings → Resume)
Upload PDF / DOCX / TXT → parsed into name, summary, skills (languages / frameworks / tools), experience, companies,
titles, projects, education, certifications, achievements, years of experience (any layout; contact details are not kept).
Stored in SQLite (original file kept for **Re-parse**). **Replace** only swaps the profile after the new file parsed
successfully; **Delete** removes it. Relevant entries are retrieved per question and sent to the LLM.

## PDF knowledge base (Settings → Knowledge Base)
Pick a folder of PDFs (also .txt/.md/.docx). Text is extracted page by page, structured (headings, lists, tables, Q&A pairs),
chunked (≈220 tokens, Q&A kept together) and indexed **once** (SHA-256 per file; unchanged files are never re-parsed or
re-embedded, changed files are re-indexed alone, deleted files are purged). Retrieval is hybrid — BM25 keywords + dense
vectors fused with Reciprocal Rank Fusion — fully offline. Vectors default to a built-in hashing embedder; set
Settings → Question Detection → *embedding model* (e.g. `nomic-embed-text`) to use Ollama embeddings instead.
Only the top `retrieval.top_k` chunks are sent to the model.

## Memory
Short-term turns + persistent interview context (candidate profile, current project/topic, technologies mentioned, facts the
interviewer attributes to you, constraints). Prompts are **token-budgeted**: the current question is never truncated; old
irrelevant turns go first, then low-ranked chunks. Long multi-clause questions are kept whole.

## Debugging
* **Settings → Debug → Enable debug mode** (or `debug.enabled`): verbose logs that include question/answer text, plus a trace
  panel on the AI window showing transcript → clean → classification → resume/PDF hits → context → prompt → raw response →
  validation → final answer. With debug off, logs contain only lengths/hashes/metadata (no transcripts, answers, e-mail, phone).
* **Logs** live in `%APPDATA%\DualScreenAI\logs\` (portable mode: `data\logs\`; otherwise `~/.dualscreen_ai/logs/`):
  `crash.log` (native crash traces; written by `faulthandler`, see *Crashes* below), `application.log`, `llm.log` (one JSON line per request: request id, model, provider, prompt/context size, retrieved ids,
  parameters, latency, raw + parsed response, validation, error + stack trace), `question_detection.log`, `retrieval.log`, `errors.log`.
  Settings → Debug → *Open logs folder*.
* **Settings → LLM → Test LLM** sends "What is Python?" without any interview pipeline and reports connection / generation /
  parsing / validation / latency. CLI: `python -m diagnostics --test-llm`.
* **Settings → Debug → Run full diagnostics** (or `python -m diagnostics`): Database, Microphone, VAD, STT, Question Detector,
  Context Manager, Resume Parser, PDF Parser, PDF Retrieval, LLM Connection, LLM Generation, Answer Parser, Display — PASS/FAIL
  with a recommended action for each failure.

## Crashes
If the app ever closes unexpectedly, the next start shows a banner. `logs/crash.log` (the previous run is kept as `crash_previous.log`)
holds the native stack of every thread and `logs/last_operation.json` says what was running (for example `resume.upload: parsing`).
Debug mode makes any widget use from a worker thread raise immediately instead of crashing later.

## Shortcuts (configurable)
`Ctrl+Shift+Space` start/pause · `Ctrl+Shift+A` process current question · `Ctrl+Shift+C` clear answer ·
`Ctrl+Shift+X` clear context · `Ctrl+Shift+M` move AI window to its monitor · `Ctrl+Shift+U` toggle click-through lock.
In the window: `Ctrl+K` command palette · `Ctrl+L` activity log · `Ctrl+Enter` send the typed question · `F1` shortcut list.

## Privacy defaults
Store transcript / questions / audio are all **OFF**. Audio is never written to disk. SQLite stores only settings
(and history/context if you switch that on).

## Project layout
`audio/ vad/ speech/ question/ llm/ display/ context/ settings/ hardware/ database/ tests/ build/` plus `resume/` (parser, storage, manager, retrieval), `knowledge/` (pdf_parser, structure, chunker, index, hybrid search, retriever), `diagnostics/`, and `applog/` (structured logging — not named `logging`, which would shadow the standard library).
`app/engine.py` is the Qt-free pipeline (fully unit-testable); `main.py` wires it to the UI. UI code (`display/`) only presents:
`display/theme.py` (tokens, themes), `display/icons.py` + `display/assets/` (offline SVG icons, fonts), `display/widgets/` (shared widgets),
`display/pages/` (resume and knowledge-base pages); threading helpers are in `app/gui_thread.py` and `app/tasks.py`, live status in `app/status.py`.
Swap engines through `STTEngine` (faster-whisper | whisper.cpp) and `LLMProvider` (Ollama | llama.cpp server).

## Tests
```
pip install -r requirements-dev.txt
pytest                # unit + offscreen UI tests (no models needed; includes a stub Ollama server). Set QT_QPA_PLATFORM=offscreen on a headless machine
pytest -m integration -s   # real WAV → VAD → Whisper → Ollama; needs tests/data/question.wav + models
```

## Build
`build\build.bat` → `dist\DualScreenAI\DualScreenAI.exe`, `dist\DualScreenAI-portable.zip`, and an installer
if Inno Setup (`iscc`) is on PATH. Models are not bundled.

## Known limitations
* Monitors are numbered left→right, top→bottom (not necessarily Windows' "Display 1/2" numbering) — the Display
  settings list resolutions so you can pick the right one.
* Some drivers send no loopback data while nothing is playing; this is normal. Changing the default output
  device mid-meeting is handled by an automatic reconnect (≤ 2 s gap).
* Answer quality and speed depend on the local model; CPU-only 8B models are slow — use `qwen3:4b` or smaller.
* Use responsibly: check your meeting/class/exam rules and local recording-consent laws before using any
  tool that listens to a call.
