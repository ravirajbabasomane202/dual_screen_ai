# Part 2 — premium UI, resume-upload crash fix, live status, Knowledge Base manager

## BUGS FOUND → ROOT CAUSES (with the stage that failed)
| Symptom | Stage | Root cause | Fix |
|---|---|---|---|
| App silently closes after a resume upload (resume was saved) | resume upload → settings notification | The upload ran on a worker thread; `settings.set("resume.path")` ran every `on_change` callback **on that thread**, and `main.on_settings` restyled windows, moved them between monitors and restarted the engine from it. Qt widgets touched off the GUI thread kill the process natively (no Python traceback). `apply_theme()` also ran on *every* settings change. | `SettingsManager`, `ResumeManager` and `StatusBus` deliver UI callbacks on the GUI thread (`app/gui_thread.py`); data-model listeners opt in with `thread="inline"`; `apply_theme()` only for `display.*` keys; `assert_gui_thread()` in widget-mutating methods (raises in debug mode and tests); `faulthandler` → `logs/crash.log`, last-operation breadcrumb, unclean-exit banner |
| Possible "Signal source has been deleted" in worker threads at shutdown | task completion | cached dispatcher object destroyed with the QApplication | `call_on_gui` rebuilds / drops quietly |
| `Internal C++ object already deleted` after closing a dialog | theme change | lambdas connected to `ThemeManager.changed` outlive their widgets | `theme.on_changed(owner, fn)` auto-disconnects; bus subscriptions drop on `destroyed` |
| Whole application restyled for every new window | UI start-up | `ThemeManager.set` always re-applied the stylesheet | no-op when theme / accent / size are unchanged |
| Legacy databases (no version table) were migrated without a backup | schema migration | backup only when `version > 0` | backup whenever user tables exist |

## FEATURES ADDED
* **Status system**: `StatusBus` (thread-safe, ordered, GUI-thread delivery), pipeline strip (Audio→VAD→STT→Detect→Context→Retrieve→LLM→Validate→Display), header health indicators (clickable), activity drawer (**Ctrl+L**: filter, copy, open log folder, redacted diagnostics bundle), toasts + inline error banner (cause · action · details) instead of modal error boxes, progress rows with a real cooperative **Cancel** (`app/tasks.py`).
* **Knowledge Base manager**: document table (sort / search / multi-select / badges Indexed · Indexing… · Needs re-index · No text found · Error · disabled), Add files / Add folder (several folders, auto-sync at start, optional watcher), New note, New Q&A (highest retrieval priority), edit in place (only changed chunks are re-embedded, version counter), rename, disable / enable, delete with **Undo**, bulk re-index / delete, preview, chunk viewer (enable / disable per chunk), search playground (fused / BM25 / vector scores, source and page), index statistics, empty state, drag-and-drop.
* **Resume**: structured overview, editable parsed fields that survive re-parse, parser warnings, version history with restore, several named profiles (one active).
* **Premium UI** (dual-screen behaviour unchanged): design tokens, Dark / Light / High-contrast + accent picker, bundled Inter + DejaVu Sans Mono, 73 offline SVG icons, 150–200 ms transitions (off with *Reduce motion*), focus rings, button hierarchy; new AI window (question card with type badge and confidence ring, answer card with grounding badge and source chips, toolbar: Copy · Shorter · Longer · Regenerate · Pin · Why? · 👍/👎, history with search, debug panel, mini mode, click-through lock **Ctrl+Shift+U**); settings with icon sidebar, search, sticky Save / Reset, inline validation, help text, restart indicator, live theme preview; first-run wizard with progress dots (Welcome → Audio → Speech → LLM → Resume → Knowledge base → Finish, each skippable).
* **Convenience**: practice / mock mode, quick re-ask box, "question / not a question" buttons on transcript lines (tune `question.min_confidence`, per-sentence overrides), interview profile (company / role / job description), answer-style presets (Concise · STAR · Technical deep-dive · Spoken-natural), export / import everything as one zip + automatic pre-migration backup, KB auto-update toast at start, **Pull model** with progress, command palette (**Ctrl+K**), shortcut list (**F1**), Privacy page with stored-data report and **Wipe all data**, crash-recovery banner, answer-ready cue (visual / sound / none).

## FILES CHANGED
New: `app/{gui_thread,crashguard,status,tasks,health,errorinfo,bundle,backup}.py`, `resume/editor.py`, `knowledge/watcher.py`,
`question/feedback.py`, `context/interview_profile.py`, `display/{theme,icons,i18n}.py`, `display/widgets/*`, `display/pages/*`,
`display/assets/{icons,fonts}`.
Rewritten: `display/{ai_window,settings_window,settings_pages,wizard}.py`, `main.py`, `knowledge/index.py`, `resume/{manager,storage}.py`,
`database/{migrations,repositories}.py`, `settings/settings_manager.py`.
Modified: `app/{engine,services,errors,paths}.py`, `llm/{prompt,service,model_manager}.py`, `context/context_manager.py`, `database/sqlite.py`,
`build/DualScreenAI.spec` (bundles icons + fonts), `README.md`.

## TESTS ADDED (`tests/`)
`test_thread_affinity` (regression + control proving the old behaviour is detected, stress: rapid upload→replace→delete→re-parse from 3 threads, corrupt / empty / 20 MB files, upload while the engine answers, crash breadcrumbs), `test_status_health_backup` (bus ordering / thread-safety, health, error text, bundle redaction, log redaction, feedback tuning, interview profile, backup round trip, wipe, pre-migration backup), `test_kb_manager` (add file / note / Q&A, edit, disable, delete + undo + expiry, chunk disable, playground, rename, needs-re-index, bulk, **cancel**, drag-and-drop, **event-loop latency p99 < 50 ms while indexing**), `test_resume_manager_ui`, `test_design_system` (token completeness, contrast, QSS placeholders, icons incl. every name used in code, no hard-coded colours, UI/logic separation, settings search / validation / reset / live preview), `test_ui_everything` (offscreen: every window and page, 3 themes, every button clicked, wizard end to end, palette, drawer, pipeline strip).

## TEST RESULTS
Before: 198 passed, 2 skipped. After: **281 passed, 2 skipped** (the 2 skips are the pre-existing integration tests that need real models / audio).
Existing tests changed (disclosed): the schema-version assertions use `latest_version()` (schema evolved to v7), and the settings-window test reads the new sidebar instead of the old tab bar.

## KNOWN LIMITATIONS / NOT VERIFIED
* Only run offscreen on Linux: **real Windows multi-monitor behaviour, WASAPI audio, Whisper and a real local LLM were not exercised**; visual polish (spacing, fonts, animations) was never looked at on a real display.
* Click-through lock is untested on Windows; it is unlocked with the global hotkey (needs the `keyboard` package).
* There is no persistent status bar; status appears as the pipeline strip, health icons, toasts and the activity drawer.
* Not done: session transcript viewer / export, translation, per-question notes. Screen-reader support is names / descriptions only (not tested with a reader). `tr()` wraps the main strings, no translation catalogue ships.
* The UI-latency test warms the page up first (first paint of the large stylesheet takes several hundred ms and is not indexing work).

## WHERE LOGS LIVE
`<app folder>/logs/`: `application.log`, `llm.log`, `question_detection.log`, `retrieval.log`, `errors.log`, **`crash.log`** (native crash traces; the previous run's copy is kept as `crash_previous.log` and `last_operation.json` names what was running). Databases are backed up to `<app folder>/backups/` before schema upgrades.

---

# Interview-assistant upgrade — report

## BUGS FOUND → ROOT CAUSES
| Symptom | Root cause (found by reading the pipeline) | Fix |
|---|---|---|
| Interview prompts without "?" ("Tell us about yourself", "Introduce yourself") ignored | `question/detector.py` only knew wh/aux/few imperatives; no interview-prompt, conversational or follow-up patterns; `How many …` scored only 0.5 | `question/classifier.py` (5 types + confidence + reason), richer wh rules |
| Long questions lose their first half | `DetectionResult.text = text[-500:]`; memory kept `answer[:300]` | full utterance kept; token-aware budgeting; answers kept to 600 chars (configurable) |
| "The speaker is asking …" instead of an answer | System prompt framed the model as an *academic assistant* that must "determine whether the speaker is asking a question" and "correct the grammar", and labelled the input `Transcript:`; schema started with `clean_question`, so small models echoed it. No output validation existed | New candidate-voice prompt with rules + examples, contract `question → intent → grounding → answer → confidence`, streaming guard + `llm/validation.py`, automatic regeneration |
| Prompt silently loses its instructions on big contexts | `num_ctx` hard-coded to 4096 and nothing budgeted the prompt; Ollama drops the *start* of an over-long prompt (the system prompt) without an error | `ai.context_window` setting sent as `num_ctx`, `ContextBudgeter`, detection of `prompt_eval_count ≈ num_ctx` (logged) |
| Truncated / cut-off answers | `max_tokens_for` = answer size + 120, but the JSON also echoes the question | budget now adds 200 + 1.2×question tokens; `finish_reason=length` triggers a retry with 1.6× tokens |
| LLM errors invisible | `raise_for_status()` discarded the HTTP body; `warmup` swallowed errors; no request logging | error bodies surfaced, context-overflow/think-option handled, one JSON line per request in `llm.log`, `errors.log` with stack traces |
| Small talk merged into the previous question | pause-stitching merged any two utterances < 6 s apart | only merged when the new segment is not standalone and the previous one is not already a question |
| (found while testing) resume profile missing in memory | listener order in `Services` | fixed + test |

## FILES CHANGED
Modified: `app/engine.py`, `app/errors.py`, `context/context_manager.py`, `database/sqlite.py`, `display/ai_window.py`,
`display/settings_window.py`, `llm/{base,llama_cpp,model_manager,ollama,prompt,service}.py`, `question/detector.py`,
`settings/settings_manager.py`, `main.py`, `requirements.txt`, `build/DualScreenAI.spec`, `README.md`.
New packages: `applog/`, `resume/`, `knowledge/`, `diagnostics/`; new modules `app/services.py`,
`context/{short_term,persistent,token_manager}.py`, `database/{migrations,repositories}.py`,
`display/settings_pages.py`, `llm/{validation,trace,diagnostics}.py`, `question/{classifier,pipeline}.py`.
Tests added: `test_interview_detection`, `test_answer_generation`, `test_context_memory`, `test_resume`, `test_knowledge`,
`test_engine_interview`, `test_ollama_http`, `test_diagnostics_logging_ui` (+ `tests/pdfgen.py`).

## TEST RESULTS
Baseline before changes: 56 passed, 2 skipped. After: **198 passed, 2 skipped** (the 2 skips are the pre-existing integration
tests that need real models/audio). The real `main.py` boots headless and creates the `logs/` files.

## LLM DIAGNOSTIC RESULTS — please read
The sandbox has **no Ollama / llama.cpp / model files**, so I could not run your actual local model. The root causes above come
from reading the code, and the fixes are proven against scripted providers and a stub Ollama HTTP server (payloads, `num_ctx`,
HTTP errors, truncation, `think` retry). On your machine run **Settings → LLM → Test LLM** or `python -m diagnostics --test-llm`
to see whether your model follows the contract; if it still describes questions after 3 attempts you will see
"The model kept describing the question…" and the raw replies in `logs/llm.log` — that means the model is too small/not
instruction-tuned (use `qwen3:4b`+ or similar).

## KNOWN LIMITATIONS
* Resume/PDF parsing is heuristic (no OCR: scanned PDFs are reported as "no text"). Table detection from PDFs relies on column spacing.
* The offline hashing embedder is lexical-semantic, not truly semantic; set an Ollama embedding model for better recall.
* Microphone / Silero VAD / Whisper / WASAPI checks and the GUI on a real multi-monitor Windows desktop could not be exercised in this Linux sandbox.
* Grounding labels come from the model; the validator only warns when numbers in a "resume"/"knowledge_base" answer are absent from the sources.
* `applog/` replaces the requested `logging/` package name (it would shadow Python's `logging`).

## HOW TO RUN
`pip install -r requirements.txt` → `python main.py`. Tests: `pip install -r requirements-dev.txt && pytest`.
Diagnostics: `python -m diagnostics` (`--test-llm`, `--no-hardware`, `--debug`).
## DEBUG MODE / LOGS
Settings → Debug → *Enable debug mode*. Logs: `%APPDATA%\DualScreenAI\logs\` (or `~/.dualscreen_ai/logs/`, portable: `data/logs/`):
`application.log, llm.log, question_detection.log, retrieval.log, errors.log`.
