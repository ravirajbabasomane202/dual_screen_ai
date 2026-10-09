"""MeetingEngine — the real pipeline, free of any Qt dependency (UI subscribes through `emit`).

  audio frames -> Silero VAD segmenter -> faster-whisper -> QuestionDetector
      -> [semantic check] -> local LLM (clean + answer, streamed) -> events for the UI

Events emitted: status, level, partial, transcript, pending, question, clean_question, answer_delta,
answer_reset, answer_done, restore_previous, history, error, info, debug_trace
"""
import logging
import queue
import re
import threading
import time
from collections import deque

import numpy as np

from app.crashguard import breadcrumb
from app.errors import AnswerRejected, AppError, LLMNotRunning, LLMTimeout
from app.status import ERROR, OK, RUNNING, WARN, get_bus
from applog import format_llm_block, is_debug, log_event, redact
from context.context_manager import ContextManager
from llm.model_manager import create_provider
from llm.service import Extras, LLMService
from question.confidence import Action, combine, decide
from question.detector import QuestionDetector
from question.pipeline import clean_transcript
from question.duplicate_filter import DuplicateFilter
from speech.stt_manager import create_engine, filter_hallucination
from vad.silero_vad import SileroVAD, SpeechSegmenter

log = logging.getLogger(__name__)


class MeetingEngine:
    def __init__(self, settings, emit, db=None, audio=None, vad=None, stt=None, provider=None, hardware=None, bus=None):
        self.s, self.emit, self.db = settings, emit, db
        self.bus = bus or get_bus()                # every stage reports here (pipeline strip, activity feed, health)
        self.context = ContextManager()
        self.services = None
        if db is not None:                       # resume + PDF knowledge base need the database
            from app.services import Services
            self.services = Services(db, settings, self.context)
        self.resume = self.services.resume if self.services else None
        self.kb = self.services.kb if self.services else None
        self.feedback = self.services.feedback if self.services else None
        self._last_answer: dict = {}
        self.apply_interview_profile()
        self.detector = QuestionDetector(self.services.resume_retriever if self.services else None, self.context)
        self.dup = DuplicateFilter()
        self._audio, self._vad, self._stt, self._provider, self._hw = audio, vad, stt, provider, hardware
        # components passed in (tests, file replay) are kept across restarts; built ones are rebuilt
        self._injected = {"audio": audio is not None, "vad": vad is not None,
                          "stt": stt is not None, "provider": provider is not None}
        self.llm: LLMService | None = None
        self._frames: queue.Queue = queue.Queue(maxsize=600)
        self._stt_q: queue.Queue = queue.Queue()
        self._brain_q: queue.Queue = queue.Queue()
        self._stop, self._cancel = threading.Event(), threading.Event()
        self._paused = False
        self._threads: list[threading.Thread] = []
        self._stt_busy = self._busy = False
        self._pending: tuple[str, float] | None = None
        self._last_text = ""
        self._recent: deque = deque(maxlen=6)
        self._last_level = 0.0

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._stop.clear()
        self._spawn(self._boot)

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()
        if self._audio:
            try:
                self._audio.stop()
            except Exception:
                log.exception("audio stop")
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2)
        self._threads.clear()
        for attr, key in (("_audio", "audio"), ("_vad", "vad"), ("_stt", "stt"), ("_provider", "provider")):
            if not self._injected[key]:
                setattr(self, attr, None)

    def restart(self) -> None:
        self.stop()
        self.start()

    def _spawn(self, fn) -> None:
        t = threading.Thread(target=fn, daemon=True, name=fn.__name__)
        t.start()
        self._threads.append(t)

    def _err(self, e: Exception, fatal: bool = False) -> None:
        msg = e.user_message if isinstance(e, AppError) else f"{type(e).__name__}: {e}"
        log.error("engine error: %s", msg, exc_info=not isinstance(e, AppError))
        self.bus.log("ERROR", msg.splitlines()[0][:200], stage="engine", detail=msg)
        self.emit("error", message=msg, fatal=fatal, kind=type(e).__name__)
        if fatal:
            self.emit("status", state="error")

    # ---------------------------------------------------------------- boot
    def _resolve(self):
        from hardware.detector import detect, recommend
        if self._hw is None:
            self._hw = detect()
        rec = recommend(self._hw)
        model = self.s.get("speech.model")
        device = self.s.get("speech.device")
        return (rec.stt_model if model == "auto" else model,
                rec.compute_device if device == "auto" else device,
                rec.llm_model if self.s.get("ai.model") == "auto" else self.s.get("ai.model"))

    def _boot(self) -> None:
        try:
            self.emit("status", state="loading")
            breadcrumb("engine: loading models")
            stt_model, device, llm_model = self._resolve()
            if self._stt is None:
                with self.bus.track("stt_model", f"Loading speech model ({stt_model})…") as t:
                    self._stt = create_engine(self.s, stt_model, device)
                    self._stt.load()
                    t.ok(f"Speech model ready ({stt_model}, {device})")
            else:
                self.bus.publish("stt_model", OK, "Speech engine provided")
            if self._vad is None:
                with self.bus.track("vad_model", "Loading voice-activity model…") as t:
                    self._vad = SileroVAD()
                    t.ok("Voice-activity model ready")
            if self._provider is None:
                self._provider = create_provider(self.s, llm_model)
            self.llm = LLMService(self._provider, self.s, self.context)
            if self.services:
                self.services.use_provider(self._provider)
            self.emit("info", message=f"Speech: {self._stt.name}/{stt_model}/{device} · LLM: {llm_model}")
        except Exception as e:
            self._err(e, fatal=True)
            return
        for fn in (self._vad_worker, self._stt_worker, self._brain_worker):
            self._spawn(fn)
        self._spawn(self._warmup)
        try:
            if self._audio is None:
                from audio.audio_manager import AudioManager
                self._audio = AudioManager(self.s, self._on_frame, on_error=self._err)
            self._audio.start()
            self.bus.publish("audio", OK, "Capturing system audio" + (" + microphone" if self.s.get("audio.mic_enabled") else ""))
        except Exception as e:
            self.bus.publish("audio", ERROR, getattr(e, "user_message", str(e)))
            self._err(e, fatal=True)
            return
        breadcrumb("engine: listening")
        self.emit("status", state="listening")

    def _warmup(self) -> None:
        try:
            self.bus.publish("llm_model", RUNNING, "Connecting to the language model…")
            if not self._provider.is_available():
                self.bus.publish("llm_model", ERROR, "The language model server is not running")
                self._err(LLMNotRunning())
            elif not self._provider.has_model(self._provider.model):
                from app.errors import LLMModelMissing
                self.bus.publish("llm_model", ERROR, f"Model '{self._provider.model}' is not installed")
                self._err(LLMModelMissing(f"Model '{self._provider.model}' is not installed.\n\n"
                                          f"Run: ollama pull {self._provider.model}"))
            else:
                self._provider.warmup()
                self.bus.publish("llm_model", OK, f"Model loaded ({self._provider.model})")
        except Exception as e:
            self.bus.publish("llm_model", ERROR, getattr(e, "user_message", str(e)))
            self._err(e)

    # ---------------------------------------------------------------- controls
    @property
    def paused(self) -> bool:
        return self._paused

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.emit("status", state="paused" if self._paused else "listening")

    def clear_answer(self) -> None:
        self._cancel.set()
        self._pending = None
        self.emit("restore_previous", clear=True)

    def clear_context(self) -> None:
        self.context.clear(keep_profile=True)
        self.dup.clear()
        if self.db:
            self.db.clear_context()
        self.emit("info", message="Context cleared")

    def process_current(self) -> None:
        """Hotkey / button: answer the pending 'possible question', else the latest transcript."""
        text = self._pending[0] if self._pending else self._last_text
        if text:
            self._brain_q.put(("manual", text, time.monotonic()))

    def ask_text(self, text: str, practice: bool = False, length: str = "") -> None:
        """Answer a question typed by the user (quick re-ask for something STT missed, or practice / mock mode).
        Practice questions run the whole pipeline but never touch the interview memory, history or database."""
        text = " ".join(text.split())
        if text:
            self._brain_q.put(("manual", text, time.monotonic(), {"practice": practice, "length": length}))

    def regenerate(self, direction: str = "") -> bool:
        """Answer the last question again, optionally 'shorter' / 'longer' than the configured length."""
        last = self._last_answer
        if not last:
            return False
        order = ["Very Short", "Short", "Medium", "Detailed"]
        cur = last.get("length") or self.s.get("ai.answer_length")
        i = order.index(cur) if cur in order else 1
        length = order[max(0, min(len(order) - 1, i + {"shorter": -1, "longer": 1}.get(direction, 0)))]
        self.ask_text(last["question"], practice=last.get("practice", False), length=length)
        return True

    # ---------------------------------------------------------------- feedback / interview target
    def apply_interview_profile(self) -> None:
        if not self.services:
            return
        p = self.services.profile_store.load()
        self.context.target_text = p.to_text(int(self.s.get("profile.jd_chars", 1200)))

    def mark_question(self, text: str, is_question: bool, detected: str = "", confidence: float = 0.0, action: str = "") -> dict:
        """'This was a question' / 'Not a question'. Returns {'tuned': new min_confidence or None, 'answering': bool}."""
        if not self.feedback:
            return {"tuned": None, "answering": False}
        from question.feedback import NOT_QUESTION, QUESTION
        self.feedback.record(text, QUESTION if is_question else NOT_QUESTION, detected, confidence, action)
        tuned = self.feedback.suggest_min_confidence(float(self.s.get("question.min_confidence", 0.30)))
        if tuned is not None:
            self.s.set("question.min_confidence", tuned)
            self.feedback.mark_tuned()
            self.bus.log("INFO", f"Question threshold adjusted to {tuned:.2f} from your corrections", stage="detect")
        answering = bool(is_question and str(action).startswith(("ignored", "confirm")))
        if answering:
            self.ask_text(text)
        return {"tuned": tuned, "answering": answering}
    # ---------------------------------------------------------------- audio -> VAD
    def _on_frame(self, frame: np.ndarray) -> None:
        if self._paused:
            return
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass

    def feed_frame(self, frame: np.ndarray) -> None:   # used by tests / file replay
        self._on_frame(frame)

    def _vad_worker(self) -> None:
        seg = SpeechSegmenter(self._vad.prob, self._on_segment, sensitivity=float(self.s.get("audio.vad_sensitivity")),
                              reset_fn=getattr(self._vad, "reset", None))
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            seg.sensitivity = float(self.s.get("audio.vad_sensitivity"))
            now = time.monotonic()
            if now - self._last_level > 0.1:
                self._last_level = now
                self.emit("level", rms=float(np.sqrt(np.mean(frame ** 2))))
            try:
                seg.feed(frame)
            except Exception as e:
                self._err(e)

    def _on_segment(self, kind: str, audio: np.ndarray | None = None) -> None:
        if kind == "speech_start":
            self.bus.publish("vad", RUNNING, "Speech detected", quiet=True)
            if not self._busy:
                self.emit("status", state="listening")
        elif kind == "partial":
            if self._stt_q.empty() and not self._stt_busy:
                self._stt_q.put(("partial", audio, time.monotonic()))
        elif kind == "final":
            self.bus.publish("vad", OK, f"Segment of {len(audio) / 16000:.1f}s" if audio is not None else "Segment", quiet=True)
            self._stt_q.put(("final", audio, time.monotonic()))

    # ---------------------------------------------------------------- STT
    def _stt_worker(self) -> None:
        while not self._stop.is_set():
            try:
                kind, audio, t_end = self._stt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "partial" and not self._stt_q.empty():
                continue
            self._stt_busy = True
            try:
                if kind == "final" and not self._busy:
                    self.emit("status", state="transcribing")
                if kind == "final":
                    self.bus.publish("stt", RUNNING, "Transcribing…")
                text = filter_hallucination(self._stt.transcribe(audio, partial=(kind == "partial")),
                                            len(audio) / 16000)
                if kind == "final":
                    self.bus.publish("stt", OK if text else WARN, f"“{text[:80]}”" if text else "No speech recognised")
                if kind == "partial":
                    if text:
                        self.emit("partial", text=text)
                elif text:
                    self.emit("transcript", text=text)
                    self._brain_q.put(("final", text, t_end))
                elif not self._busy:
                    self.emit("status", state="listening")
            except Exception as e:
                self.bus.publish("stt", ERROR, getattr(e, "user_message", str(e)))
                self._err(e)
            finally:
                self._stt_busy = False

    # ---------------------------------------------------------------- brain
    def _brain_worker(self) -> None:
        while not self._stop.is_set():
            try:
                kind, text, t_end, *rest = self._brain_q.get(timeout=0.2)
            except queue.Empty:
                continue
            opts = rest[0] if rest else {}
            try:
                if kind == "manual":
                    self._answer(text, 1.0, t_end, manual=True, practice=bool(opts.get("practice")),
                                 length=opts.get("length", ""))
                else:
                    self._handle_final(text, t_end)
            except Exception as e:
                self._err(e)
                self._busy = False
                self.emit("status", state="listening")

    def _log_detection(self, raw: str, det, action: str, final_conf: float, llm_used: bool) -> None:
        c = det.classification
        fields = dict(type=c.type.value if c else "?", confidence=det.confidence, final_confidence=final_conf,
                      action=action, reason=c.reason if c else "", signals=det.signals,
                      resume_relevance=c.resume_relevance if c else 0.0,
                      resume_terms=c.resume_terms if c else [], llm_check=llm_used,
                      words=len(raw.split()), transcript=redact(raw), detected_sentence=redact(det.question))
        log_event("question", f"detected {fields['type']} -> {action}", **fields)
        self.bus.publish("detect", OK if not action.startswith("ignored") else WARN,
                         f"{fields['type']} {det.confidence:.0%} → {action}")
        self.emit("detection", text=raw, type=fields["type"], confidence=final_conf, action=action, reason=fields["reason"])
        if is_debug():
            import logging
            logging.getLogger("dsai.question").debug("block", extra={"pretty": (
                f"\n[QUESTION DETECTION]\n\nTranscript:\n\"{raw}\"\n\nDetected:\n{fields['type']}\n\n"
                f"Confidence:\n{det.confidence:.2f}  (final {final_conf:.2f}, action: {action})\n\n"
                f"Reason:\n{fields['reason']}\n\nResume relevance:\n{fields['resume_relevance']:.2f}\n")})

    def _handle_final(self, text: str, t_end: float) -> None:
        now = time.monotonic()
        self._recent.append((now, text))
        self.bus.reset_pipeline()
        self.bus.publish("detect", RUNNING, "Classifying…", quiet=True)
        det = self.detector.detect(text)
        override = self.feedback.override_for(det.text) if self.feedback else None
        if override == "not_question":
            self._log_detection(text, det, "ignored:user_feedback", det.confidence, False)
            self.emit("status", state="listening")
            return
        c = det.classification
        standalone = det.confidence >= 0.85 or (c is not None and c.type.value in ("SMALL_TALK", "NOISE"))
        if not standalone and len(self._recent) >= 2 and now - self._recent[-2][0] < 6:   # question split by a pause
            prev = self._recent[-2][1]
            if self.detector.detect(prev).confidence < 0.6:       # an already-complete previous question is not extended
                merged = self.detector.detect(prev + " " + text)
                if merged.confidence > det.confidence:
                    det = merged
        self._last_text = det.text
        self.dup.timeout = float(self.s.get("question.duplicate_timeout"))
        floor = float(self.s.get("question.min_confidence", 0.30))
        if det.confidence < floor:
            self._log_detection(text, det, "ignored:below_min_confidence", det.confidence, False)
            self.emit("status", state="listening")
            return
        if self.dup.is_duplicate(det.question):
            self._log_detection(text, det, "ignored:duplicate", det.confidence, False)
            self.emit("status", state="listening")
            return
        auto, confirm = float(self.s.get("question.auto_threshold")), float(self.s.get("question.confirm_threshold"))
        conf, text_for_llm = det.confidence, clean_transcript(det.text)
        used_llm = False
        if override == "question":                   # the user told us this sentence is a question
            conf = max(conf, auto)
        if conf < auto:   # ambiguous -> ask the local LLM (only this band; clear cases never pay the latency)
            try:
                res = self.llm.classify(det.text, self._cancel)
                used_llm = True
                conf = combine(det.confidence, res.is_question, res.confidence)
                text_for_llm = res.clean_question or text_for_llm
            except (LLMNotRunning, LLMTimeout) as e:
                self._err(e)
                conf = det.confidence
            except Exception as e:
                log.exception("semantic check failed; using rule confidence")
                self._err(e)
        action = decide(conf, auto, confirm)
        if action is Action.AUTO and self.s.get("question.auto_answer"):
            self._log_detection(text, det, "answer", conf, used_llm)
            self._answer(text_for_llm, conf, t_end)
        elif action is Action.IGNORE:
            self._log_detection(text, det, "ignored:low_confidence", conf, used_llm)
            self.emit("status", state="listening")
        else:  # CONFIRM, or AUTO with auto-answer switched off
            self._log_detection(text, det, "confirm", conf, used_llm)
            self._pending = (text_for_llm, conf)
            self.emit("pending", text=text_for_llm, confidence=conf)

    def _retrieve(self, text: str) -> Extras:
        if not self.services:
            return Extras()
        p = self.context.persistent
        deictic = bool(re.search(r"\b(this|that|it|its|there|here|those|these)\b", text, re.I))
        hint = (p.current_project or p.current_topic) if (deictic and self.context.enabled) else ""
        r = self.services.retriever.retrieve(text, hint=hint)
        return Extras(resume=r.resume, kb=r.kb, ids=r.ids)

    def _debug_trace(self, raw: str, text: str, det_info, extras: Extras, res, answer: str, force: bool = False) -> None:
        if not force and not (is_debug() and self.s.get("debug.show_trace", True)):
            return
        d = getattr(self.llm, "last_debug", {}) or {}
        msgs = d.get("messages") or []
        prompt = "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in msgs)
        out = [
            "ORIGINAL TRANSCRIPT\n" + raw, "CLEAN TRANSCRIPT\n" + text,
            f"DETECTED QUESTION / TYPE\n{det_info}",
            "RETRIEVED RESUME\n" + ("\n".join(f"- {c.label} (score {c.score:.2f})" for c in extras.resume) or "—"),
            "RETRIEVED PDF\n" + ("\n".join(f"- {h.label} (score {h.score:.2f})" for h in extras.kb) or "—"),
            "CONTEXT USED\n" + str(d.get("plan")) + "\n" + (self.context.memory_text() or "—"),
            "LLM PROMPT\n" + (prompt or "—"), "LLM RESPONSE\n" + str(d.get("raw", "")),
            "ANSWER VALIDATION\n" + str(d.get("validation")), "FINAL ANSWER\n" + answer]
        self.emit("debug_trace", text="\n\n" + ("\n" + "-" * 60 + "\n\n").join(out))

    def _answer(self, text: str, conf: float, t_end: float, manual: bool = False, practice: bool = False,
                length: str = "") -> None:
        self._cancel.clear()
        self._busy, self._pending = True, None
        first = [None]
        parts: list[str] = []
        dup_hit = [False]
        resets = [0]
        bus = self.bus
        if manual:
            bus.reset_pipeline()
        breadcrumb("engine: answering")
        try:
            if not self._provider.is_available():
                bus.publish("llm", ERROR, "The language model server is not running")
                raise LLMNotRunning()
            if self.context.enabled != bool(self.s.get("ai.context")):
                self.context.enabled = bool(self.s.get("ai.context"))
            self.context.apply_settings(self.s)
            self.emit("status", state="question_detected")
            self.emit("question", raw=text, confidence=conf, practice=practice)
            self.emit("status", state="cleaning")
            with bus.track("context", "Updating interview memory…") as t:
                ents = self.services.resume_retriever.entities() if self.services else None
                learned = {} if practice else self.context.observe(text, ents)      # practice never alters the memory
                t.ok("Memory updated" if learned else "Memory unchanged")
            with bus.track("retrieve", "Searching resume and knowledge base…") as t:
                extras = self._retrieve(text)
                t.ok(f"{len(extras.resume)} resume · {len(extras.kb)} knowledge-base chunk(s)")
            extras.length = length
            max_attempts = 1 + (int(self.s.get("ai.max_retries", 2)) if self.s.get("ai.strict_validation", True) else 0)

            def on_clean(q: str) -> None:
                # the cleaned wording may reveal a repeat the raw transcript hid
                if q:
                    self.emit("clean_question", text=q)
                    if not manual and self.dup.is_duplicate(q):
                        dup_hit[0] = True
                        self._cancel.set()
                        return
                    self.dup.register(q)
                if not manual:
                    self.dup.register(text)    # registered only AFTER the check, never against itself
                self.emit("status", state="generating")

            def on_delta(d: str) -> None:
                if first[0] is None:
                    first[0] = time.monotonic()
                    bus.publish("llm", RUNNING, "Generating the answer…", quiet=True)
                parts.append(d)
                self.emit("answer_delta", text=d)

            def on_reset() -> None:
                parts.clear()
                resets[0] += 1
                bus.publish("validate", WARN, f"Answer rejected (it summarised the question) — regenerating "
                                              f"(attempt {min(resets[0] + 1, max_attempts)}/{max_attempts})")
                self.emit("answer_reset")

            llm_t = bus.track("llm", "Waiting for the language model…")
            try:
                res = self.llm.answer_stream(text, on_clean, on_delta, self._cancel, extras=extras, on_reset=on_reset)
            except Exception as e:
                llm_t.fail(e)
                raise
            llm_t.ok(f"Answer generated ({time.monotonic() - (first[0] or time.monotonic()):.1f}s of streaming)")
            if dup_hit[0] or (not res.is_question and not manual):
                self.emit("restore_previous", clear=False)
                self.emit("status", state="listening")
                return
            if self._cancel.is_set() and not res.answer:       # user pressed "clear"
                self.emit("status", state="listening")
                return
            if not res.valid:                                  # the model described the question instead of answering
                bus.publish("validate", ERROR, "Answer rejected after every attempt")
                self.emit("restore_previous", clear=False)
                self._err(AnswerRejected())
                self.emit("status", state="listening")
                return
            bus.publish("validate", OK if res.attempts <= 1 else WARN,
                        "Answer is valid" if res.attempts <= 1 else f"Valid after {res.attempts} attempts")
            answer = res.answer or "".join(parts)
            if not answer:
                bus.publish("validate", ERROR, "The model returned no answer")
                self.emit("restore_previous", clear=False)
                self.emit("error", message="The model returned no answer.", fatal=False)
                self.emit("status", state="listening")
                return
            q = res.clean_question or text
            final_conf = combine(conf, True, res.confidence) if (res.confidence and not manual) else conf
            done = time.monotonic()
            sources = [c.label for c in extras.resume] + [h.label for h in extras.kb]
            source_refs = ([{"kind": "resume", "label": c.label, "id": c.id} for c in extras.resume]
                           + [{"kind": "kb", "label": h.label, "doc": h.doc_path, "page": h.page, "chunk": h.chunk_id}
                              for h in extras.kb])
            self._last_answer = {"question": q, "answer": answer, "length": length or self.s.get("ai.answer_length"),
                                 "practice": practice, "grounding": res.grounding}
            self.emit("answer_done", question=q, answer=answer, confidence=final_conf,
                      seconds=done - t_end, first_token=(first[0] - t_end) if first[0] else None,
                      needs_clarification=res.needs_clarification, grounding=res.grounding, intent=res.intent,
                      sources=sources, source_refs=source_refs, practice=practice,
                      warnings=(res.validation or {}).get("warnings", []))
            bus.publish("display", OK, "Answer displayed")
            if not practice:
                self.context.add(q, answer)
                if self.db and self.s.get("privacy.store_questions"):
                    self.db.add_question(q, answer)
                    self.db.save_context(self.context.pairs())
                    if self.services:
                        self.services.save_memory()
            self._debug_trace(text, text, f"{self.detector.classifier.classify(text).to_dict()}", extras, res, answer,
                              force=practice)
            if not practice:
                self.emit("history", question=q, answer=answer)
            self.emit("status", state="ready")
        finally:
            self._busy = False
            breadcrumb("engine: listening")

    # ---------------------------------------------------------------- health (for the header indicators)
    def health(self):
        from app.health import compute_health
        return compute_health(self, self.s, self.bus)

    # ---------------------------------------------------------------- diagnostics (also used by the settings UI)
    def test_llm(self):
        from llm.diagnostics import test_llm
        if self._provider is None:
            stt_model, device, llm_model = self._resolve()
            provider = create_provider(self.s, llm_model)
        else:
            provider = self._provider
        return test_llm(provider, self.s)

    def run_diagnostics(self):
        from diagnostics.runner import run_full_diagnostics
        return run_full_diagnostics(self.s, self.db, engine=self)
