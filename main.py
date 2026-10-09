"""Dual-Screen Meeting AI Assistant — entry point. Wires the packages together; contains no business logic."""
import io
import logging
import sys
import threading
import traceback

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog

from app import crashguard
from app.engine import MeetingEngine
from app.gui_thread import call_on_gui, set_strict
from app.hotkeys import HotkeyManager
from app.paths import app_dir
from app.status import BusLogHandler, get_bus
from app.tasks import run_task
from applog import configure as configure_logging
from applog import get_log_dir
from database.sqlite import Database
from display import theme
from display.ai_window import AIWindow
from display.monitor_manager import MonitorManager
from display.settings_window import SettingsWindow
from display.widgets.command_palette import Command, CommandPalette, ShortcutsOverlay
from display.wizard import SetupWizard
from settings.settings_manager import SettingsManager

log = logging.getLogger("app.main")


class Bridge(QObject):
    """Engine threads call bridge(name, **payload); Qt delivers it on the GUI thread."""
    event = Signal(str, object)

    def __call__(self, name, **payload):
        self.event.emit(name, payload)


_unhandled_sink = [None]          # set in main(): callable(message, details) that shows a non-blocking toast/banner


def _report_unhandled(message: str, details: str) -> None:
    sink = _unhandled_sink[0]
    if sink is not None:
        try:
            call_on_gui(sink, message, details)
        except Exception:
            pass


def _harden_frozen_app() -> None:
    """A windowed (console=False) PyInstaller exe has sys.stdout/stderr == None, which crashes tqdm/huggingface
    downloads and print(); and uncaught errors vanish silently. Give them somewhere to go."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, io.StringIO())

    def hook(etype, value, tb):
        """Never modal: the stack trace goes to errors.log, the user sees a toast + banner."""
        msg = "".join(traceback.format_exception(etype, value, tb))
        logging.getLogger("app").error("Unhandled exception:\n%s", msg)
        _report_unhandled(f"{etype.__name__}: {value}", msg)
    sys.excepthook = hook

    prev = threading.excepthook

    def thread_hook(args):
        prev(args)                                   # crashguard's hook logs the traceback
        if args.exc_type is not SystemExit:
            _report_unhandled(f"{args.exc_type.__name__}: {args.exc_value}", "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
    threading.excepthook = thread_hook


def main() -> int:
    _harden_frozen_app()
    app = QApplication(sys.argv)
    db = Database()
    settings = SettingsManager(db)
    logs = app_dir() / "logs"
    # logs/: application.log, llm.log, question_detection.log, retrieval.log, errors.log, crash.log
    previous_crash = crashguard.check_previous_crash(logs)          # BEFORE install() truncates crash.log
    crashguard.install(logs)
    configure_logging(logs, debug=bool(settings.get("debug.enabled")), level=settings.get("debug.log_level"))
    set_strict(bool(settings.get("debug.enabled")))                 # debug mode: thread-affinity violations raise
    bus = get_bus()
    logging.getLogger().addHandler(BusLogHandler(bus))
    theme.apply_settings(settings)

    if not settings.get("app.first_run_done"):
        SetupWizard(settings, db=db).exec()

    bridge = Bridge()
    engine = MeetingEngine(settings, bridge, db=db, bus=bus)
    win = AIWindow(settings, bus, health_fn=engine.health)
    if settings.get("privacy.store_questions"):
        engine.context.load(db.load_context())
        for q, a in db.recent_questions():
            win.add_history(q, a)
        if engine.services:
            engine.services.load_memory()
    win.set_debug_visible(bool(settings.get("debug.enabled")) and bool(settings.get("debug.show_trace")))
    _unhandled_sink[0] = lambda m, d: win.show_error(m, d)
    if previous_crash.happened:
        win.show_recovery("The app closed unexpectedly last time" +
                          (f" (during: {previous_crash.last_operation})." if previous_crash.last_operation else "."))
    engine.apply_interview_profile()

    def place():
        if not MonitorManager.has_second_monitor():
            win.show_error("Only one monitor detected — the assistant is on your only screen.")
        MonitorManager.place(win, int(settings.get("display.ai_monitor")), settings.get("display.window_mode"),
                             bool(settings.get("display.always_on_top")))
        win.set_off_monitor(False)

    def on_event(name, p):
        if name == "status":
            win.set_status(p["state"])
        elif name == "partial":
            win.set_live_partial(p["text"])
        elif name == "transcript":
            win.set_live_partial(p["text"])
            if settings.get("privacy.store_transcript"):
                with open(app_dir() / "transcript.txt", "a", encoding="utf-8") as f:
                    f.write(p["text"] + "\n")
        elif name == "detection":
            win.add_transcript(p["text"], p.get("type", ""), p.get("confidence", 0.0), p.get("action", ""))
        elif name == "pending":
            win.show_pending(p["text"], p["confidence"])
        elif name == "question":
            win.begin_question(p["raw"], p.get("confidence", 0.0))
        elif name == "clean_question":
            win.set_clean_question(p["text"])
        elif name == "answer_delta":
            win.append_answer(p["text"])
        elif name == "answer_done":
            win.finish_answer(p["question"], p["answer"], p["confidence"], p["seconds"], p["needs_clarification"],
                              p.get("grounding", ""), p.get("sources"), p.get("source_refs"), p.get("intent", ""))
        elif name == "answer_reset":
            win.reset_answer()
        elif name == "debug_trace":
            win.show_debug(p["text"])
        elif name == "restore_previous":
            win.restore_previous(p.get("clear", False))
        elif name == "history":
            win.add_history(p["question"], p["answer"])
        elif name == "info":
            win.toast("info", p["message"])
        elif name == "error":
            win.show_error(p["message"])

    bridge.event.connect(on_event)

    def on_settings(changed):                       # always delivered on the GUI thread
        if any(k.startswith("display.") for k in changed):
            win.apply_theme()
            win.apply_display_settings()
        if "display.click_through" in changed:
            win.set_click_through(bool(settings.get("display.click_through")))
        if any(k.startswith("debug.") for k in changed):
            configure_logging(logs, debug=bool(settings.get("debug.enabled")), level=settings.get("debug.log_level"))
            set_strict(bool(settings.get("debug.enabled")))
            win.set_debug_visible(bool(settings.get("debug.enabled")) and bool(settings.get("debug.show_trace")))
        if any(k.startswith(("display.ai_monitor", "display.window_mode", "display.always_on_top")) for k in changed):
            place()
        if any(k.startswith("shortcuts.") for k in changed):
            hotkeys.register(settings.section("shortcuts"))
        if any(k.startswith(("audio.system", "audio.mic", "speech.", "ai.provider", "ai.host", "ai.model")) for k in changed):
            engine.restart()
        if any(k.startswith(("ai.context", "ai.style")) for k in changed):
            win.style_combo.setCurrentText(str(settings.get("ai.style_preset")))

    settings.on_change(on_settings)

    # ------------------------------------------------------------------ dialogs and actions
    def open_settings(page: str = ""):
        SettingsWindow(settings, win, services=engine, db=db, initial_page=page).exec()

    def clear_history():
        win.clear_history()
        db.clear_questions()

    def export_bundle():
        from app.bundle import export_diagnostics_bundle
        dest, _ = QFileDialog.getSaveFileName(win, "Export diagnostics bundle", "diagnostics.zip", "Zip (*.zip)")
        if dest:
            names = export_diagnostics_bundle(dest, get_log_dir() or logs, bus, settings)
            win.toast("success", f"Diagnostics bundle saved ({len(names)} files, personal text redacted).")

    def open_logs():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_log_dir() or logs)))

    def run_test(which: str):
        if which != "llm":
            return open_settings("Diagnostics")
        run_task(win, lambda ctx: engine.test_llm(),
                 lambda r, e: win.toast("error" if e else ("success" if getattr(r, "ok", True) else "warning"),
                                        str(e) if e else r.format().splitlines()[0]),
                 stage="llm", title="Testing the language model")

    def practice():
        text, ok = QInputDialog.getText(win, "Practice mode", "Ask yourself a mock question (nothing is stored):")
        if ok and text.strip():
            engine.ask_text(text, practice=True)

    def feedback(text, is_q, qtype, conf, action):
        res = engine.mark_question(text, is_q, qtype, conf, action)
        if res.get("tuned") is not None:
            win.toast("info", f"Question threshold adjusted to {res['tuned']:.2f} from your corrections.")

    def rate(good, q, a):
        if engine.services:
            engine.services.feedback.rate_answer(q, a, 1 if good else -1)

    def pin(q, a):
        if engine.services:
            engine.services.feedback.pin(q, a)

    def why():
        win.set_debug_visible(True)
        win.toast("info", "The Debug panel shows the question type, retrieved sources and validation for the last answer.")

    def set_style(name):
        settings.set("ai.style_preset", name)

    def toggle_mini():
        settings.set("display.mini_mode", not bool(settings.get("display.mini_mode")))

    def toggle_lock():
        settings.set("display.click_through", not bool(settings.get("display.click_through")))

    def toggle_debug():
        settings.set("debug.enabled", not bool(settings.get("debug.enabled")))

    SHORTCUT_LIST = [("Ctrl+Shift+Space", "Start / pause listening"), ("Ctrl+Shift+A", "Answer the pending question"),
                     ("Ctrl+Shift+C", "Clear the answer"), ("Ctrl+Shift+X", "Clear context"),
                     ("Ctrl+Shift+M", "Move the window to the AI monitor"), ("Ctrl+Shift+U", "Toggle click-through lock"),
                     ("Ctrl+K", "Command palette"), ("Ctrl+L", "Activity log"), ("Ctrl+Enter", "Send the typed question"),
                     ("F1", "This list")]

    def show_shortcuts():
        ShortcutsOverlay(SHORTCUT_LIST, win).exec()

    def palette():
        cmds = [Command("Pause / resume listening", engine.toggle_pause, "Ctrl+Shift+Space", "pause"),
                Command("Answer now", engine.process_current, "Ctrl+Shift+A", "bolt"),
                Command("Clear answer", engine.clear_answer, "Ctrl+Shift+C", "x"),
                Command("Practice mode", practice, "", "target"),
                Command("Open activity log", win.drawer.toggle, "Ctrl+L", "activity"),
                Command("Toggle mini mode", toggle_mini, "", "minimize"),
                Command("Toggle debug panel", toggle_debug, "", "bug"),
                Command("Export diagnostics bundle", export_bundle, "", "download"),
                Command("Open logs folder", open_logs, "", "folder"),
                Command("Keyboard shortcuts", show_shortcuts, "F1", "keyboard"),
                Command("Settings", open_settings, "", "gear")]
        cmds += [Command(f"Settings: {p}", lambda p=p: open_settings(p), "", "sliders")
                 for p in ("Resume", "Knowledge Base", "LLM", "Audio", "Display & Theme", "Privacy", "Diagnostics")]
        CommandPalette(cmds, win).open_centered()

    win.pause_toggled.connect(engine.toggle_pause)
    win.process_requested.connect(engine.process_current)
    win.clear_requested.connect(engine.clear_answer)
    win.clear_context_requested.connect(engine.clear_context)
    win.clear_history_requested.connect(clear_history)
    win.settings_requested.connect(lambda: open_settings())
    win.page_requested.connect(open_settings)
    win.test_requested.connect(run_test)
    win.move_requested.connect(place)
    win.dismiss_pending.connect(lambda: setattr(engine, "_pending", None))
    win.ask_requested.connect(lambda t: engine.ask_text(t))
    win.practice_requested.connect(practice)
    win.regenerate_requested.connect(lambda d: engine.regenerate(d) or win.toast("info", "Nothing to regenerate yet."))
    win.feedback_requested.connect(feedback)
    win.answer_rated.connect(rate)
    win.pinned.connect(pin)
    win.style_changed.connect(set_style)
    win.why_requested.connect(why)
    win.open_logs_requested.connect(open_logs)
    win.export_requested.connect(export_bundle)
    win.palette_requested.connect(palette)
    win.mini_toggled.connect(lambda on: settings.set("display.mini_mode", bool(on)))
    win.click_through_toggled.connect(lambda on: settings.set("display.click_through", bool(on)))
    win.debug_toggled.connect(lambda on: None)
    QShortcut(QKeySequence("F1"), win, activated=show_shortcuts)

    hotkeys = HotkeyManager()
    actions = {"start_pause": engine.toggle_pause, "process": engine.process_current,
               "clear_answer": engine.clear_answer, "clear_context": engine.clear_context, "move_window": place,
               "toggle_lock": toggle_lock}
    hotkeys.triggered.connect(lambda a: actions[a]())
    hotkeys.register(settings.section("shortcuts"))

    # warn if the AI window gets dragged onto the meeting monitor
    watchdog = QTimer(interval=1500)
    watchdog.timeout.connect(lambda: win.set_off_monitor(
        not MonitorManager.is_on_monitor(win, int(settings.get("display.ai_monitor")))
        and MonitorManager.has_second_monitor()))
    watchdog.start()

    # ------------------------------------------------------------------ knowledge base: auto-sync + watcher
    def kb_report_toast(rep):
        n = rep.added + rep.updated
        if n or rep.removed:
            win.toast("warning" if rep.failed else "success",
                      f"Knowledge base updated: {rep.added} new, {rep.updated} changed, {rep.removed} removed.")

    if engine.services:
        svc = engine.services
        svc.watcher.on_report = lambda rep: call_on_gui(kb_report_toast, rep)
        if settings.get("knowledge.auto_sync") and svc.kb_folders():
            run_task(win, lambda ctx: svc.kb.sync_folders(svc.kb_folders(), progress_cb=ctx.progress_cb, cancel=ctx),
                     lambda rep, err: kb_report_toast(rep) if err is None else None, stage="kb", title="Checking knowledge folders")
        if settings.get("knowledge.watch"):
            svc.watcher.start()
        QTimer.singleShot(1000, lambda: svc.kb.purge_expired())

    place()
    win.show()
    engine.start()
    code = app.exec()
    engine.stop()
    if engine.services:
        engine.services.watcher.stop()
    hotkeys.unregister()
    crashguard.mark_clean_exit()
    return code


if __name__ == "__main__":
    sys.exit(main())
