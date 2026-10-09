"""Shared plumbing of the action pages: error banner, progress row with a real Cancel, toasts, background tasks.
Pages only present; every operation lives in resume/, knowledge/ or app/."""
from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.gui_thread import assert_gui_thread
from app.tasks import Task, run_task
from display.widgets.common import label
from display.widgets.feedback import Banner, ProgressRow, ToastManager


class ActionPage(QWidget):
    """A page with a banner (errors), a progress row (long operations) and a toast area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.banner = Banner()
        self.banner.hide()
        self.progress_row = ProgressRow()
        self.progress_row.cancel_requested.connect(self._cancel_current)
        self._task: Task | None = None
        self._toasts: ToastManager | None = None
        self.status = label("", "caption", wrap=True)

    # ---- feedback
    def toast(self, kind: str, message: str, action_text: str = "", on_action=None) -> None:
        assert_gui_thread("ActionPage.toast")
        host = self.window() or self
        mgr = getattr(host, "toasts", None)
        if mgr is None:
            if self._toasts is None:
                self._toasts = ToastManager(self)
            mgr = self._toasts
        mgr.show(kind, message, action_text, on_action)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def fail(self, err) -> None:
        assert_gui_thread("ActionPage.fail")
        info = self.banner.show_error(err)
        self.status.setText(info.cause)

    # ---- tasks
    def busy(self) -> bool:
        return self._task is not None and not self._task.finished

    def start(self, fn, on_done, title: str, stage: str, cancellable: bool = True) -> Task | None:
        """Run fn(ctx) off the GUI thread. on_done(value, error) runs on the GUI thread; Cancel is cooperative."""
        assert_gui_thread("ActionPage.start")
        if self.busy():
            self.toast("warn", "Another operation is still running — wait for it or press Cancel.")
            return None
        self.banner.hide()
        self.progress_row.start(title, cancellable)
        self.set_busy_state(True)

        def done(value, err):
            self._task = None
            self.set_busy_state(False)
            cancelled = bool(err) and getattr(err, "cancelled", False)
            self.progress_row.finish(err is None, "Cancelled" if cancelled else ("Done" if err is None else "Failed"))
            if err is not None and not cancelled:
                self.fail(err)
            try:
                on_done(value, err)
            finally:
                self.after_task()

        self._task = run_task(self, fn, done, lambda f, m: self.progress_row.update_progress(f, m), stage=stage, title=title)
        return self._task

    def _cancel_current(self) -> None:
        if self._task is not None:
            self._task.cancel()

    # hooks for subclasses
    def set_busy_state(self, busy: bool) -> None:
        pass

    def after_task(self) -> None:
        pass

    def layout_head(self, lay: QVBoxLayout) -> None:
        lay.addWidget(self.banner)
        lay.addWidget(self.progress_row)
