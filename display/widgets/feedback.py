"""Feedback surfaces: inline error banner (cause · action · Details), non-blocking toasts, progress row with a real
Cancel button, skeleton loaders and empty states."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from app.errorinfo import ErrorInfo, describe
from app.gui_thread import assert_gui_thread
from display import icons, theme
from display.widgets.common import bind_icon, button, icon_button, label, set_prop

KIND_ICON = {"error": "error", "warning": "warning", "info": "info", "success": "success"}
KIND_TOKEN = {"error": "danger", "warning": "warning", "info": "info", "success": "success"}


class Banner(QFrame):
    """Inline banner. Errors are never modal-and-lost: plain-English cause, recommended action, expandable details."""
    action_clicked = Signal(str)          # the key passed to set_action
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("banner")
        self.setProperty("kind", "error")
        self._action_key = ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 8, 10)
        outer.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.icon = QLabel()
        self.icon.setFixedSize(20, 20)
        row.addWidget(self.icon, 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.cause = label("", "section", wrap=True)
        self.action = label("", "muted", wrap=True)
        text.addWidget(self.cause)
        text.addWidget(self.action)
        row.addLayout(text, 1)
        self.btn_action = button("", "primary")
        self.btn_action.clicked.connect(lambda: self.action_clicked.emit(self._action_key))
        self.btn_details = button("Details", "ghost", "chevron_down")
        self.btn_details.setCheckable(True)
        self.btn_details.toggled.connect(self._toggle)
        self.btn_close = icon_button("x", "Dismiss", self.dismiss)
        for w in (self.btn_action, self.btn_details, self.btn_close):
            row.addWidget(w, 0, Qt.AlignTop)
        outer.addLayout(row)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setProperty("mono", True)
        self.details.setMaximumHeight(160)
        self.details.hide()
        outer.addWidget(self.details)
        self.btn_action.hide()
        self.hide()
        self._timer = QTimer(self, singleShot=True, timeout=self.dismiss)
        theme.on_changed(self, self._paint)

    def _paint(self) -> None:
        kind = self.property("kind")
        self.icon.setPixmap(icons.pixmap(KIND_ICON.get(kind, "info"), theme.current()[KIND_TOKEN.get(kind, "info")], 20))

    def _toggle(self, on: bool) -> None:
        self.details.setVisible(on)
        self.btn_details.setText("Hide details" if on else "Details")

    def set_action(self, text: str, key: str = "") -> None:
        self._action_key = key
        self.btn_action.setText(text)
        self.btn_action.setVisible(bool(text))

    def show_message(self, kind: str, cause: str, action: str = "", details: str = "", timeout_ms: int = 0) -> None:
        assert_gui_thread("Banner.show_message")
        set_prop(self, "kind", kind)
        self.cause.setText(cause)
        self.action.setText(action)
        self.action.setVisible(bool(action))
        self.details.setPlainText(details)
        self.btn_details.setVisible(bool(details))
        self.btn_details.setChecked(False)
        self.details.hide()
        self._paint()
        self.setAccessibleName(f"{kind}: {cause}")
        self.show()
        self._timer.stop()
        if timeout_ms:
            self._timer.start(timeout_ms)

    def show_error(self, err, action_button: str = "", key: str = "") -> ErrorInfo:
        info = err if isinstance(err, ErrorInfo) else describe(err)
        self.set_action(action_button, key or info.page)
        self.show_message("error", info.cause, info.action, info.details)
        return info

    def dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


class Toast(QFrame):
    closed = Signal(object)

    def __init__(self, kind: str, message: str, action_text: str = "", on_action=None, timeout_ms: int = 5000, parent=None):
        super().__init__(parent)
        self.setObjectName("toast")
        self.setProperty("kind", kind)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 8, 10)
        lay.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(KIND_ICON.get(kind, "info"), theme.current()[KIND_TOKEN.get(kind, "info")], 18))
        lay.addWidget(ic, 0, Qt.AlignTop)
        self.msg = label(message, "", wrap=True)
        self.msg.setMinimumWidth(220)
        self.msg.setMaximumWidth(380)
        lay.addWidget(self.msg, 1)
        if action_text:
            b = button(action_text, "ghost")
            b.clicked.connect(lambda: (on_action() if on_action else None, self.close_toast()))
            lay.addWidget(b)
        lay.addWidget(icon_button("x", "Dismiss", self.close_toast, size=14))
        self.setAccessibleName(f"{kind}: {message}")
        self._timeout = timeout_ms
        self._done = False

    def start(self) -> None:
        if self._timeout:
            QTimer.singleShot(self._timeout, self.close_toast)

    def close_toast(self) -> None:
        if self._done:
            return
        self._done = True
        self.closed.emit(self)


class ToastManager:
    """Stacks toasts in the bottom-right corner of a host window. Never blocks, never steals focus."""

    def __init__(self, host: QWidget):
        self.host = host
        self.toasts: list[Toast] = []
        self.history: list[tuple[str, str]] = []

    def show(self, kind: str, message: str, action_text: str = "", on_action=None, timeout_ms: int = 5000) -> Toast:
        assert_gui_thread("ToastManager.show")
        self.history.append((kind, message))
        t = Toast(kind, message, action_text, on_action, 0 if kind == "error" and not timeout_ms else timeout_ms, self.host)
        t.closed.connect(self._remove)
        t.show()
        t.raise_()
        self.toasts.append(t)
        self._layout()
        if theme.duration("normal"):
            eff = QGraphicsOpacityEffect(t)
            t.setGraphicsEffect(eff)
            a = QPropertyAnimation(eff, b"opacity", t)
            a.setDuration(theme.duration("normal"))
            a.setStartValue(0.0)
            a.setEndValue(1.0)
            a.finished.connect(lambda: t.setGraphicsEffect(None))
            a.start(QPropertyAnimation.DeleteWhenStopped)
        t.start()
        while len(self.toasts) > 4:
            self._remove(self.toasts[0])
        return t

    def _remove(self, t: Toast) -> None:
        if t in self.toasts:
            self.toasts.remove(t)
            t.hide()
            t.deleteLater()
            self._layout()

    def _layout(self) -> None:
        m, gap = 16, 8
        y = self.host.height() - m
        for t in reversed(self.toasts):
            t.adjustSize()
            y -= t.height()
            t.move(self.host.width() - t.width() - m, y)
            y -= gap
            t.raise_()

    def relayout(self) -> None:
        self._layout()


class ProgressRow(QFrame):
    """Title · message · progress bar (indeterminate when no fraction is known) · Cancel."""
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("card", "flat")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.title = label("", "section")
        self.msg = label("", "caption")
        self.btn_cancel = button("Cancel", "ghost", "x")
        self.btn_cancel.clicked.connect(self._cancel)
        top.addWidget(self.title)
        top.addWidget(self.msg, 1)
        top.addWidget(self.btn_cancel)
        lay.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.hide()

    def _cancel(self) -> None:
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Cancelling…")
        self.cancel_requested.emit()

    def start(self, title: str, cancellable: bool = True) -> None:
        assert_gui_thread("ProgressRow.start")
        self.title.setText(title)
        self.msg.setText("")
        self.bar.setRange(0, 0)               # indeterminate until the first fraction arrives
        set_prop(self.bar, "state", "")
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("Cancel")
        self.btn_cancel.setVisible(cancellable)
        self.show()

    def update_progress(self, fraction: float | None, message: str = "") -> None:
        assert_gui_thread("ProgressRow.update_progress")
        if fraction is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        if message:
            self.msg.setText(message)

    def finish(self, ok: bool = True, message: str = "", hide_after_ms: int = 1800) -> None:
        assert_gui_thread("ProgressRow.finish")
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if ok else 0)
        set_prop(self.bar, "state", "ok" if ok else "error")
        self.msg.setText(message)
        self.btn_cancel.hide()
        if hide_after_ms:
            QTimer.singleShot(hide_after_ms, self.hide)

    def is_active(self) -> bool:
        return self.isVisible() and self.btn_cancel.isVisible()


class Skeleton(QFrame):
    """Placeholder block with a gentle shimmer (static with reduced motion)."""

    def __init__(self, height: int = 14, width: int = 0, parent=None):
        super().__init__(parent)
        self.setObjectName("skeleton")
        self.setFixedHeight(height)
        if width:
            self.setFixedWidth(width)
        self._eff = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._eff)
        self._anim = QPropertyAnimation(self._eff, b"opacity", self)
        self._anim.setDuration(900)
        self._anim.setStartValue(0.45)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)
        self._anim.setLoopCount(-1)

    def showEvent(self, e) -> None:
        if not theme.reduced_motion():
            self._anim.start()
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self._anim.stop()
        super().hideEvent(e)


class SkeletonBlock(QWidget):
    def __init__(self, lines: int = 3, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        for i in range(lines):
            lay.addWidget(Skeleton(14, 0 if i < lines - 1 else 180))


class EmptyState(QWidget):
    """Illustration-style icon + one sentence + one clear next action."""
    action = Signal()

    def __init__(self, icon: str, title: str, text: str = "", action_text: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(10)
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignCenter)
        self._icon_name = icon
        self.icon.setFixedSize(72, 72)
        self.icon.setStyleSheet("")
        lay.addWidget(self.icon, 0, Qt.AlignHCenter)
        self.title = label(title, "section")
        self.title.setAlignment(Qt.AlignCenter)
        self.text = label(text, "muted", wrap=True)
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setMaximumWidth(420)
        lay.addWidget(self.title)
        lay.addWidget(self.text, 0, Qt.AlignHCenter)
        self.btn = button(action_text, "primary")
        self.btn.clicked.connect(self.action)
        self.btn.setVisible(bool(action_text))
        lay.addWidget(self.btn, 0, Qt.AlignHCenter)
        self._paint()
        theme.on_changed(self, self._paint)

    def _paint(self) -> None:
        t = theme.current()
        self.icon.setPixmap(icons.pixmap(self._icon_name, t["accent"], 40))
        self.icon.setStyleSheet(f"background:{t['accent_soft']}; border-radius:36px;")

    def set_content(self, icon: str | None = None, title: str | None = None, text: str | None = None,
                    action_text: str | None = None) -> None:
        if icon:
            self._icon_name = icon
            self._paint()
        if title is not None:
            self.title.setText(title)
        if text is not None:
            self.text.setText(text)
        if action_text is not None:
            self.btn.setText(action_text)
            self.btn.setVisible(bool(action_text))


def copy_text(text: str) -> None:
    QGuiApplication.clipboard().setText(text)
