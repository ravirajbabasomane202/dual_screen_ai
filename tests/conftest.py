import pytest
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _fresh_status_bus():
    """Every test gets its own StatusBus so windows leaked by earlier tests are not subscribed to this test's events."""
    from app.status import set_bus
    set_bus(None)
    yield
    set_bus(None)


@pytest.fixture(autouse=True)
def _close_leaked_widgets():
    """Windows / pages a test created must not keep timers, animations and bus subscriptions alive for the next test."""
    yield
    try:
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    for w in app.topLevelWidgets():
        try:
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
