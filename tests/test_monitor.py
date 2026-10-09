import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QWidget
from display.monitor_manager import MonitorManager


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_lists_monitors_in_stable_order(app):
    mons = MonitorManager.list_monitors()
    assert mons and [m.index for m in mons] == list(range(len(mons)))
    assert all(m.width > 0 and m.height > 0 for m in mons)


def test_out_of_range_monitor_falls_back_without_crashing(app):
    assert MonitorManager.screen_for(7) is not None
    assert MonitorManager.screen_for(-1) is not None


@pytest.mark.parametrize("mode", ["windowed", "borderless", "fullscreen"])
def test_place_window_on_configured_monitor(app, mode):
    w = QWidget()
    screen = MonitorManager.place(w, 1, mode, always_on_top=True)
    geo = screen.geometry()
    assert geo.contains(w.frameGeometry().center())
    assert MonitorManager.is_on_monitor(w, MonitorManager.list_monitors().index(
        next(m for m in MonitorManager.list_monitors() if m.x == geo.x() and m.y == geo.y())))
    w.close()
