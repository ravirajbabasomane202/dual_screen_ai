import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from database.sqlite import Database
from display.ai_window import AIWindow
from display.settings_window import SettingsWindow
from settings.settings_manager import SettingsManager


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_ai_window_streaming_flow(app):
    s = SettingsManager(Database(":memory:"))
    w = AIWindow(s)
    for state in ("loading", "listening", "transcribing", "question_detected", "cleaning", "generating", "ready", "paused", "error"):
        w.set_status(state)
    w.show_pending("Is this overfitting?", 0.7)
    w.begin_question("raw words")
    w.set_clean_question("What is overfitting?")
    for part in ("Over", "fitting is ", "memorising noise."):
        w.append_answer(part)
    assert "memorising noise" in w.answer.toPlainText()
    w.finish_answer("What is overfitting?", "Memorising noise.", 0.94, 1.7, False)
    assert "94%" in w.footer.text() and "1.7 sec" in w.footer.text()
    w.add_history("What is overfitting?", "Memorising noise.")
    w.restore_previous(clear=False)
    assert w.question.text() == "What is overfitting?"
    w.set_off_monitor(True)


def test_settings_window_roundtrip(app):
    s = SettingsManager(Database(":memory:"))
    dlg = SettingsWindow(s)
    kind, w = dlg._widgets["ai.temperature"]
    w.setValue(0.5)
    dlg._widgets["privacy.store_audio"][1].setChecked(False)
    dlg._save()
    assert s.get("ai.temperature") == 0.5 and s.get("privacy.store_audio") is False
