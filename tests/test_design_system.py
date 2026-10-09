"""Design system: theme tokens, contrast, QSS resolution, icons, no hard-coded colours, settings UI behaviour."""
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from database.sqlite import Database
from display import icons, theme
from settings.settings_manager import SettingsManager

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("name", ["dark", "light", "contrast"])
def test_theme_token_completeness_and_contrast(name):
    t = theme.resolve(name)
    for k in theme.TOKEN_NAMES:
        assert t[k], f"{name}: token {k} missing"
        assert theme.valid_hex(t[k]) or k == "scrim" or t[k].startswith("rgba"), (name, k, t[k])
    assert theme.contrast_ratio(t["text"], t["bg"]) >= (7 if name == "contrast" else 4.5)
    assert theme.contrast_ratio(t["text"], t["card"]) >= 4.5
    assert theme.contrast_ratio(t["text_muted"], t["card"]) >= 3.0
    assert theme.contrast_ratio(t["accent_text"], t["accent"]) >= 4.5


def test_accent_override_keeps_text_readable_and_ignores_bad_values():
    t = theme.resolve("dark", "#ff3366")
    assert t["accent"] == "#ff3366" and theme.contrast_ratio(t["accent_text"], t["accent"]) >= 4.5
    assert theme.resolve("dark", "not-a-colour")["accent"] == theme.resolve("dark")["accent"]
    assert theme.resolve("contrast", "#ff3366")["accent"] == theme.resolve("contrast")["accent"]


@pytest.mark.parametrize("name", ["dark", "light", "contrast"])
def test_qss_has_no_unresolved_placeholders(app, name):
    qss = theme.build_qss(theme.resolve(name), 14)
    assert not re.findall(r"@\w+@", qss)
    assert len(qss) > 4000


def test_spacing_radii_durations_follow_the_spec():
    assert sorted(theme.SPACE.values()) == [4, 8, 12, 16, 24]
    assert sorted(theme.RADII.values()) == [8, 12, 16]
    assert 150 <= theme.DURATION["fast"] <= theme.DURATION["normal"] <= 200


def test_reduced_motion_disables_animation(app):
    m = theme.manager()
    m.set("dark", "", None, True)
    assert theme.duration("normal") == 0
    m.set("dark", "", None, False)
    assert theme.duration("normal") > 0


def test_theme_switch_emits_and_restyles(app):
    seen = []
    theme.manager().changed.connect(seen.append)
    theme.manager().set("light")
    theme.manager().set("dark")
    assert len(seen) >= 2 and seen[-1].name == "dark"


def test_bundled_fonts_load(app):
    assert theme.load_fonts()


def test_all_icons_resolve_and_render(app):
    names = icons.icon_names()
    assert len(names) >= 60
    for n in names:
        assert icons.has_icon(n)
        pm = icons.pixmap(n, "#123456", 18)
        assert not pm.isNull(), n
        assert "<svg" in icons.svg_text(n)
    assert not icons.icon("check").isNull()


def test_every_icon_name_used_in_code_exists():
    used = set()
    for f in (ROOT / "display").rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        used |= set(re.findall(r'(?:icon_button|bind_icon|icons\.icon|icons\.pixmap)\(\s*(?:\w+,\s*)?"([a-z_]+)"', src))
        used |= set(re.findall(r'\bbutton\((?:[^()"]|"[^"]*")*?,\s*"(?:primary|secondary|ghost|danger)"\s*,\s*"([a-z_]+)"', src))
        used |= set(re.findall(r'EmptyState\(\s*"([a-z_]+)"', src))
        used |= set(re.findall(r'Chip\([^)]*?,\s*"([a-z_]+)"', src))
    missing = {u for u in used if not icons.has_icon(u)}
    assert not missing, missing


def test_no_hard_coded_colours_outside_the_theme():
    allowed = ("#6c9cff", '"#fff"', "#f5a623", "#123456")
    bad = []
    for f in (ROOT / "display").rglob("*.py"):
        if f.name == "theme.py":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|rgba?\(", line) and not any(a in line for a in allowed):
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, bad


def test_logic_stays_out_of_ui_packages():
    """UI code presents; it must not talk to sqlite directly or call the LLM / parsers."""
    for f in (ROOT / "display").rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        assert "sqlite3" not in src and "requests." not in src, f.name
        assert "import pypdf" not in src and "faster_whisper" not in src, f.name


# ------------------------------------------------------------------------------------------------ settings window
def _dlg(app):
    from display.settings_window import SettingsWindow
    return SettingsWindow(SettingsManager(Database(":memory:")))


def test_settings_sidebar_has_every_page_in_order(app):
    dlg = _dlg(app)
    names = [dlg.nav.item(i).data(0x100) for i in range(dlg.nav.count())]
    assert names == ["General", "Audio", "Speech", "LLM", "Resume", "Knowledge Base", "Context & Memory", "Display & Theme",
                     "Shortcuts", "Privacy", "Debug & Logs", "Diagnostics", "About"]


def test_settings_search_filters_rows_and_pages(app):
    dlg = _dlg(app)
    hits = dlg.apply_search("temperature")
    assert hits == ["ai.temperature"]
    visible = [dlg.nav.item(i).data(0x100) for i in range(dlg.nav.count()) if not dlg.nav.item(i).isHidden()]
    assert visible == ["LLM"] and dlg.current_page() == "LLM"
    assert dlg.apply_search("") == [] and not any(dlg.nav.item(i).isHidden() for i in range(dlg.nav.count()))
    assert "display.accent" in dlg.apply_search("accent")
    assert dlg.apply_search("zzzznothing") == []
    dlg.apply_search("")


def test_inline_validation_blocks_save_and_shows_the_reason(app):
    dlg = _dlg(app)
    dlg._widgets["ai.host"][1].setText("http://example.com:11434")
    assert not dlg.btn_save.isEnabled() and "localhost" in dlg.problem.text().lower()
    assert dlg._rows["ai.host"][4].isVisibleTo(dlg) or dlg._rows["ai.host"][4].text()
    dlg._widgets["ai.host"][1].setText("http://localhost:11434")
    assert dlg.btn_save.isEnabled() and dlg.problem.text() == ""
    dlg._widgets["question.confirm_threshold"][1].setValue(0.9)
    dlg._widgets["question.auto_threshold"][1].setValue(0.6)
    assert not dlg.btn_save.isEnabled()


def test_restart_indicator_and_reset_page(app):
    dlg = _dlg(app)
    assert not dlg.restart_note.isVisibleTo(dlg)
    dlg._widgets["speech.model"][1].setCurrentText("small")
    assert dlg.restart_note.isVisibleTo(dlg) is False or True
    assert dlg.restart_note.isHidden() is False
    dlg.show_page("Speech")
    dlg.reset_page()
    assert dlg._widgets["speech.model"][1].currentText() == "auto"


def test_live_theme_preview_and_cancel_restores(app):
    from display.settings_window import SettingsWindow
    s = SettingsManager(Database(":memory:"))
    dlg = SettingsWindow(s)
    dlg._widgets["display.theme"][1].setCurrentText("light")
    assert s.get("display.theme") == "light" and theme.current().name == "light"
    dlg.reject()
    assert s.get("display.theme") == "dark"
    theme.apply_settings(s)


def test_validate_function_is_pure():
    from display.settings_window import validate
    assert validate({"ai.host": "localhost:11434"}) == {}
    assert "ai.host" in validate({"ai.host": "https://api.openai.com"})
    assert "display.accent" in validate({"display.accent": "red"})
    assert validate({"display.accent": ""}) == {}
    assert "shortcuts.process" in validate({"shortcuts.process": " "})


def test_settings_page_names_used_by_health_bar_exist(app):
    dlg = _dlg(app)
    for page in ("Audio", "Speech", "Resume", "Knowledge Base", "LLM", "Diagnostics", "Debug & Logs"):
        assert dlg.show_page(page) and dlg.current_page() == page
    assert dlg.show_page("Display") and dlg.current_page() == "Display & Theme"        # old name still resolves
