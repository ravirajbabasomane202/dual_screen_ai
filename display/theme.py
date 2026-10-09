"""Design system: tokens, three themes (Dark default, Light, High-contrast), accent colour, fonts and one global
stylesheet. No other module hard-codes a colour; widgets ask `current()[token]` or are styled by object name /
dynamic property from the stylesheet built here.

Tokens
    surfaces   bg · surface · card · card_hover
    lines      border · border_strong · focus
    text       text · text_muted · text_faint
    accent     accent · accent_hover · accent_pressed · accent_text · accent_soft · selection
    status     success · warning · danger · info (+ *_soft backgrounds)
    misc       scrim · code_bg · shadow
Scales
    SPACE (4, 8, 12, 16, 24) · RADII (8, 12, 16) · FONT sizes derive from the body size · DURATION fast/normal (ms)
"""
from __future__ import annotations

import logging
import re

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QFontDatabase, QPalette

from app.paths import resource_dir

log = logging.getLogger("app.theme")

SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}
RADII = {"sm": 8, "md": 12, "lg": 16}
DURATION = {"fast": 150, "normal": 200}
UI_FONTS = ('"Inter", "Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif')
MONO_FONTS = ('"DejaVu Sans Mono", "Cascadia Mono", "Consolas", "Menlo", monospace')

TOKEN_NAMES = (
    "bg", "surface", "card", "card_hover", "border", "border_strong", "focus",
    "text", "text_muted", "text_faint",
    "accent", "accent_hover", "accent_pressed", "accent_text", "accent_soft", "selection",
    "success", "warning", "danger", "info", "success_soft", "warning_soft", "danger_soft", "info_soft",
    "scrim", "code_bg", "shadow",
)

THEME_LABELS = {"dark": "Dark", "light": "Light", "contrast": "High contrast"}


# ------------------------------------------------------------------------------------------------ colour helpers
def _rgb(c: str) -> tuple[int, int, int]:
    q = QColor(c)
    return q.red(), q.green(), q.blue()


def mix(a: str, b: str, t: float) -> str:
    """Blend colour a towards b (t=0 -> a, t=1 -> b)."""
    (r1, g1, b1), (r2, g2, b2) = _rgb(a), _rgb(b)
    return "#%02x%02x%02x" % (round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t))


def luminance(c: str) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(c)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def on_color(bg: str) -> str:
    """Readable text colour for a filled background."""
    return "#0b0f19" if contrast_ratio(bg, "#0b0f19") >= contrast_ratio(bg, "#ffffff") else "#ffffff"


def valid_hex(c: str) -> bool:
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", c or ""))


# ------------------------------------------------------------------------------------------------ themes
_BASE = {
    "dark": dict(
        bg="#0e1116", surface="#141820", card="#1a1f29", card_hover="#222937", border="#2a3140", border_strong="#3b4458",
        text="#eaeef5", text_muted="#9ba6b9", text_faint="#68758c", accent="#6c9cff", accent_hover="#88b0ff",
        accent_pressed="#5586ee", accent_text="#0b1020", success="#3ddc97", warning="#f5b84a", danger="#ff6b6b",
        info="#6cb6ff", scrim="rgba(4, 7, 12, 0.62)", code_bg="#10141b", shadow="#000000"),
    "light": dict(
        bg="#f3f5f9", surface="#ffffff", card="#ffffff", card_hover="#eff2f8", border="#dfe4ee", border_strong="#c3cbdb",
        text="#151a23", text_muted="#546077", text_faint="#8792a8", accent="#2f5fe3", accent_hover="#4a74ee",
        accent_pressed="#2349b8", accent_text="#ffffff", success="#12855a", warning="#a8660f", danger="#c93232",
        info="#1d68c4", scrim="rgba(15, 20, 32, 0.38)", code_bg="#eef1f7", shadow="#1b2540"),
    "contrast": dict(
        bg="#000000", surface="#050505", card="#0b0b0b", card_hover="#1b1b1b", border="#ffffff", border_strong="#ffffff",
        text="#ffffff", text_muted="#ececec", text_faint="#c4c4c4", accent="#ffd60a", accent_hover="#ffe45c",
        accent_pressed="#e2b800", accent_text="#000000", success="#62ff94", warning="#ffb020", danger="#ff7070",
        info="#7ad0ff", scrim="rgba(0, 0, 0, 0.8)", code_bg="#000000", shadow="#000000"),
}


class Theme:
    def __init__(self, name: str, tokens: dict[str, str]):
        self.name, self.tokens = name, tokens

    def __getitem__(self, k: str) -> str:
        return self.tokens[k]

    def get(self, k: str, default=None):
        return self.tokens.get(k, default)

    @property
    def dark(self) -> bool:
        return luminance(self.tokens["bg"]) < 0.4


def _finish(name: str, t: dict) -> dict:
    t = dict(t)
    t.setdefault("focus", t["accent"] if name != "contrast" else "#ffffff")
    t["accent_soft"] = mix(t["card"], t["accent"], 0.20 if name != "contrast" else 0.0)
    t["selection"] = mix(t["card"], t["accent"], 0.34 if name != "contrast" else 0.0)
    for k in ("success", "warning", "danger", "info"):
        t[f"{k}_soft"] = mix(t["card"], t[k], 0.16 if name != "contrast" else 0.0)
    if name == "contrast":
        t["accent_soft"] = t["selection"] = "#1f1f00"
        for k in ("success", "warning", "danger", "info"):
            t[f"{k}_soft"] = "#101010"
    return t


def resolve(name: str = "dark", accent: str = "") -> Theme:
    name = name if name in _BASE else "dark"
    t = dict(_BASE[name])
    if valid_hex(accent) and name != "contrast":
        t["accent"] = accent.lower()
        t["accent_hover"] = mix(accent, "#ffffff", 0.18)
        t["accent_pressed"] = mix(accent, "#000000", 0.18)
        t["accent_text"] = on_color(accent)
        t["focus"] = accent.lower()
    return Theme(name, _finish(name, t))


THEMES: dict[str, Theme] = {n: resolve(n) for n in _BASE}


def theme_names() -> list[str]:
    return list(_BASE)


# ------------------------------------------------------------------------------------------------ fonts
_fonts_loaded = False
UI_FAMILY = "Inter"


def load_fonts() -> bool:
    """Register the bundled fonts (Inter, DejaVu Sans Mono). Safe to call repeatedly; falls back to system fonts."""
    global _fonts_loaded
    if _fonts_loaded:
        return True
    d = resource_dir() / "display" / "assets" / "fonts"
    ok = 0
    for f in sorted(d.glob("*.[ot]tf")):
        if QFontDatabase.addApplicationFont(str(f)) >= 0:
            ok += 1
        else:
            log.warning("could not load font %s", f.name)
    _fonts_loaded = ok > 0
    return _fonts_loaded


# ------------------------------------------------------------------------------------------------ manager
class ThemeManager(QObject):
    changed = Signal(object)                    # Theme

    def __init__(self):
        super().__init__()
        self.theme = THEMES["dark"]
        self.body_px = 14
        self.reduced_motion = False

    def set(self, name: str, accent: str = "", body_px: int | None = None, reduced_motion: bool | None = None) -> Theme:
        new = resolve(name, accent)
        if body_px:
            body_px = int(max(11, min(24, body_px)))
        key = (new.name, new.tokens["accent"], body_px or self.body_px, bool(self.reduced_motion if reduced_motion is None else reduced_motion))
        if key == getattr(self, "_key", None):
            return self.theme                      # nothing changed: don't restyle the whole application again
        self._key = key
        if body_px:
            self.body_px = body_px
        if reduced_motion is not None:
            self.reduced_motion = bool(reduced_motion)
        self.theme = new
        self.apply()
        self.changed.emit(new)
        return new

    def apply(self) -> None:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return
        load_fonts()
        app.setStyle("Fusion")
        app.setPalette(build_palette(self.theme))
        app.setStyleSheet(build_qss(self.theme, self.body_px))


_manager: ThemeManager | None = None


def manager() -> ThemeManager:
    global _manager
    if _manager is None:
        _manager = ThemeManager()
    return _manager


def on_changed(owner, fn) -> None:
    """Call fn() whenever the theme changes, for as long as `owner` (a widget) is alive. Lambdas connected directly to a
    signal are NOT disconnected when their widget is deleted (closed Settings dialogs, tests) and would raise
    'Internal C++ object already deleted' on the next theme change."""
    try:
        import shiboken6
        valid = shiboken6.isValid
    except Exception:                                  # pragma: no cover
        valid = lambda o: True
    mgr = manager()

    def slot(_t=None):
        if not valid(owner):
            try:
                mgr.changed.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
            return
        fn()
    mgr.changed.connect(slot)


def current() -> Theme:
    return manager().theme


def reduced_motion() -> bool:
    return manager().reduced_motion


def duration(kind: str = "normal") -> int:
    return 0 if reduced_motion() else DURATION[kind]


def apply_settings(settings) -> Theme:
    """Apply display.theme / display.accent / display.font_size / display.reduced_motion from the settings."""
    body = max(12, int(settings.get("display.font_size", 18)) - 4)
    return manager().set(str(settings.get("display.theme", "dark")), str(settings.get("display.accent", "") or ""), body,
                         bool(settings.get("display.reduced_motion", False)))


# ------------------------------------------------------------------------------------------------ palette + qss
def build_palette(t: Theme) -> QPalette:
    pal = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.Base, "surface"), (QPalette.AlternateBase, "card"),
                      (QPalette.Button, "card"), (QPalette.WindowText, "text"), (QPalette.Text, "text"),
                      (QPalette.ButtonText, "text"), (QPalette.ToolTipText, "text"), (QPalette.ToolTipBase, "card"),
                      (QPalette.Highlight, "accent"), (QPalette.Link, "accent"), (QPalette.PlaceholderText, "text_faint"),
                      (QPalette.HighlightedText, "accent_text")):
        pal.setColor(role, QColor(t[key]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(t["text_faint"]))
    return pal


_QSS = """
* { font-family: @ui@; font-size: @body@px; outline: 0; }
QWidget { color: @text@; }
QMainWindow, QDialog, QWidget#window, QWidget#page { background: @bg@; }
QWidget:disabled { color: @text_faint@; }
QToolTip { background: @card@; color: @text@; border: 1px solid @border_strong@; border-radius: 6px; padding: 6px 8px; }

QLabel { background: transparent; }
QLabel[role="display"] { font-size: @display@px; font-weight: 700; }
QLabel[role="title"]   { font-size: @title@px; font-weight: 700; }
QLabel[role="section"] { font-size: @section@px; font-weight: 600; }
QLabel[role="caption"] { font-size: @caption@px; color: @text_muted@; }
QLabel[role="muted"]   { color: @text_muted@; }
QLabel[role="overline"]{ font-size: @caption@px; color: @text_faint@; font-weight: 600; letter-spacing: 1.4px; }
QLabel[role="mono"]    { font-family: @mono@; font-size: @caption@px; color: @text_muted@; }
QLabel[role="success"] { color: @success@; } QLabel[role="warning"] { color: @warning@; } QLabel[role="danger"] { color: @danger@; }

QFrame#card, QFrame[card="true"] { background: @card@; border: 1px solid @border@; border-radius: @r_md@px; }
QFrame#card[elevated="true"] { border: 1px solid @border_strong@; }
QFrame[card="flat"] { background: @surface@; border: 1px solid @border@; border-radius: @r_sm@px; }
QFrame#divider { background: @border@; max-height: 1px; min-height: 1px; border: none; }
QFrame#vdivider { background: @border@; max-width: 1px; min-width: 1px; border: none; }

/* buttons ------------------------------------------------------------------------------------------ */
QPushButton, QToolButton { background: @card@; color: @text@; border: 1px solid @border_strong@; border-radius: @r_sm@px;
    padding: 7px 14px; font-weight: 500; }
QPushButton:hover, QToolButton:hover { background: @card_hover@; border-color: @accent@; }
QPushButton:pressed, QToolButton:pressed { background: @accent_soft@; }
QPushButton:focus, QToolButton:focus { border: 2px solid @focus@; padding: 6px 13px; }
QPushButton:disabled, QToolButton:disabled { background: @surface@; color: @text_faint@; border-color: @border@; }
QPushButton[variant="primary"] { background: @accent@; color: @accent_text@; border: 1px solid @accent@; font-weight: 600; }
QPushButton[variant="primary"]:hover { background: @accent_hover@; border-color: @accent_hover@; }
QPushButton[variant="primary"]:pressed { background: @accent_pressed@; }
QPushButton[variant="primary"]:disabled { background: @border@; color: @text_faint@; border-color: @border@; }
QPushButton[variant="ghost"], QToolButton[variant="ghost"] { background: transparent; border: 1px solid transparent; color: @text_muted@; }
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover { background: @card_hover@; color: @text@; border-color: transparent; }
QPushButton[variant="ghost"]:checked, QToolButton[variant="ghost"]:checked { background: @accent_soft@; color: @accent@; }
QPushButton[variant="danger"] { background: transparent; color: @danger@; border: 1px solid @danger@; }
QPushButton[variant="danger"]:hover { background: @danger_soft@; }
QPushButton[variant="danger-fill"] { background: @danger@; color: #ffffff; border: 1px solid @danger@; font-weight: 600; }
QPushButton:checked, QToolButton:checked { background: @accent_soft@; border-color: @accent@; color: @text@; }
QToolButton[icononly="true"] { padding: 6px; min-width: 18px; min-height: 18px; }
QPushButton[chip="true"] { border-radius: 14px; padding: 4px 12px; background: @accent_soft@; border: 1px solid transparent; color: @text@; }
QPushButton[chip="true"]:hover { border-color: @accent@; }

/* inputs ------------------------------------------------------------------------------------------- */
QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {
    background: @surface@; color: @text@; border: 1px solid @border_strong@; border-radius: @r_sm@px; padding: 6px 10px;
    selection-background-color: @accent@; selection-color: @accent_text@; }
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 2px solid @focus@; padding: 5px 9px; }
QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled { background: @bg@; color: @text_faint@; }
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"] { border: 2px solid @danger@; padding: 5px 9px; }
QTextBrowser { background: @card@; border: none; padding: 4px; }
QPlainTextEdit[mono="true"], QTextBrowser[mono="true"], QTextEdit[mono="true"] { font-family: @mono@; font-size: @caption@px; background: @code_bg@; }
QComboBox { padding-right: 28px; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: url(@chev_down@); width: 14px; height: 14px; }
QComboBox QAbstractItemView { background: @card@; color: @text@; border: 1px solid @border_strong@; selection-background-color: @accent@;
    selection-color: @accent_text@; padding: 4px; outline: 0; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; background: transparent; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(@chev_up@); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(@chev_down@); width: 10px; height: 10px; }

QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1.5px solid @border_strong@; border-radius: 5px; background: @surface@; }
QCheckBox::indicator:hover { border-color: @accent@; }
QCheckBox::indicator:checked { background: @accent@; border-color: @accent@; image: url(@check@); }
QCheckBox::indicator:disabled { background: @bg@; border-color: @border@; }
QCheckBox:focus { color: @text@; }
QRadioButton::indicator { width: 16px; height: 16px; border: 1.5px solid @border_strong@; border-radius: 9px; background: @surface@; }
QRadioButton::indicator:checked { border: 5px solid @accent@; background: @accent_text@; }

QSlider::groove:horizontal { height: 4px; background: @border_strong@; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent@; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: @accent@; border: 2px solid @bg@; }

QProgressBar { background: @border@; border: none; border-radius: 4px; max-height: 8px; min-height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: @accent@; border-radius: 4px; }
QProgressBar[state="error"]::chunk { background: @danger@; }
QProgressBar[state="ok"]::chunk { background: @success@; }

/* lists / tables / trees -------------------------------------------------------------------------- */
QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView { background: @surface@; alternate-background-color: @card@;
    border: 1px solid @border@; border-radius: @r_sm@px; selection-background-color: @selection@; selection-color: @text@; outline: 0; }
QListWidget::item { padding: 8px 10px; border-radius: 6px; margin: 1px 4px; }
QListWidget::item:hover { background: @card_hover@; }
QListWidget::item:selected { background: @selection@; color: @text@; }
QTableView::item, QTableWidget::item, QTreeView::item { padding: 6px 8px; border: none; }
QTableView::item:selected, QTableWidget::item:selected { background: @selection@; color: @text@; }
QTableView::item:hover { background: @card_hover@; }
QHeaderView::section { background: @card@; color: @text_muted@; padding: 8px 10px; border: none; border-bottom: 1px solid @border@; font-weight: 600; }
QTableView { gridline-color: @border@; }
QTableCornerButton::section { background: @card@; border: none; }

QTabWidget::pane { border: 1px solid @border@; border-radius: @r_sm@px; top: -1px; }
QTabBar::tab { background: transparent; color: @text_muted@; padding: 8px 16px; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: @text@; border-bottom: 2px solid @accent@; }
QTabBar::tab:hover { color: @text@; }

QMenu { background: @card@; border: 1px solid @border_strong@; border-radius: @r_sm@px; padding: 6px; }
QMenu::item { padding: 7px 24px 7px 12px; border-radius: 6px; }
QMenu::item:selected { background: @accent_soft@; color: @text@; }
QMenu::item:disabled { color: @text_faint@; }
QMenu::separator { height: 1px; background: @border@; margin: 6px 4px; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: @border_strong@; border-radius: 4px; min-height: 32px; margin: 0 2px; }
QScrollBar::handle:vertical:hover { background: @text_faint@; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: @border_strong@; border-radius: 4px; min-width: 32px; margin: 2px 0; }
QScrollBar::handle:horizontal:hover { background: @text_faint@; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

QSplitter::handle { background: @border@; }
QGroupBox { border: 1px solid @border@; border-radius: @r_sm@px; margin-top: 14px; padding: 14px 10px 10px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: @text_muted@; font-weight: 600; }

/* app-specific ------------------------------------------------------------------------------------- */
QWidget#sidebar { background: @surface@; border-right: 1px solid @border@; }
QListWidget#nav { background: transparent; border: none; }
QListWidget#nav::item { padding: 10px 12px; margin: 2px 8px; border-radius: 8px; color: @text_muted@; }
QListWidget#nav::item:hover { background: @card_hover@; color: @text@; }
QListWidget#nav::item:selected { background: @accent_soft@; color: @text@; font-weight: 600; }
QWidget#footer { background: @surface@; border-top: 1px solid @border@; }
QWidget#header { background: @surface@; border-bottom: 1px solid @border@; }
QFrame#banner[kind="error"]   { background: @danger_soft@;  border: 1px solid @danger@;  border-radius: @r_sm@px; }
QFrame#banner[kind="warning"] { background: @warning_soft@; border: 1px solid @warning@; border-radius: @r_sm@px; }
QFrame#banner[kind="info"]    { background: @info_soft@;    border: 1px solid @info@;    border-radius: @r_sm@px; }
QFrame#banner[kind="success"] { background: @success_soft@; border: 1px solid @success@; border-radius: @r_sm@px; }
QFrame#toast { background: @card@; border: 1px solid @border_strong@; border-radius: @r_md@px; }
QFrame#toast[kind="error"] { border-left: 4px solid @danger@; } QFrame#toast[kind="warning"] { border-left: 4px solid @warning@; }
QFrame#toast[kind="success"] { border-left: 4px solid @success@; } QFrame#toast[kind="info"] { border-left: 4px solid @info@; }
QLabel[badge="true"] { border-radius: 10px; padding: 2px 10px; font-size: @caption@px; font-weight: 600; background: @accent_soft@; color: @text@; }
QLabel[badge="true"][tone="success"] { background: @success_soft@; color: @success@; }
QLabel[badge="true"][tone="warning"] { background: @warning_soft@; color: @warning@; }
QLabel[badge="true"][tone="danger"]  { background: @danger_soft@;  color: @danger@; }
QLabel[badge="true"][tone="info"]    { background: @info_soft@;    color: @info@; }
QLabel[badge="true"][tone="neutral"] { background: @card_hover@;   color: @text_muted@; }
QFrame#chip { background: @card_hover@; border: 1px solid @border@; border-radius: 12px; }
QFrame#chip:hover { border-color: @accent@; }
QFrame#stage[state="idle"]    { background: @surface@;       border: 1px solid @border@; }
QFrame#stage[state="running"] { background: @info_soft@;     border: 1px solid @info@; }
QFrame#stage[state="ok"]      { background: @success_soft@;  border: 1px solid @success@; }
QFrame#stage[state="warn"]    { background: @warning_soft@;  border: 1px solid @warning@; }
QFrame#stage[state="error"]   { background: @danger_soft@;   border: 1px solid @danger@; }
QFrame#stage { border-radius: 10px; }
QWidget#answerSurface { background: @card@; }
QTextBrowser#answer { background: transparent; border: none; selection-background-color: @accent@; selection-color: @accent_text@; }
QFrame#drop[active="true"] { border: 2px dashed @accent@; background: @accent_soft@; }
QFrame#drop { border: 2px dashed @border_strong@; border-radius: @r_md@px; }
QFrame#skeleton { background: @card_hover@; border-radius: 6px; border: none; }
QFrame#palette { background: @card@; border: 1px solid @border_strong@; border-radius: @r_lg@px; }
QLabel#kbd { background: @card_hover@; border: 1px solid @border_strong@; border-radius: 5px; padding: 1px 6px; font-family: @mono@; font-size: @caption@px; color: @text_muted@; }
"""


def build_qss(t: Theme, body_px: int = 14) -> str:
    from display.icons import icon_file
    sizes = {"body": body_px, "caption": max(10, body_px - 2), "section": body_px + 2, "title": body_px + 7,
             "display": body_px + 14}
    subs = {**{k: t[k] for k in TOKEN_NAMES}, **{k: str(v) for k, v in sizes.items()},
            "ui": UI_FONTS, "mono": MONO_FONTS,
            "r_sm": str(RADII["sm"]), "r_md": str(RADII["md"]), "r_lg": str(RADII["lg"]),
            "chev_down": icon_file("chevron_down", t["text_muted"]), "chev_up": icon_file("chevron_up", t["text_muted"]),
            "check": icon_file("check", t["accent_text"])}
    return re.sub(r"@(\w+)@", lambda m: subs.get(m.group(1), m.group(0)), _QSS)


def repolish(w) -> None:
    """Re-evaluate the stylesheet after changing a dynamic property."""
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


def apply_app_palette(legacy: dict | None = None) -> None:     # kept for old imports
    manager().apply()
