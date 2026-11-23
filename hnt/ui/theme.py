"""BODY Studio themes (Qt stylesheets) — dark and light, same design language.

Keep both calm, spacious and high-contrast: generous rounded cards, a gradient
accent, pill-shaped chips, and stat tiles. The active mode lives in module
state (`set_mode`/`active_mode`); the main window persists the user's choice
and applies `stylesheet(mode)` to the whole application.
"""

from __future__ import annotations

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "ACCENT": "#7c9cff",
        "ACCENT2": "#b39dff",
        "BG": "#0e1016",
        "PANEL": "#161923",
        "PANEL2": "#1d2130",
        "LINE": "#2a2f3d",
        "TEXT": "#e7e9ee",
        "MUTED": "#9aa1b1",
        "OK": "#57d38c",
        "WARN": "#ffb454",
        "ERR": "#ff6b6b",
        "SEL_TEXT": "#0b0d12",
        "CONSOLE_BG": "#0b0d13",
        "SCROLL_HANDLE": "#333a4d",
        "PRIMARY_TOP": "#8aa6ff",
        "PRIMARY_HOVER": "#9db4ff",
        "PRIMARY_TEXT": "#0b0d12",
        "DISABLED_TEXT": "#5a6070",
        "DISABLED_LINE": "#232734",
    },
    "light": {
        "ACCENT": "#4f6fe0",
        "ACCENT2": "#8a5fd6",
        "BG": "#f5f6fa",
        "PANEL": "#ffffff",
        "PANEL2": "#eef0f6",
        "LINE": "#d9dde8",
        "TEXT": "#1a1d27",
        "MUTED": "#5b6272",
        "OK": "#1f9d63",
        "WARN": "#b56f14",
        "ERR": "#d64545",
        "SEL_TEXT": "#ffffff",
        "CONSOLE_BG": "#ffffff",
        "SCROLL_HANDLE": "#b9bfd0",
        "PRIMARY_TOP": "#6d87ff",
        "PRIMARY_HOVER": "#5d7bf0",
        "PRIMARY_TEXT": "#ffffff",
        "DISABLED_TEXT": "#a4aab8",
        "DISABLED_LINE": "#e3e6ee",
    },
}

MODES = tuple(PALETTES)

_active = "dark"

# Backwards-compatible module constants (the dark palette). Chart colors and
# any other importers keep working; dynamic readers use `active()`.
ACCENT = PALETTES["dark"]["ACCENT"]
ACCENT2 = PALETTES["dark"]["ACCENT2"]
BG = PALETTES["dark"]["BG"]
PANEL = PALETTES["dark"]["PANEL"]
PANEL2 = PALETTES["dark"]["PANEL2"]
LINE = PALETTES["dark"]["LINE"]
TEXT = PALETTES["dark"]["TEXT"]
MUTED = PALETTES["dark"]["MUTED"]
OK = PALETTES["dark"]["OK"]
WARN = PALETTES["dark"]["WARN"]
ERR = PALETTES["dark"]["ERR"]


def active_mode() -> str:
    return _active


def active() -> dict[str, str]:
    return PALETTES[_active]


def set_mode(mode: str, app=None) -> None:
    """Switch the active palette process-wide and restyle the app."""
    global _active
    if mode not in PALETTES:
        raise ValueError(f"unknown theme mode: {mode!r}")
    _active = mode
    if app is not None:
        app.setStyleSheet(stylesheet(mode))


def stylesheet(mode: str | None = None) -> str:
    p = PALETTES[mode] if mode else active()
    return f"""
* {{ font-family: 'Inter','Segoe UI','Noto Sans',system-ui,sans-serif; font-size: 14px; color: {p['TEXT']}; }}

QMainWindow {{ background: {p['BG']}; }}

QGroupBox {{
  background: {p['PANEL']};
  border: 1px solid {p['LINE']};
  border-radius: 16px;
  margin-top: 16px;
  padding: 18px 18px 16px 18px;
}}
QGroupBox::title {{
  subcontrol-origin: margin; left: 14px; padding: 0 8px;
  color: {p['MUTED']}; font-weight: 700; letter-spacing: 1px;
}}

QLabel#h1 {{ font-size: 27px; font-weight: 800; }}
QLabel#subtitle {{ color: {p['ACCENT']}; font-size: 14px; font-weight: 600; }}
QLabel#muted {{ color: {p['MUTED']}; }}
QLabel#chip {{
  color: {p['MUTED']}; background: {p['PANEL']}; border: 1px solid {p['LINE']};
  border-radius: 12px; padding: 5px 12px; font-size: 12.5px;
}}

QWidget#stat {{ background: {p['PANEL2']}; border: 1px solid {p['LINE']}; border-radius: 12px; }}
QLabel#statlabel {{ color: {p['MUTED']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#metric {{ font-size: 21px; font-weight: 700; }}

QFrame#chartpanel {{ background: {p['PANEL2']}; border: 1px solid {p['LINE']}; border-radius: 12px; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QListWidget {{
  background: {p['PANEL2']}; border: 1px solid {p['LINE']}; border-radius: 12px;
  padding: 6px; outline: none;
}}
QListWidget::item {{ padding: 7px 8px; border-radius: 8px; }}
QListWidget::item:selected {{ background: {p['ACCENT']}; color: {p['SEL_TEXT']}; }}

QPushButton {{
  background: {p['PANEL2']}; border: 1px solid {p['LINE']}; border-radius: 12px;
  padding: 10px 16px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {p['ACCENT']}; }}
QPushButton:pressed {{ background: {p['PANEL']}; }}
QPushButton:disabled {{ color: {p['DISABLED_TEXT']}; border-color: {p['DISABLED_LINE']}; }}
QPushButton#primary {{
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {p['PRIMARY_TOP']}, stop:1 {p['ACCENT']});
  color: {p['PRIMARY_TEXT']}; border: none; padding: 12px 22px; font-weight: 800;
}}
QPushButton#primary:hover {{
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {p['PRIMARY_HOVER']}, stop:1 {p['ACCENT']});
}}
QPushButton#primary:disabled {{ background: {p['DISABLED_LINE']}; color: {p['DISABLED_TEXT']}; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
  background: {p['PANEL2']}; border: 1px solid {p['LINE']}; border-radius: 10px; padding: 8px 10px;
  selection-background-color: {p['ACCENT']}; selection-color: {p['SEL_TEXT']};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover,
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {p['ACCENT']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox[recommended="true"] {{ border: 1px solid {p['OK']}; }}
QLineEdit[invalid="true"], QComboBox[invalid="true"] {{ border: 1px solid {p['ERR']}; }}
QComboBox QAbstractItemView {{
  background: {p['PANEL2']}; border: 1px solid {p['LINE']}; outline: none;
  selection-background-color: {p['ACCENT']}; selection-color: {p['SEL_TEXT']};
}}

QProgressBar {{
  background: {p['PANEL2']}; border: 1px solid {p['LINE']}; border-radius: 9px;
  height: 16px; text-align: center; color: {p['MUTED']}; font-size: 11px;
}}
QProgressBar::chunk {{
  border-radius: 8px;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {p['ACCENT']}, stop:1 {p['ACCENT2']});
}}

QPlainTextEdit {{
  background: {p['CONSOLE_BG']}; border: 1px solid {p['LINE']}; border-radius: 12px;
  font-family: 'JetBrains Mono','Consolas',monospace; font-size: 12px; padding: 8px;
}}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {p['LINE']}; background: {p['PANEL2']}; }}
QCheckBox::indicator:hover {{ border-color: {p['ACCENT']}; }}
QCheckBox::indicator:checked {{ background: {p['ACCENT']}; border-color: {p['ACCENT']}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
QScrollBar::handle:vertical {{ background: {p['SCROLL_HANDLE']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p['ACCENT']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QToolTip {{ background: {p['PANEL2']}; color: {p['TEXT']}; border: 1px solid {p['LINE']}; padding: 6px; border-radius: 8px; }}
"""


# Kept for any importer that still expects the original dark-only constant.
QSS = stylesheet("dark")
