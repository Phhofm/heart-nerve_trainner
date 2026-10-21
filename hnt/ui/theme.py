"""Modern dark theme (Qt stylesheet) — matches the website's design language.

Keep it calm, spacious and high-contrast: generous rounded cards, a gradient
accent, pill-shaped chips, and stat tiles.
"""

ACCENT = "#7c9cff"
ACCENT2 = "#b39dff"
BG = "#0e1016"
PANEL = "#161923"
PANEL2 = "#1d2130"
LINE = "#2a2f3d"
TEXT = "#e7e9ee"
MUTED = "#9aa1b1"
OK = "#57d38c"
WARN = "#ffb454"
ERR = "#ff6b6b"

QSS = f"""
* {{ font-family: 'Inter','Segoe UI','Noto Sans',system-ui,sans-serif; font-size: 14px; color: {TEXT}; }}

QMainWindow {{ background: {BG}; }}

QGroupBox {{
  background: {PANEL};
  border: 1px solid {LINE};
  border-radius: 16px;
  margin-top: 16px;
  padding: 18px 18px 16px 18px;
}}
QGroupBox::title {{
  subcontrol-origin: margin; left: 14px; padding: 0 8px;
  color: {MUTED}; font-weight: 700; letter-spacing: 1px;
}}

QLabel#h1 {{ font-size: 27px; font-weight: 800; }}
QLabel#subtitle {{ color: {ACCENT}; font-size: 14px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#chip {{
  color: {MUTED}; background: {PANEL}; border: 1px solid {LINE};
  border-radius: 12px; padding: 5px 12px; font-size: 12.5px;
}}

QWidget#stat {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 12px; }}
QLabel#statlabel {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#metric {{ font-size: 21px; font-weight: 700; }}

QFrame#chartpanel {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 12px; }}

QPushButton {{
  background: {PANEL2}; border: 1px solid {LINE}; border-radius: 12px;
  padding: 10px 16px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {PANEL}; }}
QPushButton:disabled {{ color: #5a6070; border-color: #232734; }}
QPushButton#primary {{
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #8aa6ff, stop:1 {ACCENT});
  color: #0b0d12; border: none; padding: 12px 22px; font-weight: 800;
}}
QPushButton#primary:hover {{
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #9db4ff, stop:1 #8aa6ff);
}}
QPushButton#primary:disabled {{ background: #3b4560; color: #8a90a0; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
  background: {PANEL2}; border: 1px solid {LINE}; border-radius: 10px; padding: 8px 10px;
  selection-background-color: {ACCENT}; selection-color: #0b0d12;
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover,
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox[recommended="true"] {{ border: 1px solid {OK}; }}
QLineEdit[invalid="true"], QComboBox[invalid="true"] {{ border: 1px solid {ERR}; }}
QComboBox QAbstractItemView {{
  background: {PANEL2}; border: 1px solid {LINE}; outline: none;
  selection-background-color: {ACCENT}; selection-color: #0b0d12;
}}

QProgressBar {{
  background: {PANEL2}; border: 1px solid {LINE}; border-radius: 9px;
  height: 16px; text-align: center; color: {MUTED}; font-size: 11px;
}}
QProgressBar::chunk {{
  border-radius: 8px;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT}, stop:1 {ACCENT2});
}}

QPlainTextEdit {{
  background: #0b0d13; border: 1px solid {LINE}; border-radius: 12px;
  font-family: 'JetBrains Mono','Consolas',monospace; font-size: 12px; padding: 8px;
}}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {LINE}; background: {PANEL2}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
QScrollBar::handle:vertical {{ background: #333a4d; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {LINE}; padding: 6px; border-radius: 8px; }}
"""
