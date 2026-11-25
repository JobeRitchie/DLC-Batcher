"""Colors, fonts and the application stylesheet."""
from __future__ import annotations

BG = "#15131f"
BG2 = "#1b1829"
CARD = "#221f33"
CARD2 = "#2a2640"
BORDER = "#363152"
TEXT = "#efecff"
MUTED = "#a09bc0"
PINK = "#ff5fa2"
CYAN = "#3ddbd9"
YELLOW = "#ffd166"
GREEN = "#5be584"
RED = "#ff6b6b"
ORANGE = "#ff9f5a"
PURPLE = "#9b7bff"

STATUS_COLORS = {"good": GREEN, "check": YELLOW, "problem": RED,
                 "info": CYAN, "warn": YELLOW, "bad": RED}


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_int(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{int(n):,}"


STYLESHEET = f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QToolTip {{ background: {CARD2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#banner {{ background: {CARD2}; border: 1px solid {PINK}; border-radius: 12px; }}
QLabel#cardTitle {{ font-size: 11pt; font-weight: 700; }}
QLabel#step {{ background: {PINK}; color: #1a0a12; border-radius: 11px; font-weight: 800;
               min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#h1 {{ font-size: 19pt; font-weight: 800; }}
QLabel#tagline {{ color: {MUTED}; font-size: 10pt; }}
QLabel#tileValue {{ font-size: 18pt; font-weight: 800; }}
QLabel#tileCaption {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#boost {{ color: {YELLOW}; font-weight: 700; }}
QLabel#bigBoost {{ color: {YELLOW}; font-size: 26pt; font-weight: 800; }}
QLabel#splashTitle {{ font-size: 34pt; font-weight: 900; color: {TEXT}; }}
QLabel#splashTagline {{ font-size: 12pt; font-weight: 600; color: {CYAN}; }}
QLabel#splashAuthor {{ font-size: 11.5pt; font-weight: 700; color: {TEXT}; }}
QLabel#splashOrg {{ font-size: 10.5pt; font-weight: 600; color: {MUTED}; }}
QLabel#splashMeta {{ font-size: 9pt; color: {MUTED}; }}
QLabel#splashText {{ font-size: 10.5pt; color: {TEXT}; }}
QLabel#splashEnv {{ font-size: 10pt; color: {TEXT}; }}
QLabel#splashWarn {{ font-size: 9.5pt; color: {YELLOW}; }}
QFrame#splashRule {{ color: {BORDER}; }}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:selected {{ background: #3a3360; color: {TEXT}; }}
QPushButton {{ background: {CARD2}; border: 1px solid {BORDER}; border-radius: 9px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {PURPLE}; background: #322d4d; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: #6c6788; background: {BG2}; border-color: {BG2}; }}
QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {PINK}, stop:1 {PURPLE});
                       border: none; color: white; font-weight: 800; font-size: 11.5pt; padding: 10px 18px; border-radius: 12px; }}
QPushButton#primary:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ff7db4, stop:1 #b198ff); }}
QPushButton#primary:disabled {{ background: {BORDER}; color: #8d88aa; }}
QPushButton#danger {{ border-color: {RED}; color: {RED}; font-weight: 700; }}
QPushButton#danger:hover {{ background: #3a2030; }}
QPushButton#accent {{ border-color: {CYAN}; color: {CYAN}; font-weight: 700; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {BG2}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 5px 8px; selection-background-color: {PURPLE}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {CYAN}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {CARD2}; border: 1px solid {BORDER}; selection-background-color: {PURPLE}; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {BORDER}; background: {BG2}; }}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 9px 18px; margin-right: 4px;
                border-bottom: 3px solid transparent; font-weight: 700; font-size: 10.5pt; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 3px solid {PINK}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTableWidget, QTableView, QPlainTextEdit, QTextEdit, QListWidget {{ background: {BG2}; border: 1px solid {BORDER};
    border-radius: 10px; gridline-color: transparent; alternate-background-color: #1f1c2f; }}
QTableWidget::item {{ padding: 3px 6px; border: none; }}
QTableWidget::item:selected {{ background: #3a3360; color: {TEXT}; }}
QHeaderView::section {{ background: {BG2}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
                        padding: 6px; font-weight: 700; font-size: 8.5pt; }}
QTableCornerButton::section {{ background: {BG2}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QProgressBar {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 7px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {CYAN}; border-radius: 6px; }}
QPlainTextEdit {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; }}
"""
