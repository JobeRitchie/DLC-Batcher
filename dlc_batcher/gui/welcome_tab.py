"""Welcome page and the dialog for choosing the DeepLabCut environment."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QBoxLayout, QDialog, QDialogButtonBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __date__, __version__, dlc_env
from . import theme as T

DESCRIPTION = (
    "Runs DeepLabCut's analyze_videos over a list of videos and writes the same "
    "result files DeepLabCut would. The model stays loaded between videos, and "
    "analyzed videos are ranked by how well the model tracked them."
)
AUTO_TUNE = (
    "Auto-tune times your model on this computer with short trials (larger "
    "batches, more workers, half precision), checks free GPU memory before each "
    "step, and recommends the fastest settings. Run it once per model before a "
    "large batch."
)


def _label(text: str, name: str, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(name)
    lab.setAlignment(Qt.AlignCenter)
    lab.setWordWrap(wrap)
    return lab


class EnvDialog(QDialog):
    """Pick one of the DeepLabCut environments that were found, or browse."""

    def __init__(self, candidates: list, current: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("DeepLabCut environment")
        self.choice: dlc_env.Candidate | None = None
        em = self.fontMetrics().height()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(em, em, em, em)
        lay.setSpacing(em // 2)
        if candidates:
            intro = "Workers run in the environment selected here."
        else:
            intro = ("No environment with DeepLabCut was found. Browse to the python.exe "
                     "of the environment you use for DeepLabCut.")
        head = QLabel(intro)
        head.setWordWrap(True)
        lay.addWidget(head)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        lay.addWidget(self.list, 1)
        for cand in candidates:
            self._add(cand, select=_same(cand.python, current))
        if self.list.count() and self.list.currentRow() < 0:
            self.list.setCurrentRow(0)
        self.list.setMinimumWidth(self.fontMetrics().averageCharWidth() * 80)
        self.list.setMinimumHeight(em * 6)
        row = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        lay.addLayout(row)

    def _add(self, cand, select: bool = False) -> None:
        note = "" if cand.tested else "  (not tested with this version)"
        item = QListWidgetItem(f"DeepLabCut {cand.version}{note}\n{cand.python}")
        item.setData(Qt.UserRole, cand)
        self.list.addItem(item)
        if select:
            self.list.setCurrentItem(item)

    def _browse(self) -> None:
        pattern = "Python (python.exe python)" if dlc_env._WINDOWS else "Python (python*)"
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the Python of your DeepLabCut environment", str(Path.home()),
            f"{pattern};;All files (*)")
        if not path:
            return
        cand, why = dlc_env.validate(path)
        if cand is None:
            QMessageBox.warning(self, "Not a DeepLabCut environment", why)
            return
        for i in range(self.list.count()):
            if _same(self.list.item(i).data(Qt.UserRole).python, cand.python):
                self.list.setCurrentRow(i)
                return
        self._add(cand, select=True)

    def accept(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self.choice = item.data(Qt.UserRole)
        super().accept()


def _same(a: str, b: str) -> bool:
    return bool(a and b) and Path(a) == Path(b)


class WelcomeTab(QWidget):
    open_project_requested = Signal()
    run_requested = Signal()
    recent_requested = Signal(str)
    change_env_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        em = self.fontMetrics().height()
        self._em = em

        title = _label(APP_NAME.upper(), "splashTitle")
        font = QFont(title.font())
        font.setLetterSpacing(QFont.PercentageSpacing, 118)
        title.setFont(font)
        tagline = _label("Batch video analysis for DeepLabCut models", "splashTagline", True)
        author = _label("Created by Jobe Ritchie, PhD", "splashAuthor")
        lab = _label("Kash Lab", "splashOrg")
        org = _label("University of North Carolina at Chapel Hill", "splashOrg", True)
        meta = _label(f"v{__version__}  ·  {__date__}", "splashMeta")
        divider = QFrame()
        divider.setObjectName("splashRule")
        divider.setFrameShape(QFrame.HLine)
        about = _label(DESCRIPTION, "splashText", True)

        self.env_line = _label("", "splashEnv", True)
        self.env_line.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.env_warn = _label("", "splashWarn", True)
        self.env_warn.hide()
        self.change_btn = QPushButton("Change…")
        self.change_btn.setToolTip("Choose which DeepLabCut environment runs the analysis")
        self.change_btn.clicked.connect(self.change_env_requested)
        self.change_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.env_row = QBoxLayout(QBoxLayout.LeftToRight)
        self.env_row.setSpacing(em // 2)
        self.env_row.addWidget(self.env_line, 1)
        self.env_row.addWidget(self.change_btn, 0, Qt.AlignCenter)

        self.open_btn = QPushButton("Open a project…")
        self.open_btn.setObjectName("primary")
        self.open_btn.clicked.connect(self.open_project_requested)
        self.run_btn = QPushButton("Go to Run")
        self.run_btn.clicked.connect(self.run_requested)
        self.btn_row = QBoxLayout(QBoxLayout.LeftToRight)
        self.btn_row.setSpacing(em // 2)
        self.btn_row.addWidget(self.open_btn, 1)
        self.btn_row.addWidget(self.run_btn, 1)

        self.recent_label = QLabel("Recent projects")
        self.recent_label.setObjectName("cardTitle")
        self.recent = QListWidget()
        self.recent.setToolTip("Double-click a project to open it")
        self.recent.itemActivated.connect(
            lambda item: self.recent_requested.emit(item.data(Qt.UserRole)))
        self.recent_empty = QLabel("Projects you open are listed here.")
        self.recent_empty.setObjectName("muted")

        self.column = QWidget()
        col = QVBoxLayout(self.column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(em // 4)
        for w in (title, tagline):
            col.addWidget(w)
        col.addSpacing(em // 2)
        for w in (author, lab, org, meta):
            col.addWidget(w)
        col.addSpacing(em // 2)
        col.addWidget(divider)
        col.addSpacing(em // 2)
        col.addWidget(about)
        col.addSpacing(em // 2)
        col.addWidget(_label(AUTO_TUNE, "splashText", True))
        col.addSpacing(em)
        col.addLayout(self.env_row)
        col.addWidget(self.env_warn)
        col.addSpacing(em)
        col.addLayout(self.btn_row)
        col.addSpacing(em)
        col.addWidget(self.recent_label)
        col.addWidget(self.recent)
        col.addWidget(self.recent_empty)

        content = QWidget()
        self.outer = QHBoxLayout(content)
        self.outer.addStretch(1)
        self.outer.addWidget(self.column, 0, Qt.AlignTop)
        self.outer.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)
        self.set_recent([])
        self._fit()

    # ---------------------------------------------------------------- state
    def set_environment(self, cand, count: int = 1) -> None:
        if cand is None:
            self.env_line.setText(
                f"<span style='color:{T.RED}'>DeepLabCut was not found.</span> "
                "Choose the Python environment it is installed in.")
            self.change_btn.setText("Choose…")
            self.env_warn.hide()
            return
        self.change_btn.setText("Change…")
        self.env_line.setText(f"DeepLabCut {cand.version} at {cand.python}")
        if count > 1:
            self.change_btn.setToolTip(f"{count} DeepLabCut environments were found")
        if cand.tested:
            self.env_warn.hide()
        else:
            tested = " and ".join(dlc_env.TESTED_VERSIONS)
            self.env_warn.setText(
                f"DLC Batcher was tested with DeepLabCut {tested}. Version {cand.version} "
                "may work, but check the first results against a plain DeepLabCut run.")
            self.env_warn.show()

    def set_recent(self, paths: list[str]) -> None:
        self.recent.clear()
        for p in paths:
            path = Path(p)
            item = QListWidgetItem(f"{path.parent.name}\n{path}")
            item.setData(Qt.UserRole, str(path))
            self.recent.addItem(item)
        has = self.recent.count() > 0
        self.recent.setVisible(has)
        self.recent_empty.setVisible(not has)
        if has:
            rows = min(self.recent.count(), 5)
            self.recent.setFixedHeight(self.recent.sizeHintForRow(0) * rows
                                       + 2 * self.recent.frameWidth() + self._em // 2)

    # --------------------------------------------------------------- layout
    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._fit()

    def _fit(self) -> None:
        """Keep the text column readable: capped on wide windows, full width
        with stacked buttons on narrow ones. All sizes follow the font."""
        char = self.fontMetrics().averageCharWidth()
        margin = self._em * (1 if self.width() < char * 90 else 2)
        self.outer.setContentsMargins(margin, margin, margin, margin)
        avail = max(char * 20, self.width() - 2 * margin - self._em)
        self.column.setFixedWidth(min(char * 110, avail))
        narrow = avail < char * 70
        direction = QBoxLayout.TopToBottom if narrow else QBoxLayout.LeftToRight
        self.btn_row.setDirection(direction)
        self.env_row.setDirection(direction)
