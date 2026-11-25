"""Main window: header and the Welcome / Run / Auto-tune / Quality / Log tabs."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, dlc_env, hardware, settings, update
from . import theme as T
from .quality_tab import QualityTab
from .run_tab import RunTab
from .tune_tab import TuneTab
from .welcome_tab import EnvDialog, WelcomeTab
from .widgets import HardwareMonitor, muted

MAX_RECENT = 8


class MainWindow(QMainWindow):
    update_checked = Signal(object, bool)  # status, asked for by the user
    update_applied = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self.updater = update.Updater()
        self._update_busy = False
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.resize(int(area.width() * 0.8), int(area.height() * 0.85))
        self.store = settings.load()
        self.hw = hardware.detect_hardware()
        self.dlc, self.dlc_candidates = dlc_env.resolve(self.store)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(18, 12, 18, 14)
        lay.setSpacing(4)

        head = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("h1")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(muted(self.hw.summary(), wrap=False), 0, Qt.AlignBottom)
        self.check_btn = QPushButton("Check for updates")
        self.check_btn.clicked.connect(lambda: self.check_for_updates(manual=True))
        head.addWidget(self.check_btn, 0, Qt.AlignBottom)
        lay.addLayout(head)

        self.banner = QFrame()
        self.banner.setObjectName("banner")
        bl = QHBoxLayout(self.banner)
        self.banner_text = QLabel("")
        self.banner_text.setWordWrap(True)
        self.update_btn = QPushButton("Update and restart")
        self.update_btn.setObjectName("accent")
        self.update_btn.clicked.connect(self.install_update)
        later = QPushButton("Later")
        later.clicked.connect(self.banner.hide)
        bl.addWidget(self.banner_text, 1)
        bl.addWidget(self.update_btn)
        bl.addWidget(later)
        self.banner.hide()
        lay.addWidget(self.banner)
        self.update_checked.connect(self._on_update_checked)
        self.update_applied.connect(self._on_update_applied)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.welcome = WelcomeTab()
        self.run_tab = RunTab(self.hw, self.store)
        self.tune_tab = TuneTab(self.hw, self.store, self.run_tab)
        self.quality_tab = QualityTab()
        logw = QWidget()
        ll = QVBoxLayout(logw)
        ll.setContentsMargins(0, 8, 0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        clear = QPushButton("Clear log")
        clear.clicked.connect(self.log.clear)
        ll.addWidget(self.log, 1)
        ll.addWidget(clear, 0, Qt.AlignRight)
        self.tabs.addTab(self.welcome, "Welcome")
        self.tabs.addTab(self.run_tab, "Run")
        self.tabs.addTab(self.tune_tab, "Auto-tune")
        self.tabs.addTab(self.quality_tab, "Quality")
        self.tabs.addTab(logw, "Log")

        for src in (self.run_tab, self.tune_tab, self.quality_tab):
            src.log.connect(self._log)
        self.run_tab.tune_requested.connect(lambda: self.tabs.setCurrentWidget(self.tune_tab))
        self.run_tab.project_loaded.connect(self._project_loaded)
        self.run_tab.run_finished.connect(self._run_finished)
        self.tune_tab.applied.connect(lambda: self.tabs.setCurrentWidget(self.run_tab))
        self.welcome.run_requested.connect(lambda: self.tabs.setCurrentWidget(self.run_tab))
        self.welcome.open_project_requested.connect(self._open_project)
        self.welcome.recent_requested.connect(self._open_recent)
        self.welcome.change_env_requested.connect(self.choose_environment)
        self.welcome.set_recent(self._recent())
        if self.run_tab.project is not None:
            self._project_loaded(self.run_tab.project)

        self.monitor = HardwareMonitor(1.5, self)
        self.monitor.sample.connect(self.run_tab.on_hw_sample)
        self.monitor.sample.connect(self.tune_tab.on_hw_sample)
        self.monitor.start()
        self._log(f"{APP_NAME} {__version__} · {self.hw.summary()}")
        self._set_environment(self.dlc, save=False)
        if self.dlc is None:
            QTimer.singleShot(0, self.choose_environment)
        QTimer.singleShot(1500, self.check_for_updates)

    # --------------------------------------------------------------- updates
    def check_for_updates(self, manual: bool = False) -> None:
        """Ask the git remote on a background thread; the banner appears only
        when there is something newer."""
        if self._update_busy:
            return
        self._update_busy = True
        self.check_btn.setEnabled(False)
        threading.Thread(
            target=lambda: self.update_checked.emit(self.updater.check(), manual),
            daemon=True).start()

    def _on_update_checked(self, status, manual: bool) -> None:
        self._update_busy = False
        self.check_btn.setEnabled(True)
        if status.available:
            self.banner_text.setText(status.summary)
            self.update_btn.setEnabled(True)
            self.banner.show()
            self._log(status.summary)
        elif status.error:
            self._log(f"Update check: {status.error}")
            if manual:
                QMessageBox.information(self, "Check for updates", status.error)
        elif manual:
            QMessageBox.information(self, "Check for updates",
                                    f"{APP_NAME} {__version__} is up to date.")

    def install_update(self) -> None:
        if self.run_tab.manager.running or self.tune_tab.cal.running:
            QMessageBox.information(self, "Update",
                                    "Wait for the current run to finish, then update.")
            return
        if self._update_busy:
            return
        self._update_busy = True
        self.update_btn.setEnabled(False)
        self.banner_text.setText("Updating…")
        threading.Thread(target=lambda: self.update_applied.emit(*self.updater.apply()),
                         daemon=True).start()

    def _on_update_applied(self, ok: bool, output: str) -> None:
        self._update_busy = False
        self._log(f"Update: {output}")
        if not ok:
            self.banner_text.setText("The update could not be applied.")
            self.update_btn.setEnabled(True)
            QMessageBox.warning(self, "Update failed", output or "git reported an error.")
            return
        self.banner_text.setText("Updated. Restarting…")
        self.updater.relaunch()
        self.close()
        QApplication.quit()

    def _log(self, text: str) -> None:
        if text:
            self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    # ------------------------------------------------- DeepLabCut environment
    def _set_environment(self, cand, save: bool = True, picked: bool = False) -> None:
        self.dlc = cand
        self.run_tab.python_exe = cand.python if cand else ""
        self.welcome.set_environment(cand, len(self.dlc_candidates))
        if cand is None:
            self._log("DeepLabCut was not found. Choose its environment on the Welcome tab.")
            return
        self._log(f"Using {cand.label}")
        if not cand.tested:
            self._log(f"Note: DeepLabCut {cand.version} is not one of the tested versions "
                      f"({', '.join(dlc_env.TESTED_VERSIONS)}).")
        # DLC_BATCHER_PYTHON applies to this session only and is never saved
        if cand.source != "env var" and (save or cand.source != "saved"):
            dlc_env.remember(self.store, cand, picked=picked or cand.source == "saved")
            settings.save(self.store)

    def choose_environment(self) -> None:
        if self.run_tab.manager.running or self.tune_tab.cal.running:
            return
        dlg = EnvDialog(self.dlc_candidates, self.dlc.python if self.dlc else "", self)
        if dlg.exec() != QDialog.Accepted or dlg.choice is None:
            return
        cand = dlg.choice
        if all(Path(c.python) != Path(cand.python) for c in self.dlc_candidates):
            self.dlc_candidates.append(cand)
        self._set_environment(cand, picked=True)

    # -------------------------------------------------------------- projects
    def _recent(self) -> list[str]:
        return [p for p in self.store.get("recent_projects", []) if Path(p).is_file()]

    def _open_project(self) -> None:
        before = self.run_tab.project
        self.run_tab.browse_project()
        if self.run_tab.project is not before:
            self.tabs.setCurrentWidget(self.run_tab)

    def _open_recent(self, path: str) -> None:
        if self.run_tab.manager.running:
            return
        if self.run_tab.load(path):
            self.tabs.setCurrentWidget(self.run_tab)

    def _project_loaded(self, project) -> None:
        self.quality_tab.set_project(project)
        self.tune_tab.show_profile(self.run_tab.current_profile())
        path = str(project.config_path)
        recent = [path] + [p for p in self._recent() if Path(p) != Path(path)]
        self.store["recent_projects"] = recent[:MAX_RECENT]
        self.welcome.set_recent(self.store["recent_projects"])

    def _run_finished(self, folders: list) -> None:
        self.quality_tab.video_paths.update(
            {Path(v.path).stem: v.path for v in self.run_tab.videos})
        if folders:
            self.quality_tab.set_folders(folders, auto=True)

    def closeEvent(self, ev) -> None:
        self.monitor.stop()
        self.tune_tab.shutdown()
        self.run_tab.shutdown()
        super().closeEvent(ev)


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(T.STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()
