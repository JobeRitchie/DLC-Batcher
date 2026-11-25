"""The Auto-tune tab: time the model on this machine and recommend settings."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .. import hardware, settings
from ..calibration import Calibrator, bundled_clip
from . import theme as T
from .widgets import BarChart, Card, MeterBar, muted

COLS = ["Test", "Batch", "Workers", "Speed (fps)", "vs DLC default", "GPU", "GPU memory"]


class TuneTab(QWidget):
    log = Signal(str)
    applied = Signal()

    def __init__(self, hw: hardware.Hardware, store: dict, run_tab, parent=None):
        super().__init__(parent)
        self.hw, self.store, self.run_tab = hw, store, run_tab
        self.cal = Calibrator(self)
        self.cal.progress.connect(self._on_progress)
        self.cal.trial_done.connect(self._on_trial)
        self.cal.log.connect(self.log)
        self.cal.finished.connect(self._on_finished)
        self.profile: dict | None = None
        self.profile_key: str | None = None
        self.gpu_load = None
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(10)

        hwc = Card("What this computer has")
        hwc.add(QLabel(f"CPU: {self.hw.cpu_name}"))
        hwc.add(muted(f"{self.hw.physical_cores} cores ({self.hw.logical_cores} threads) · "
                      f"{self.hw.ram_gb:.0f} GB RAM · {self.hw.os}"))
        if self.hw.gpus:
            for g in self.hw.gpus:
                hwc.add(QLabel(f"GPU: {g.name} · {g.total_mb / 1024:.0f} GB"))
        else:
            hwc.add(QLabel("No NVIDIA GPU detected. Analysis will run on the CPU."))
        d = hardware.default_settings(self.hw)
        hwc.add(muted(
            f"Starting settings for a machine like this: batch {d['batch_size']}, "
            f"{d['workers']} worker(s). They depend only on the installed hardware, not on "
            "what is currently free."))
        left.addWidget(hwc)

        now = Card("What is in use right now")
        self.m_gpu = MeterBar("GPU", T.PINK)
        self.m_vram = MeterBar("GPU memory", T.PURPLE)
        self.m_cpu = MeterBar("CPU", T.CYAN)
        for m in (self.m_gpu, self.m_vram, self.m_cpu):
            now.add(m)
        self.shared_note = muted("")
        now.add(self.shared_note)
        left.addWidget(now)

        run = Card("Run the tuner")
        run.add(muted(
            "Runs short timed tests with your selected model: first DLC's default "
            "settings as a baseline, then larger batches, then more parallel workers. "
            "Takes 2-4 minutes and writes nothing next to your videos."))
        row = QHBoxLayout()
        self.source = QComboBox()
        self.source.addItem("Use my first video (most accurate)", "video")
        self.source.addItem("Use the bundled test clip", "bundled")
        row.addWidget(QLabel("Test footage"))
        row.addWidget(self.source, 1)
        run.add(row)
        row = QHBoxLayout()
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 32)
        self.max_workers.setValue(min(8, hardware.max_reasonable_workers(self.hw)))
        row.addWidget(QLabel("Try up to"))
        row.addWidget(self.max_workers)
        row.addWidget(QLabel("parallel workers"))
        row.addStretch(1)
        run.add(row)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start auto-tune")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cal.cancel)
        row.addWidget(self.start_btn, 1)
        row.addWidget(self.cancel_btn)
        run.add(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        run.add(self.bar)
        self.status = muted("Not tuned yet for this model.")
        run.add(self.status)
        left.addWidget(run)
        left.addStretch(1)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(470)
        root.addWidget(lw, 4)

        right = QVBoxLayout()
        right.setSpacing(10)
        res = Card("Recommendation")
        row = QHBoxLayout()
        self.boost = QLabel("—")
        self.boost.setObjectName("bigBoost")
        col = QVBoxLayout()
        self.rec_title = QLabel("Run the tuner to get a recommendation")
        self.rec_title.setObjectName("cardTitle")
        self.rec_detail = muted("")
        col.addWidget(self.rec_title)
        col.addWidget(self.rec_detail)
        row.addWidget(self.boost)
        row.addSpacing(12)
        row.addLayout(col, 1)
        res.add(row)
        row = QHBoxLayout()
        self.amp_check = QCheckBox("Also turn on mixed precision")
        self.amp_check.setVisible(False)
        self.apply_btn = QPushButton("Use these settings")
        self.apply_btn.setObjectName("accent")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        row.addWidget(self.amp_check, 1)
        row.addWidget(self.apply_btn)
        res.add(row)
        self.notes = muted("")
        res.add(self.notes)
        right.addWidget(res)

        chart = Card("Speed of every test (frames per second)")
        self.chart = BarChart()
        chart.add(self.chart, 1)
        chart.add(muted("Gray = DLC default · cyan = batch size · purple = parallel workers · "
                        "pink = fine-tune · yellow = mixed precision · dashed line = DLC default"))
        right.addWidget(chart, 2)

        tc = Card("Details")
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tc.add(self.table, 1)
        right.addWidget(tc, 3)
        root.addLayout(right, 7)

    # ------------------------------------------------------------- actions
    def on_hw_sample(self, data: dict) -> None:
        gpus = data.get("gpus") or []
        load = gpus[0] if gpus else None
        self.gpu_load = load
        if load:
            self.m_gpu.set(load.util / 100, f"{load.util}%")
            self.m_vram.set(load.used_mb / load.total_mb,
                            f"{load.used_mb / 1024:.1f} / {load.total_mb / 1024:.0f} GB")
            if self.cal.running:
                pass
            elif hardware.gpu_is_shared(load):
                self.shared_note.setText(
                    "Another program is using the GPU. Tuning still works and stays within "
                    "the free memory, but the measured speeds will be lower than with an "
                    "idle GPU. Re-tune when the GPU is free.")
            else:
                self.shared_note.setText("The GPU is idle. This is a good time to tune.")
        else:
            self.m_gpu.set(0, "no GPU")
            self.m_vram.set(0, "—")
        self.m_cpu.set(data.get("cpu", 0) / 100, f"{data.get('cpu', 0):.0f}%")

    def start(self) -> None:
        if self.cal.running:
            return
        if self.run_tab.manager.running:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish first.")
            return
        if not self.run_tab.dlc_ready():
            return
        cfg = self.run_tab.build_config()
        if cfg is None:
            QMessageBox.information(self, "No project loaded",
                                    "Load a DLC project on the Run tab first.")
            return
        if cfg.engine != "pytorch":
            QMessageBox.information(
                self, "Not available",
                "Auto-tune currently supports PyTorch-engine models only.")
            return
        clip = None
        if self.source.currentData() == "video":
            v = next((v for v in self.run_tab.videos if v.n_frames > 0), None)
            if v is None:
                self.source.setCurrentIndex(1)
            else:
                clip = v.path
        if clip is None:
            clip = str(bundled_clip())
        cfg.amp = False
        cfg.dynamic = (False, 0.5, 10)
        self.profile_key = self.run_tab.profile_key()
        self.table.setRowCount(0)
        self.chart.set_items([])
        self.apply_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.boost.setText("…")
        self.rec_title.setText("Tuning in progress")
        self.rec_detail.setText("")
        self.notes.setText("")
        dev = cfg.device
        idx = int(dev.split(":")[1]) if dev.startswith("cuda:") else 0
        self.log.emit(f"Auto-tune started on {clip}")
        self.cal.start(cfg, clip, self.hw, self.run_tab.project.batch_size,
                       self.max_workers.value(), idx)

    def _on_progress(self, message: str, frac: float) -> None:
        self.status.setText(message)
        self.bar.setValue(int(frac * 1000))

    def _color(self, t: dict) -> str:
        return {"DLC default": "#7d7896", "batch size": T.CYAN, "workers": T.PURPLE,
                "fine-tune": T.PINK, "mixed precision": T.YELLOW}.get(t["label"], T.CYAN)

    def _on_trial(self, t: dict) -> None:
        self._show_trials(self.cal.trials)

    def _show_trials(self, trials: list[dict]) -> None:
        base = next((t["fps"] for t in trials if t["label"] == "DLC default"), None)
        self.table.setRowCount(len(trials))
        items = []
        for r, t in enumerate(trials):
            vals = [t["label"], str(t["batch_size"]), str(t["workers"]), f"{t['fps']:.0f}",
                    f"{t['fps'] / base:.2f}×" if base else "—", f"{t['gpu_util']}%",
                    f"+{t['vram_added_mb'] / 1024:.1f} GB"]
            for c, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                cell.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, cell)
            tag = "AMP " if t.get("amp") else ""
            items.append((f"{tag}b{t['batch_size']}\n×{t['workers']}", t["fps"], self._color(t)))
        self.chart.set_items(items, base)
        self.table.scrollToBottom()

    def _on_finished(self, profile: dict) -> None:
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if not profile:
            self.status.setText("Tuning did not complete. " + " ".join(self.cal.notes))
            self.boost.setText("—")
            self.rec_title.setText("No recommendation")
            self.bar.setValue(0)
            return
        if self.profile_key:
            settings.put_profile(self.store, self.profile_key, profile)
            settings.save(self.store)
        self.show_profile(profile)
        self.log.emit(
            f"Auto-tune result: batch {profile['batch_size']} × {profile['workers']} worker(s) "
            f"= {profile['fps']:.0f} fps, {profile['speedup']:.1f}× DLC default")

    def show_profile(self, profile: dict | None) -> None:
        self.profile = profile
        if not profile:
            return
        self._show_trials(profile.get("trials", []))
        self.boost.setText(f"{profile['speedup']:.1f}×")
        self.rec_title.setText(
            f"Batch size {profile['batch_size']} · {profile['workers']} parallel "
            f"worker{'s' if profile['workers'] != 1 else ''}")
        self.rec_detail.setText(
            f"{profile['fps']:.0f} frames/s versus {profile['baseline_fps']:.0f} with DLC's "
            f"default settings (batch {profile['baseline_batch']}, one process). "
            f"An hour of 30 fps video takes about "
            f"{T.fmt_duration(108000 / profile['fps'])} instead of "
            f"{T.fmt_duration(108000 / profile['baseline_fps'])}.")
        amp = float(profile.get("amp_fps") or 0)
        if amp > profile["fps"] * 1.05:
            self.amp_check.setVisible(True)
            self.amp_check.setText(
                f"Also turn on mixed precision"
                + (f" at batch {profile['amp_batch']}" if profile.get("amp_batch") else "")
                + f" → {amp:.0f} fps "
                f"({amp / profile['baseline_fps']:.1f}× DLC default; sub-pixel differences)")
        else:
            self.amp_check.setVisible(False)
            self.amp_check.setChecked(False)
        self.notes.setText("\n".join("• " + n for n in profile.get("notes", [])))
        self.status.setText(f"Last tuned {profile.get('date', '')} for this model.")
        self.apply_btn.setEnabled(True)
        self.bar.setValue(1000)

    def _apply(self) -> None:
        if self.profile:
            self.run_tab.apply_profile(self.profile, self.amp_check.isChecked()
                                       and self.amp_check.isVisible())
            self.applied.emit()

    def shutdown(self) -> None:
        self.cal.cancel()
