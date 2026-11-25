"""The Run tab: project, videos, run options and the live dashboard."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog,
                               QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import hardware, settings
from ..calibration import estimate_speedup
from ..engine import RunConfig, RunManager, VideoItem
from ..project import (VIDEO_EXTENSIONS, ModelInfo, ProjectInfo, is_dlc_output_video,
                       list_videos, load_project)
from . import theme as T
from .widgets import (BodypartBars, Card, Confetti, MeterBar, RunnerBar, Sparkline,
                      StatTile, muted)

COLS = ["Video", "Length", "Frames", "Size", "Status"]


def probe_video(path: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {}
        return dict(
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


class RunTab(QWidget):
    probed = Signal(str, dict)
    tune_requested = Signal()
    run_finished = Signal(list)  # folders that now contain results
    project_loaded = Signal(object)
    log = Signal(str)

    def __init__(self, hw: hardware.Hardware, store: dict, parent=None):
        super().__init__(parent)
        self.hw, self.store = hw, store
        self.python_exe = str(store.get("python_exe") or "")
        self.project: ProjectInfo | None = None
        self.videos: list[VideoItem] = []
        self.rows: dict[str, int] = {}
        self.manager = RunManager(self)
        self.gpu_load: hardware.GpuLoad | None = None
        self.gpu_baseline_mb = 0
        self.worker_rows: list[tuple[QLabel, QProgressBar]] = []
        self.setAcceptDrops(True)
        self._build()
        self.probed.connect(self._on_probed)
        self.manager.log.connect(self.log)
        self.manager.video_changed.connect(self._update_row)
        self.manager.finished.connect(self._on_finished)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh)
        self._restore()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # ---------------- left: the three steps
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(10)

        c1 = Card("Pick your DLC project", "1")
        row = QHBoxLayout()
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("Project folder or config.yaml…")
        self.config_edit.returnPressed.connect(lambda: self.load(self.config_edit.text()))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_project)
        row.addWidget(self.config_edit, 1)
        row.addWidget(browse)
        c1.add(row)
        self.project_info = muted("No project loaded yet.")
        c1.add(self.project_info)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.snap_combo = QComboBox()
        grid.addWidget(QLabel("Model"), 0, 0)
        grid.addWidget(self.model_combo, 0, 1)
        grid.addWidget(QLabel("Snapshot"), 1, 0)
        grid.addWidget(self.snap_combo, 1, 1)
        c1.add(grid)
        lv.addWidget(c1)

        c2 = Card("Choose videos", "2")
        row = QHBoxLayout()
        for text, slot in (("Add videos…", self._add_files), ("Add folder…", self._add_folder),
                           ("Remove", self._remove_selected), ("Clear", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        c2.add(row)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(4, 160)
        self.table.setMinimumHeight(170)
        c2.add(self.table, 1)
        self.video_summary = muted("Drop videos or folders anywhere on this tab.")
        c2.add(self.video_summary)
        lv.addWidget(c2, 1)

        c3 = Card("Tune the run", "3")
        tune = QPushButton("Auto-tune…")
        tune.setObjectName("accent")
        tune.setToolTip("Time your model on this computer and pick the fastest settings that fit in memory")
        tune.clicked.connect(self.tune_requested)
        c3.header.addWidget(tune)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setToolTip("Frames sent to the GPU at once. Larger batches are faster until the GPU saturates.")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setToolTip("Videos analyzed in parallel, each in its own process.")
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU if available)", "auto")
        for gpu in self.hw.gpus:
            self.device_combo.addItem(f"GPU {gpu.index} · {gpu.name}", f"cuda:{gpu.index}")
        self.device_combo.addItem("CPU only", "cpu")
        self.exist_combo = QComboBox()
        self.exist_combo.addItem("Skip videos already analyzed", False)
        self.exist_combo.addItem("Re-analyze and overwrite", True)
        g.addWidget(QLabel("Batch size"), 0, 0)
        g.addWidget(self.batch_spin, 0, 1)
        g.addWidget(QLabel("Parallel workers"), 0, 2)
        g.addWidget(self.workers_spin, 0, 3)
        g.addWidget(QLabel("Device"), 1, 0)
        g.addWidget(self.device_combo, 1, 1)
        g.addWidget(QLabel("Existing results"), 1, 2)
        g.addWidget(self.exist_combo, 1, 3)
        c3.add(g)
        row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Next to each video (DLC default)")
        ob = QPushButton("Output folder…")
        ob.clicked.connect(self._browse_out)
        row.addWidget(QLabel("Save to"))
        row.addWidget(self.out_edit, 1)
        row.addWidget(ob)
        c3.add(row)
        g2 = QGridLayout()
        self.turbo_check = QCheckBox("Prepare frames on a second thread")
        self.turbo_check.setToolTip(
            "Decode and preprocess frames on a background thread while the GPU works.\n"
            "Only has an effect before DeepLabCut 3.0.2, which does this itself.")
        self.amp_check = QCheckBox("Half precision (GPU)")
        self.amp_check.setToolTip(
            "Run the network in 16-bit floats. Auto-tune measures the effect on speed.\n"
            "Predictions can differ by a fraction of a pixel.")
        self.csv_check = QCheckBox("Also save .csv")
        self.longest_check = QCheckBox("Longest videos first")
        self.longest_check.setToolTip("Keeps all workers busy until the end of the run.")
        self.crop_check = QCheckBox("Use cropping from config.yaml")
        self.dynamic_check = QCheckBox("Dynamic cropping (batch size 1)")
        self.dynamic_check.setToolTip(
            "DLC's dynamic cropping for single-animal bottom-up models.\n"
            "DLC forces batch size 1 in this mode, so it is usually slower on a GPU.")
        for i, w in enumerate((self.turbo_check, self.amp_check, self.csv_check,
                               self.longest_check, self.crop_check, self.dynamic_check)):
            g2.addWidget(w, i // 2, i % 2)
        c3.add(g2)
        self.estimate = QLabel("")
        self.estimate.setObjectName("boost")
        self.estimate.setWordWrap(True)
        c3.add(self.estimate)
        for w in (self.batch_spin, self.workers_spin):
            w.valueChanged.connect(self._update_estimate)
        self.amp_check.toggled.connect(self._update_estimate)
        self.turbo_check.toggled.connect(self._update_estimate)
        lv.addWidget(c3)

        row = QHBoxLayout()
        self.start_btn = QPushButton("Start analysis")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self._stop_clicked)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.start_btn, 1)
        row.addWidget(self.stop_btn)
        lv.addLayout(row)

        for combo in (self.model_combo, self.snap_combo, self.device_combo, self.exist_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self.project_info.setMinimumWidth(100)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(left)
        scroll.setMinimumWidth(480)
        split.addWidget(scroll)

        # ---------------- right: live dashboard
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(10)

        top = Card()
        self.headline = QLabel("Ready")
        self.headline.setObjectName("cardTitle")
        self.headline.setWordWrap(True)
        top.add(self.headline)
        self.runner = RunnerBar()
        top.add(self.runner)
        self.subline = muted("Load a project, add videos and press Start.")
        top.add(self.subline)
        rv.addWidget(top)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.t_elapsed = StatTile("RUNTIME", "0:00")
        self.t_eta = StatTile("TIME LEFT", "—", T.YELLOW)
        self.t_speed = StatTile("SPEED", "—", T.CYAN)
        self.t_videos = StatTile("VIDEOS", "0 / 0", T.GREEN)
        self.t_frames = StatTile("FRAMES", "0")
        for t in (self.t_elapsed, self.t_eta, self.t_speed, self.t_videos, self.t_frames):
            tiles.addWidget(t, 1)
        rv.addLayout(tiles)

        mid = QHBoxLayout()
        mid.setSpacing(10)
        sc = Card("Speed")
        self.spark = Sparkline()
        sc.add(self.spark, 1)
        mid.addWidget(sc, 3)
        sysc = Card("This computer")
        self.m_gpu = MeterBar("GPU", T.PINK)
        self.m_vram = MeterBar("GPU memory", T.PURPLE)
        self.m_cpu = MeterBar("CPU", T.CYAN)
        self.m_ram = MeterBar("RAM", T.GREEN)
        for m in (self.m_gpu, self.m_vram, self.m_cpu, self.m_ram):
            sysc.add(m)
        self.sys_note = muted("")
        sysc.add(self.sys_note)
        mid.addWidget(sysc, 2)
        rv.addLayout(mid)

        bot = QHBoxLayout()
        bot.setSpacing(10)
        qcard = Card("Tracking quality so far")
        qt = QHBoxLayout()
        self.t_conf = StatTile("KEYPOINTS CONFIDENT", "—", T.GREEN)
        self.t_mean = StatTile("MEAN CONFIDENCE", "—")
        self.t_lost = StatTile("ANIMAL LOST", "—")
        for t in (self.t_conf, self.t_mean, self.t_lost):
            qt.addWidget(t, 1)
        qcard.add(qt)
        self.bars = BodypartBars()
        qcard.add(self.bars, 1)
        bot.addWidget(qcard, 3)
        wcard = Card("Workers")
        self.worker_box = QVBoxLayout()
        self.worker_box.setSpacing(6)
        wcard.add(self.worker_box)
        self.worker_hint = muted("Workers appear here once a run starts.")
        self.worker_box.addWidget(self.worker_hint)
        wcard.outer.addStretch(1)
        bot.addWidget(wcard, 2)
        rv.addLayout(bot, 1)

        rscroll = QScrollArea()
        rscroll.setWidgetResizable(True)
        rscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rscroll.setWidget(right)
        split.addWidget(rscroll)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 6)
        split.setSizes([600, 840])
        self.confetti = Confetti(self)

    # ------------------------------------------------------------ settings
    def _restore(self) -> None:
        d = hardware.default_settings(self.hw)
        opts = self.store.get("options", {})
        self.batch_spin.setValue(int(opts.get("batch_size", d["batch_size"])))
        self.workers_spin.setValue(int(opts.get("workers", d["workers"])))
        self.turbo_check.setChecked(bool(opts.get("prefetch", True)))
        self.amp_check.setChecked(bool(opts.get("amp", False)))
        self.csv_check.setChecked(bool(opts.get("save_as_csv", False)))
        self.longest_check.setChecked(bool(opts.get("longest_first", True)))
        self.out_edit.setText(opts.get("destfolder", ""))
        i = self.device_combo.findData(opts.get("device", d["device"]))
        self.device_combo.setCurrentIndex(max(0, i))
        self.crop_check.setEnabled(False)
        self.dynamic_check.setEnabled(False)
        last = self.store.get("last_project")
        if last and Path(last).exists():
            self.load(last, quiet=True)
        self._update_estimate()

    def _save_options(self) -> None:
        self.store["options"] = dict(
            batch_size=self.batch_spin.value(), workers=self.workers_spin.value(),
            prefetch=self.turbo_check.isChecked(), amp=self.amp_check.isChecked(),
            save_as_csv=self.csv_check.isChecked(), longest_first=self.longest_check.isChecked(),
            destfolder=self.out_edit.text().strip(), device=self.device_combo.currentData(),
        )
        if self.project:
            self.store["last_project"] = str(self.project.config_path)
        settings.save(self.store)

    # ------------------------------------------------------------- project
    def browse_project(self) -> None:
        start = self.config_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select the project's config.yaml", start,
                                              "DLC config (config.yaml *.yaml)")
        if path:
            self.load(path)

    def load(self, path: str, quiet: bool = False) -> bool:
        try:
            proj = load_project(path.strip().strip('"'))
        except Exception as exc:
            if not quiet:
                QMessageBox.warning(self, "Could not load project", str(exc))
            return False
        self.project = proj
        self.config_edit.setText(str(proj.config_path))
        kind = "multi-animal" if proj.multianimal else "single animal"
        self.project_info.setText(
            f"<b>{proj.task}</b> · {proj.scorer} · {kind} · {len(proj.bodyparts)} keypoints: "
            f"{', '.join(proj.bodyparts[:8])}{'…' if len(proj.bodyparts) > 8 else ''}"
        )
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in proj.models:
            self.model_combo.addItem(m.label, m)
        self.model_combo.blockSignals(False)
        if not proj.models:
            self.project_info.setText(self.project_info.text()
                                      + f"<br><span style='color:{T.RED}'>No trained model "
                                        "snapshots found in this project.</span>")
        self._model_changed()
        self.crop_check.setEnabled(proj.cropping is not None)
        self.crop_check.setChecked(proj.cropping is not None)
        if proj.cropping:
            self.crop_check.setText(f"Use cropping from config.yaml {proj.cropping}")
        self.project_loaded.emit(proj)
        self._update_estimate()
        self.log.emit(f"Loaded project {proj.config_path}")
        return True

    def current_model(self) -> ModelInfo | None:
        return self.model_combo.currentData()

    def _model_changed(self) -> None:
        m = self.current_model()
        self.snap_combo.clear()
        if not m or not self.project:
            return
        idx = self.project.snapshot_index
        default = "?"
        if isinstance(idx, int) and m.snapshots and -len(m.snapshots) <= idx < len(m.snapshots):
            default = m.snapshots[idx]
        self.snap_combo.addItem(f"Project default → {default}", None)
        for i, name in enumerate(m.snapshots):
            self.snap_combo.addItem(name, i)
        can_dyn = m.engine == "tensorflow" or (m.method == "bu" and not self.project.multianimal)
        self.dynamic_check.setEnabled(can_dyn)
        if not can_dyn:
            self.dynamic_check.setChecked(False)
        self.amp_check.setEnabled(m.engine == "pytorch")
        self.turbo_check.setEnabled(m.engine == "pytorch")
        self._update_estimate()

    # -------------------------------------------------------------- videos
    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        vids = []
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                if (pp / "config.yaml").is_file() and not self.project:
                    self.load(str(pp))
                vids += [str(v) for v in list_videos(pp)]
            elif pp.name == "config.yaml":
                self.load(str(pp))
            elif pp.suffix.lower() in VIDEO_EXTENSIONS:
                vids.append(str(pp))
        self.add_videos(vids)

    def _start_dir(self) -> str:
        if self.videos:
            return str(Path(self.videos[-1].path).parent)
        if self.project and (self.project.project_path / "videos").is_dir():
            return str(self.project.project_path / "videos")
        return str(Path.home())

    def _add_files(self) -> None:
        exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(self, "Select videos", self._start_dir(),
                                                f"Videos ({exts});;All files (*)")
        self.add_videos(paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of videos",
                                                  self._start_dir())
        if folder:
            self.add_videos([str(v) for v in list_videos(folder)])

    def add_videos(self, paths: list[str]) -> None:
        if self.manager.running:
            return
        new = []
        known = {v.path for v in self.videos}
        for p in paths:
            p = str(Path(p))
            if p in known or is_dlc_output_video(Path(p)):
                continue
            known.add(p)
            item = VideoItem(path=p)
            self.videos.append(item)
            new.append(item)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.rows[p] = row
            name = QTableWidgetItem(item.name)
            name.setToolTip(p)
            self.table.setItem(row, 0, name)
            for c in range(1, len(COLS)):
                cell = QTableWidgetItem("…" if c < 4 else "queued")
                cell.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, c, cell)
        if new:
            threading.Thread(target=self._probe_all, args=([v.path for v in new],),
                             daemon=True).start()
        self._summarize()

    def _probe_all(self, paths: list[str]) -> None:
        for p in paths:
            try:
                info = probe_video(p)
            except Exception:
                info = {}
            self.probed.emit(p, info)

    def _on_probed(self, path: str, info: dict) -> None:
        item = next((v for v in self.videos if v.path == path), None)
        row = self.rows.get(path)
        if item is None or row is None:
            return
        if not info:
            self.table.item(row, 4).setText("unreadable?")
            return
        item.n_frames, item.fps = info["n_frames"], info["fps"]
        item.width, item.height = info["width"], info["height"]
        self.table.item(row, 1).setText(T.fmt_duration(item.duration))
        self.table.item(row, 2).setText(f"{item.n_frames:,}")
        self.table.item(row, 3).setText(f"{item.width}×{item.height}")
        self._summarize()
        self._update_estimate()

    def _remove_selected(self) -> None:
        if self.manager.running:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            path = self.table.item(r, 0).toolTip()
            self.videos = [v for v in self.videos if v.path != path]
            self.table.removeRow(r)
        self.rows = {self.table.item(r, 0).toolTip(): r for r in range(self.table.rowCount())}
        self._summarize()

    def _clear(self) -> None:
        if self.manager.running:
            return
        self.videos, self.rows = [], {}
        self.table.setRowCount(0)
        self._summarize()

    def _summarize(self) -> None:
        if not self.videos:
            self.video_summary.setText("Drop videos or folders anywhere on this tab.")
            return
        frames = sum(v.n_frames for v in self.videos)
        dur = sum(v.duration for v in self.videos)
        self.video_summary.setText(
            f"{len(self.videos)} videos · {T.fmt_int(frames)} frames · "
            f"{T.fmt_duration(dur)} of footage"
        )

    def _update_row(self, path: str) -> None:
        row = self.rows.get(path)
        v = self.manager.by_path.get(path)
        if row is None or v is None:
            return
        cell = self.table.item(row, 4)
        color = T.MUTED
        if v.status == "queued":
            text = "queued" if not v.message else v.message
        elif v.status == "loading":
            text, color = "loading model…", T.PURPLE
        elif v.status == "running":
            pct = 100 * v.frames_done / v.n_frames if v.n_frames else 0
            text, color = f"▶ {pct:.0f}% · {v.speed:.0f} fps", T.CYAN
            if v.frames_done == 0 and v.message:
                text = v.message
        elif v.status == "saving":
            text, color = "saving…", T.CYAN
        elif v.status == "done":
            conf = sum(v.frac_conf) / len(v.frac_conf) if v.frac_conf else None
            text = "✔ done" + (f" · {conf:.0%} conf." if conf is not None else "")
            color = T.GREEN if conf is None or conf >= 0.85 else T.YELLOW
        elif v.status == "skipped":
            text, color = "↷ already analyzed", T.MUTED
        else:
            text, color = "✖ " + (v.message or "error")[:40], T.RED
            cell.setToolTip(v.message)
        cell.setText(text)
        cell.setForeground(QColor(color))
        if v.n_frames:
            self.table.item(row, 2).setText(f"{v.n_frames:,}")

    # ------------------------------------------------------------ estimate
    def profile_key(self) -> str | None:
        m = self.current_model()
        if not m:
            return None
        dev = self.device_combo.currentData()
        gpu = ""
        if dev != "cpu" and self.hw.gpus:
            idx = int(dev.split(":")[1]) if dev.startswith("cuda:") else self.hw.gpus[0].index
            gpu = next((g.name for g in self.hw.gpus if g.index == idx), self.hw.gpus[0].name)
        v = next((v for v in self.videos if v.width), None)
        w, h = (v.width, v.height) if v else (0, 0)
        return settings.profile_key(gpu, m.net_type, w, h)

    def current_profile(self) -> dict | None:
        key = self.profile_key()
        prof = settings.get_profile(self.store, key) if key else None
        if prof is None and key:
            # fall back to any profile for the same GPU + network (other frame size)
            prefix = key.rsplit("|", 1)[0]
            for k, p in (self.store.get("profiles") or {}).items():
                if k.startswith(prefix):
                    return p
        return prof

    def _update_estimate(self) -> None:
        prof = self.current_profile()
        if not self.turbo_check.isChecked() and self.turbo_check.isEnabled():
            est = None
        else:
            est = estimate_speedup(prof, self.batch_spin.value(), self.workers_spin.value(),
                                   amp=self.amp_check.isChecked())
        if est is None:
            self.estimate.setText(
                "Speed-up vs. DLC defaults: not measured yet. Run Auto-tune to measure it."
                if prof is None else
                "Speed-up vs. DLC defaults: no estimate for these settings.")
            return
        how = "measured" if est["exact"] else "estimated"
        tail = ""
        if prof.get("shared"):
            tail = " · tuned while the GPU was busy, so likely higher when it is free"
        frames = sum(v.n_frames for v in self.videos)
        time_txt = ""
        if frames and est["fps"] > 0:
            time_txt = (f" · this batch ≈ {T.fmt_duration(frames / est['fps'])} "
                        f"instead of {T.fmt_duration(frames / est['baseline_fps'])}")
        self.estimate.setText(
            f"≈ {est['speedup']:.1f}× faster than DLC defaults "
            f"({est['fps']:.0f} vs {est['baseline_fps']:.0f} fps, {how}){time_txt}{tail}"
        )

    def apply_profile(self, profile: dict, use_amp: bool = False) -> None:
        batch = profile["batch_size"]
        if use_amp and profile.get("amp_batch"):
            batch = profile["amp_batch"]
        self.batch_spin.setValue(int(batch))
        self.workers_spin.setValue(int(profile["workers"]))
        self.turbo_check.setChecked(True)
        self.amp_check.setChecked(bool(use_amp))
        self._update_estimate()
        self._save_options()

    # ----------------------------------------------------------------- run
    def build_config(self) -> RunConfig | None:
        m = self.current_model()
        if not self.project or not m:
            return None
        dyn = self.dynamic_check.isChecked() and self.dynamic_check.isEnabled()
        return RunConfig(
            config=str(self.project.config_path),
            engine=m.engine,
            shuffle=m.shuffle,
            trainingsetindex=m.trainingset_index,
            snapshot_index=self.snap_combo.currentData(),
            device=self.device_combo.currentData(),
            batch_size=self.batch_spin.value(),
            workers=self.workers_spin.value(),
            prefetch=self.turbo_check.isChecked(),
            amp=self.amp_check.isChecked() and self.amp_check.isEnabled(),
            destfolder=self.out_edit.text().strip(),
            save_as_csv=self.csv_check.isChecked(),
            overwrite=bool(self.exist_combo.currentData()),
            dynamic=(dyn, 0.5, 10),
            cropping=self.project.cropping if self.crop_check.isChecked() else None,
            pcutoff=self.project.pcutoff,
            longest_first=self.longest_check.isChecked(),
            python_exe=self.python_exe,
        )

    def dlc_ready(self) -> bool:
        if self.python_exe:
            return True
        QMessageBox.information(
            self, "DeepLabCut not found",
            "Choose your DeepLabCut environment on the Welcome tab first.")
        return False

    def _browse_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self._start_dir())
        if folder:
            self.out_edit.setText(folder)

    def start(self) -> None:
        if self.manager.running or not self.dlc_ready():
            return
        cfg = self.build_config()
        if cfg is None:
            QMessageBox.information(self, "No project loaded", "Load a DLC project with a trained model first.")
            return
        if not self.videos:
            QMessageBox.information(self, "No videos", "Add at least one video to analyze.")
            return
        if cfg.overwrite:
            ok = QMessageBox.question(
                self, "Overwrite existing results?",
                "Existing DLC result files for these videos will be replaced. Continue?")
            if ok != QMessageBox.Yes:
                return
        load = self.gpu_load
        self.gpu_baseline_mb = load.used_mb if load else 0
        if cfg.device != "cpu" and hardware.gpu_is_shared(load):
            self.log.emit(
                f"Another program is using the GPU ({load.used_mb / 1024:.1f} GB, "
                f"{load.util}% busy). The run will share it and may be slower than tuned."
            )
            prof = self.current_profile()
            need = 0
            for t in (prof or {}).get("trials", []):
                if t["batch_size"] == cfg.batch_size and t["workers"] == cfg.workers:
                    need = t.get("vram_added_mb", 0)
            if need and need > load.free_mb - 1024:
                ok = QMessageBox.question(
                    self, "GPU memory is tight",
                    f"These settings need about {need / 1024:.1f} GB of GPU memory but only "
                    f"{load.free_mb / 1024:.1f} GB is free because another program is using "
                    "the GPU. Starting could crash that program or this run.\n\nStart anyway?")
                if ok != QMessageBox.Yes:
                    return
        self._save_options()
        for v in self.videos:
            row = self.rows.get(v.path)
            if row is not None:
                self.table.item(row, 4).setText("queued")
                self.table.item(row, 4).setForeground(QColor(T.MUTED))
        self._build_worker_rows(min(cfg.workers, len(self.videos)))
        self.manager.start(cfg, self.videos)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.headline.setText(f"Analyzing video 1 of {len(self.videos)}")
        self.subline.setText(
            f"{len(self.videos)} videos · batch {cfg.batch_size} · {cfg.workers} worker(s)")
        self.runner.set_state(0.0, True)
        self._timer.start()

    def _stop_clicked(self) -> None:
        if not self.manager.running:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Stop the run?")
        box.setText("Videos already finished are kept. What about the ones in progress?")
        finish = box.addButton("Finish current videos", QMessageBox.AcceptRole)
        now = box.addButton("Stop now", QMessageBox.DestructiveRole)
        box.addButton("Keep running", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is finish:
            self.manager.stop(graceful=True)
            self.headline.setText("Finishing the videos in progress…")
        elif box.clickedButton() is now:
            self.manager.stop(graceful=False)

    def shutdown(self) -> None:
        if self.manager.running:
            self.manager.stop(graceful=False)
        self._save_options()

    def _build_worker_rows(self, n: int) -> None:
        while self.worker_box.count():
            w = self.worker_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.worker_rows = []
        for i in range(n):
            lab = QLabel(f"Worker {i + 1} · starting…")
            lab.setObjectName("muted")
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            self.worker_box.addWidget(lab)
            self.worker_box.addWidget(bar)
            self.worker_rows.append((lab, bar))

    # ----------------------------------------------------------- dashboard
    def on_hw_sample(self, data: dict) -> None:
        gpus = data.get("gpus") or []
        dev = self.device_combo.currentData() or "auto"
        idx = int(dev.split(":")[1]) if dev.startswith("cuda:") else 0
        load = next((g for g in gpus if g.index == idx), gpus[0] if gpus else None)
        self.gpu_load = load
        if load:
            self.m_gpu.set(load.util / 100, f"{load.util}%"
                           + (f" · {load.temp}°C" if load.temp else ""))
            marker = None
            if self.manager.running and self.gpu_baseline_mb:
                marker = self.gpu_baseline_mb / load.total_mb
            self.m_vram.set(load.used_mb / load.total_mb,
                            f"{load.used_mb / 1024:.1f} / {load.total_mb / 1024:.0f} GB", marker)
            if self.manager.running and marker is not None:
                self.sys_note.setText("White mark = GPU memory already in use by other "
                                      "programs when the run started.")
            elif hardware.gpu_is_shared(load):
                self.sys_note.setText("Another program is using the GPU right now.")
            else:
                self.sys_note.setText("")
        else:
            self.m_gpu.set(0, "no GPU")
            self.m_vram.set(0, "—")
        self.m_cpu.set(data.get("cpu", 0) / 100, f"{data.get('cpu', 0):.0f}%")
        self.m_ram.set(data.get("ram", 0) / 100, f"{data.get('ram', 0):.0f}%")

    def _refresh(self) -> None:
        s = self.manager.snapshot()
        c = s["counts"]
        total = len(self.manager.videos)
        finished = c["done"] + c["skipped"] + c["error"]
        if s["running"] and not self.manager.stopping:
            self.headline.setText(f"Analyzing video {min(finished + 1, total)} of {total}")
        self.runner.set_state(s["fraction"], s["running"], not s["running"] and finished == total
                              and total > 0)
        self.t_elapsed.set(T.fmt_duration(s["elapsed"]))
        eta = s["eta"]
        if s["running"]:
            if eta is None:
                self.t_eta.set("…", "estimating")
            else:
                done_at = time.strftime("%H:%M", time.localtime(time.time() + eta))
                self.t_eta.set(T.fmt_duration(eta), f"done around {done_at}")
        vfps = [v.fps for v in self.manager.videos if v.fps > 0]
        rt = s["speed"] / (sum(vfps) / len(vfps)) if vfps else 0
        self.t_speed.set(f"{s['speed']:.0f} fps", f"{rt:.1f}× real time" if rt else "")
        extra = []
        if c["active"]:
            extra.append(f"{c['active']} running")
        if c["skipped"]:
            extra.append(f"{c['skipped']} skipped")
        if c["error"]:
            extra.append(f"{c['error']} failed")
        self.t_videos.set(f"{finished} / {total}", " · ".join(extra))
        self.t_frames.set(T.fmt_int(s["frames_done"]), f"of {T.fmt_int(s['frames_total'])}")
        self.spark.set_series(s["speed_series"])
        conf = s["confidence"]
        if conf:
            self.t_conf.set(f"{conf['overall_frac']:.1%}",
                            f"above p-cutoff {self.manager.cfg.pcutoff:g}")
            self.t_mean.set(f"{conf['overall_mean']:.2f}")
            self.t_lost.set(f"{conf['lost']:.2%}", "of frames")
            names = self.project.bodyparts if self.project else []
            if len(names) != len(conf["frac"]):
                names = [f"kp {i + 1}" for i in range(len(conf["frac"]))]
            self.bars.set_data(names, conf["frac"], conf["mean"])
        for w, (lab, bar) in zip(self.manager.workers, self.worker_rows):
            v = self.manager.by_path.get(w.current or "")
            if v is not None:
                pct = v.frames_done / v.n_frames if v.n_frames else 0
                state = {"loading": "loading model", "saving": "saving"}.get(
                    v.status, f"{pct:.0%} · {v.speed:.0f} fps")
                lab.setText(f"Worker {w.index + 1} · {v.name} · {state}")
                bar.setValue(int(pct * 1000))
            elif not w.ready and w.alive():
                lab.setText(f"Worker {w.index + 1} · starting DeepLabCut…")
            else:
                lab.setText(f"Worker {w.index + 1} · idle")
                bar.setValue(0 if self.manager.running else 1000)

    def _on_finished(self, summary: dict) -> None:
        self._timer.stop()
        self._refresh()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        c = summary["counts"]
        done, err, skipped = c["done"], c["error"], c["skipped"]
        parts = [f"{done} video{'s' if done != 1 else ''} analyzed"]
        if skipped:
            parts.append(f"{skipped} already done")
        if err:
            parts.append(f"{err} failed")
        self.t_eta.set("0:00", "finished " + time.strftime("%H:%M"))
        if summary.get("stopped"):
            self.headline.setText("Run stopped · " + " · ".join(parts))
        elif not done and (err or not skipped):
            self.headline.setText("Nothing was analyzed. See the Log tab for details.")
        elif not done:
            self.headline.setText("Nothing to do · every video was already analyzed")
        else:
            self.headline.setText("Finished · " + " · ".join(parts))
            if done:
                self.confetti.burst()
        line = (f"{T.fmt_int(summary['frames_done'])} frames in "
                f"{T.fmt_duration(summary['elapsed'])} · average "
                f"{summary['avg_speed']:.0f} fps")
        if summary.get("realtime_factor"):
            line += f" · {summary['realtime_factor']:.1f}× real time"
        prof = self.current_profile()
        if done and prof and prof.get("baseline_fps") and summary["avg_speed"] > 0:
            base = float(prof["baseline_fps"])
            boost = summary["avg_speed"] / base
            saved = summary["frames_done"] / base - summary["elapsed"]
            line += f"<br>≈ {boost:.1f}× faster than DLC defaults"
            if saved > 30:
                line += f", about {T.fmt_duration(saved)} saved"
        elif done:
            line += "<br>Run Auto-tune once to see how this compares with DLC defaults."
        self.subline.setText(line)
        folders = sorted({self.manager.cfg.destfolder or str(Path(v.path).parent)
                          for v in self.manager.videos if v.status in ("done", "skipped")})
        self.log.emit("Run finished: " + " · ".join(parts) + " · " + line.replace("<br>", " · "))
        self.run_finished.emit(folders)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.confetti.setGeometry(self.rect())
