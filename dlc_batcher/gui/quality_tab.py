"""The Quality tab: which videos are problematic, and how to improve the model."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import quality
from . import theme as T
from .widgets import Card, Heatmap, Timeline, muted, qc

COLS = ["", "Video", "Score", "Confident", "Lost", "Jumps /1k", "Odd poses", "In training set"]
ICON = {"good": "✔", "check": "!", "problem": "✖"}
KP_COLORS = [T.PINK, T.CYAN, T.YELLOW, T.GREEN, T.PURPLE, T.ORANGE, "#7ab8ff", "#ff8fd0",
             "#b8ff7a", "#ffffff"]


class NumItem(QTableWidgetItem):
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self.value = value
        self.setTextAlignment(Qt.AlignCenter)

    def __lt__(self, other) -> bool:
        return self.value < getattr(other, "value", 0)


class QualityTab(QWidget):
    log = Signal(str)
    analyzed = Signal(object)  # (videos, scorer, scorers, error)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.folders: list[str] = []
        self.videos: list[quality.VideoQuality] = []
        self.video_paths: dict[str, str] = {}  # stem -> path, for results saved elsewhere
        self._again = False
        self.current: quality.VideoQuality | None = None
        self.pred = None  # (xy, lik) of the selected video
        self._busy = False
        self._build()
        self.analyzed.connect(self._on_analyzed)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(10)

        bar = QHBoxLayout()
        self.source = QLabel("No results loaded")
        self.source.setObjectName("muted")
        pick = QPushButton("Choose results folder…")
        pick.clicked.connect(self._pick_folder)
        self.scorer = QComboBox()
        self.scorer.setMinimumWidth(260)
        self.scorer.setToolTip("Which model's results to inspect")
        self.pcut = QDoubleSpinBox()
        self.pcut.setRange(0.05, 0.99)
        self.pcut.setSingleStep(0.05)
        self.pcut.setValue(0.6)
        self.pcut.setToolTip("Likelihood above which a keypoint counts as confident")
        self.go = QPushButton("Check quality")
        self.go.setObjectName("primary")
        self.go.clicked.connect(lambda: self.analyze())
        bar.addWidget(pick)
        bar.addWidget(self.source, 1)
        bar.addWidget(QLabel("Model"))
        bar.addWidget(self.scorer)
        bar.addWidget(QLabel("p-cutoff"))
        bar.addWidget(self.pcut)
        bar.addWidget(self.go)
        root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # ---- left: ranking + heatmap
        left = QSplitter(Qt.Vertical)
        rank = Card("Videos, worst first")
        self.verdict = muted("Analyze a folder of DLC results to rank the videos.")
        rank.add(self.verdict)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setMinimumSectionSize(30)
        for col, width in ((0, 42), (2, 58), (3, 78), (4, 62), (5, 76), (6, 78), (7, 100)):
            self.table.setColumnWidth(col, width)
        self.table.itemSelectionChanged.connect(self._row_selected)
        rank.add(self.table, 1)
        left.addWidget(rank)
        heat = Card("Confidence by keypoint (% of frames above p-cutoff)")
        self.heat = Heatmap()
        self.heat.row_clicked.connect(self._heat_clicked)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.heat)
        heat.add(sc, 1)
        left.addWidget(heat)
        left.setSizes([330, 330])
        split.addWidget(left)

        # ---- middle: selected video
        mid = Card("Look inside a video")
        self.sel_title = QLabel("Select a video")
        self.sel_title.setObjectName("cardTitle")
        mid.add(self.sel_title)
        self.sel_detail = muted("")
        mid.add(self.sel_detail)
        self.timeline = Timeline()
        self.timeline.frame_picked.connect(self._show_frame)
        mid.add(self.timeline)
        row = QHBoxLayout()
        worst = QPushButton("Jump to worst moment")
        worst.clicked.connect(self._goto_worst)
        prev = QPushButton("◀")
        nxt = QPushButton("▶")
        prev.clicked.connect(lambda: self._step(-1))
        nxt.clicked.connect(lambda: self._step(1))
        self.frame_label = muted("")
        row.addWidget(worst)
        row.addWidget(prev)
        row.addWidget(nxt)
        row.addStretch(1)
        mid.add(row)
        mid.add(self.frame_label)
        self.frame = QLabel("Click the timeline to see that frame with the predicted keypoints")
        self.frame.setAlignment(Qt.AlignCenter)
        self.frame.setObjectName("muted")
        self.frame.setMinimumHeight(220)
        self.frame.setFrameShape(QFrame.NoFrame)
        mid.add(self.frame, 1)
        split.addWidget(mid)

        # ---- right: advice
        adv = Card("How to improve the model")
        self.advice_box = QVBoxLayout()
        self.advice_box.setSpacing(8)
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.setContentsMargins(0, 0, 4, 0)
        hv.addLayout(self.advice_box)
        hv.addStretch(1)
        sc2 = QScrollArea()
        sc2.setWidgetResizable(True)
        sc2.setWidget(holder)
        adv.add(sc2, 1)
        self.advice_box.addWidget(muted("Advice appears here after a quality check."))
        split.addWidget(adv)
        split.setSizes([640, 440, 400])
        self._cur_frame = 0

    # ------------------------------------------------------------- loading
    def set_project(self, project) -> None:
        self.project = project
        if project is not None:
            self.pcut.setValue(project.pcutoff)
            if not self.folders and (project.project_path / "videos").is_dir():
                self.set_folders([str(project.project_path / "videos")], auto=False)

    def set_folders(self, folders: list[str], auto: bool = True) -> None:
        self.folders = [f for f in folders if f]
        if not self.folders:
            return
        text = self.folders[0] if len(self.folders) == 1 else f"{len(self.folders)} folders"
        self.source.setText(text)
        found = quality.find_results(self.folders)
        current = self.scorer.currentData()
        scorers = sorted(found, key=lambda s: -max(f.stat().st_mtime for f in found[s]))
        self.scorer.blockSignals(True)
        self.scorer.clear()
        for s in scorers:
            self.scorer.addItem(f"{s}  ({len(found[s])} videos)", s)
        if current in scorers and not auto:
            self.scorer.setCurrentIndex(scorers.index(current))
        self.scorer.blockSignals(False)
        if auto and scorers:
            self.analyze()

    def _pick_folder(self) -> None:
        start = self.folders[0] if self.folders else str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Folder with DLC result files", start)
        if folder:
            self.set_folders([folder], auto=True)

    def analyze(self) -> None:
        if not self.folders:
            return
        if self._busy:
            self._again = True  # re-run with the newest folder when this one ends
            return
        if self.scorer.count() == 0:
            self.verdict.setText("No DLC result files (*.h5) found in that folder yet.")
            return
        self._busy = True
        self.go.setEnabled(False)
        self.verdict.setText("Analyzing results…")
        args = (list(self.folders), self.pcut.value(), self.scorer.currentData(), self.project)

        def work():
            try:
                vids, sc, scs = quality.analyze_folder(args[0], args[1], args[2], args[3])
                self.analyzed.emit((vids, sc, scs, None))
            except Exception as exc:
                self.analyzed.emit(([], None, [], str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _on_analyzed(self, result) -> None:
        vids, _scorer, _scorers, err = result
        self._busy = False
        self.go.setEnabled(True)
        if self._again:
            self._again = False
            self.analyze()
            return
        if err:
            self.verdict.setText(f"Could not analyze: {err}")
            return
        self.videos = sorted(vids, key=lambda q: q.score)
        self._fill()
        self.log.emit(f"Quality check: {len(vids)} videos analyzed")

    # ------------------------------------------------------------- display
    def _fill(self) -> None:
        vids = self.videos
        self.current = None
        self.table.clearSelection()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(vids))
        for r, q in enumerate(vids):
            icon = NumItem(ICON[q.status], {"problem": 0, "check": 1, "good": 2}[q.status])
            icon.setToolTip("; ".join(q.reasons) or "looks good")
            name = QTableWidgetItem(q.video_name)
            name.setData(Qt.UserRole, q.h5_path)
            name.setToolTip("; ".join(q.reasons) or "looks good")
            score = NumItem(f"{q.score:.0f}", q.score)
            score.setForeground(QColor(T.STATUS_COLORS[q.status]))
            f = score.font()
            f.setBold(True)
            score.setFont(f)
            train = "—" if q.in_training is None else ("yes" if q.in_training else "no")
            cells = [icon, name, score,
                     NumItem(f"{q.confident:.1%}", q.confident),
                     NumItem(f"{q.lost:.2%}", q.lost),
                     NumItem(f"{q.jumps_per_1k:.1f}", q.jumps_per_1k),
                     NumItem(f"{q.odd_pose:.2%}", q.odd_pose),
                     NumItem(train, 1 if q.in_training else 0)]
            for c, cell in enumerate(cells):
                self.table.setItem(r, c, cell)
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.AscendingOrder)
        n = len(vids)
        if not n:
            self.verdict.setText("No readable result files found.")
            self.heat.set_data([], [], None)
            return
        bad = sum(q.status == "problem" for q in vids)
        chk = sum(q.status == "check" for q in vids)
        self.verdict.setText(
            f"{n - bad - chk} good · {chk} worth a look · {bad} problematic   ·   "
            f"score = confidence, jumps, odd poses and lost frames combined (0-100)")
        labels = vids[0].labels
        same = [q for q in vids if q.labels == labels]
        self.heat.set_data([q.video_name for q in same], labels,
                           np.stack([q.frac_conf for q in same]))
        self._heat_videos = same
        self._fill_advice()
        if self.table.rowCount():
            self.table.selectRow(0)

    def _fill_advice(self) -> None:
        while self.advice_box.count():
            w = self.advice_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        for a in quality.recommendations(self.videos, self.project, self.pcut.value()):
            box = QFrame()
            box.setObjectName("card")
            color = T.STATUS_COLORS.get(a.level, T.CYAN)
            box.setStyleSheet(f"QFrame#card {{ border-left: 4px solid {color}; "
                              f"background: {T.CARD2}; }}")
            lay = QVBoxLayout(box)
            lay.setContentsMargins(12, 9, 10, 10)
            lay.setSpacing(5)
            title = QLabel(a.title)
            title.setWordWrap(True)
            title.setStyleSheet("font-weight: 700;")
            lay.addWidget(title)
            body = muted(a.detail)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(body)
            if a.command:
                code = QLabel(a.command)
                code.setWordWrap(True)
                code.setTextInteractionFlags(Qt.TextSelectableByMouse)
                code.setStyleSheet(f"font-family: Consolas, monospace; font-size: 8.5pt; "
                                   f"background: {T.BG2}; border-radius: 6px; padding: 6px;")
                lay.addWidget(code)
                copy = QPushButton("Copy code")
                copy.clicked.connect(lambda _=False, c=a.command, b=copy: (
                    QGuiApplication.clipboard().setText(c), b.setText("Copied ✓")))
                lay.addWidget(copy, 0, Qt.AlignLeft)
            self.advice_box.addWidget(box)

    def _row_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        h5 = self.table.item(rows[0].row(), 1).data(Qt.UserRole)
        q = next((v for v in self.videos if v.h5_path == h5), None)
        if q is not None:
            self._select(q)

    def _heat_clicked(self, row: int) -> None:
        q = self._heat_videos[row]
        for r in range(self.table.rowCount()):
            if self.table.item(r, 1).data(Qt.UserRole) == q.h5_path:
                self.table.selectRow(r)
                break

    def _select(self, q: quality.VideoQuality) -> None:
        if not q.video_path and q.video_name in self.video_paths:
            q.video_path = self.video_paths[q.video_name]
        self.current = q
        self.pred = None
        if q in getattr(self, "_heat_videos", []):
            self.heat.set_selected(self._heat_videos.index(q))
        self.sel_title.setText(f"{ICON[q.status]}  {q.video_name}  ·  score {q.score:.0f}")
        dur = f"{q.n_frames / q.fps / 60:.1f} min" if q.fps > 0 else f"{q.n_frames} frames"
        res = f" · {q.resolution[0]}×{q.resolution[1]}" if q.resolution else ""
        why = "; ".join(q.reasons) or "nothing stands out"
        self.sel_detail.setText(f"{dur}{res} · {why}")
        self.timeline.set_data(q.timeline_conf, q.timeline_bad, q.n_frames, q.fps,
                               self.pcut.value())
        self.frame.setPixmap(QPixmap())
        self.frame.setText("Click the timeline to see that frame with the predicted keypoints"
                           if q.video_path else "The video file was not found next to the results")
        self.frame_label.setText("")

    def _goto_worst(self) -> None:
        if self.current:
            self.timeline.set_cursor_frame(self.current.worst_frame)
            self._show_frame(self.current.worst_frame)

    def _step(self, delta: int) -> None:
        if self.current:
            f = max(0, min(self.current.n_frames - 1, self._cur_frame + delta))
            self.timeline.set_cursor_frame(f)
            self._show_frame(f)

    def _show_frame(self, index: int) -> None:
        q = self.current
        if q is None or not q.video_path:
            return
        self._cur_frame = index
        try:
            import cv2

            if self.pred is None:
                xy, lik, _ = quality.load_predictions(q.h5_path)
                self.pred = (xy, lik)
            xy, lik = self.pred
            cap = cv2.VideoCapture(q.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, img = cap.read()
            cap.release()
            if not ok:
                self.frame.setText("Could not read that frame")
                return
            img = np.ascontiguousarray(img[..., ::-1])
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            scale = max(1.0, min(4.0, 640 / w))
            pix = QPixmap.fromImage(qimg).scaled(int(w * scale), int(h * scale),
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            cut = self.pcut.value()
            idx = min(index, len(xy) - 1)
            for k, label in enumerate(q.labels):
                x, y, c = float(xy[idx, k, 0]), float(xy[idx, k, 1]), float(lik[idx, k])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                col = KP_COLORS[k % len(KP_COLORS)]
                confident = c >= cut
                p.setPen(QPen(qc("#000000", 200), 3))
                p.setBrush(qc(col) if confident else Qt.NoBrush)
                if not confident:
                    p.setPen(QPen(qc(col), 2, Qt.DotLine))
                p.drawEllipse(int(x * scale) - 5, int(y * scale) - 5, 10, 10)
                p.setPen(qc("#000000"))
                p.drawText(int(x * scale) + 8, int(y * scale) + 5, f"{label} {c:.2f}")
                p.setPen(qc(col))
                p.drawText(int(x * scale) + 7, int(y * scale) + 4, f"{label} {c:.2f}")
            p.end()
            avail = self.frame.size()
            self.frame.setPixmap(pix.scaled(avail, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            stamp = f" · {index / q.fps:.1f} s" if q.fps > 0 else ""
            n_ok = int((lik[idx] >= cut).sum())
            self.frame_label.setText(f"frame {index:,}{stamp} · {n_ok}/{len(q.labels)} keypoints "
                                     "confident (dotted = below cutoff)")
        except Exception as exc:
            self.frame.setText(f"Could not show frame: {exc}")
