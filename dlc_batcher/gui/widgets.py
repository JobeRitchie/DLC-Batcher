"""Custom-painted widgets: cards, tiles, the mouse progress bar, charts."""
from __future__ import annotations

import math
import random
import threading
import time

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath,
                           QPen)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
                               QWidget)

from .. import hardware
from . import theme as T


def qc(hex_color: str, alpha: int = 255) -> QColor:
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


def lerp_color(a: str, b: str, t: float) -> QColor:
    ca, cb = QColor(a), QColor(b)
    t = max(0.0, min(1.0, t))
    return QColor(
        int(ca.red() + (cb.red() - ca.red()) * t),
        int(ca.green() + (cb.green() - ca.green()) * t),
        int(ca.blue() + (cb.blue() - ca.blue()) * t),
    )


def quality_color(frac: float) -> QColor:
    """0..1 -> red / yellow / green."""
    if frac < 0.75:
        return lerp_color(T.RED, T.YELLOW, max(0.0, (frac - 0.4) / 0.35))
    return lerp_color(T.YELLOW, T.GREEN, (frac - 0.75) / 0.25)


class Card(QFrame):
    def __init__(self, title: str = "", step: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(14, 12, 14, 14)
        self.outer.setSpacing(9)
        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        if step:
            badge = QLabel(step)
            badge.setObjectName("step")
            badge.setAlignment(Qt.AlignCenter)
            self.header.addWidget(badge)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.header.addWidget(self.title)
        self.header.addStretch(1)
        if title or step:
            self.outer.addLayout(self.header)

    def add(self, item, stretch: int = 0):
        if isinstance(item, QWidget):
            self.outer.addWidget(item, stretch)
        else:
            self.outer.addLayout(item, stretch)
        return item


def muted(text: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(wrap)
    return lab


class StatTile(QFrame):
    def __init__(self, caption: str, value: str = "—", color: str = T.TEXT, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(1)
        self.value = QLabel(value)
        self.value.setObjectName("tileValue")
        self.value.setStyleSheet(f"color: {color};")
        self.caption = QLabel(caption)
        self.caption.setObjectName("tileCaption")
        self.sub = QLabel("")
        self.sub.setObjectName("tileCaption")
        lay.addWidget(self.caption)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)

    def set(self, value: str, sub: str | None = None) -> None:
        self.value.setText(value)
        if sub is not None:
            self.sub.setText(sub)


class RunnerBar(QWidget):
    """Overall progress: a mouse runs along the bar toward the cheese."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.fraction = 0.0
        self.active = False
        self.done = False
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._animate)

    def set_state(self, fraction: float, active: bool, done: bool = False) -> None:
        self.fraction = max(0.0, min(1.0, fraction))
        self.done = done
        if active != self.active:
            self.active = active
            if active:
                self._timer.start()
            else:
                self._timer.stop()
        self.update()

    def _animate(self) -> None:
        self._phase += 0.35
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        pad, bar_h = 26, 16
        y = h - bar_h - 8
        track = QRectF(pad, y, w - 2 * pad, bar_h)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(T.BG2))
        p.drawRoundedRect(track, bar_h / 2, bar_h / 2)
        fill_w = track.width() * self.fraction
        if fill_w > 2:
            grad = QLinearGradient(track.left(), 0, track.right(), 0)
            grad.setColorAt(0.0, qc(T.PURPLE))
            grad.setColorAt(0.55, qc(T.PINK))
            grad.setColorAt(1.0, qc(T.YELLOW))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(track.left(), y, max(fill_w, bar_h), bar_h),
                              bar_h / 2, bar_h / 2)
            if self.active:  # moving shine
                p.setClipRect(QRectF(track.left(), y, fill_w, bar_h))
                sx = track.left() + ((self._phase * 14) % (track.width() + 120)) - 60
                shine = QLinearGradient(sx, 0, sx + 60, 0)
                shine.setColorAt(0, qc("#ffffff", 0))
                shine.setColorAt(0.5, qc("#ffffff", 70))
                shine.setColorAt(1, qc("#ffffff", 0))
                p.setBrush(QBrush(shine))
                p.drawRect(QRectF(sx, y, 60, bar_h))
                p.setClipping(False)
        emoji = QFont("Segoe UI Emoji", 20)
        p.setFont(emoji)
        p.setPen(qc(T.TEXT))
        if not self.done:
            p.drawText(QRectF(w - pad - 14, y - 32, 36, 32), Qt.AlignCenter, "🧀")
        bob = math.sin(self._phase) * 3 if self.active else 0
        mx = track.left() + fill_w
        mouse = "🎉" if self.done else "🐭"
        p.drawText(QRectF(mx - 18, y - 33 + bob, 36, 32), Qt.AlignCenter, mouse)
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(qc("#1a0a12") if self.fraction > 0.12 else qc(T.MUTED))
        if self.fraction > 0.12:
            p.drawText(QRectF(track.left(), y, fill_w, bar_h), Qt.AlignCenter,
                       f"{self.fraction * 100:.1f}%")
        else:
            p.drawText(QRectF(track.left() + fill_w + 6, y, 80, bar_h),
                       Qt.AlignVCenter | Qt.AlignLeft, f"{self.fraction * 100:.1f}%")
        p.end()


class Sparkline(QWidget):
    def __init__(self, color: str = T.CYAN, unit: str = "fps", parent=None):
        super().__init__(parent)
        self.color, self.unit = color, unit
        self.series: list[float] = []
        self.setMinimumHeight(70)

    def set_series(self, series: list[float]) -> None:
        self.series = list(series)
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        data = self.series[-600:]
        if len(data) < 2:
            p.setPen(qc(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "speed will appear here")
            p.end()
            return
        top = max(max(data), 1e-6) * 1.15
        left, bottom = 4, h - 14
        step = (w - left - 4) / (len(data) - 1)
        pts = [QPointF(left + i * step, bottom - (v / top) * (bottom - 6))
               for i, v in enumerate(data)]
        path = QPainterPath(pts[0])
        for pt in pts[1:]:
            path.lineTo(pt)
        area = QPainterPath(path)
        area.lineTo(pts[-1].x(), bottom)
        area.lineTo(pts[0].x(), bottom)
        grad = QLinearGradient(0, 0, 0, bottom)
        grad.setColorAt(0, qc(self.color, 110))
        grad.setColorAt(1, qc(self.color, 0))
        p.fillPath(area, QBrush(grad))
        p.setPen(QPen(qc(self.color), 2))
        p.drawPath(path)
        p.setBrush(qc(self.color))
        p.drawEllipse(pts[-1], 3.5, 3.5)
        p.setPen(qc(T.MUTED))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(left, 0, w, 14), Qt.AlignLeft, f"peak {max(data):.0f} {self.unit}")
        p.drawText(QRectF(0, h - 13, w - 4, 13), Qt.AlignRight, "time →")
        p.end()


class MeterBar(QWidget):
    """Label + horizontal meter + value text."""

    def __init__(self, label: str, color: str = T.CYAN, parent=None):
        super().__init__(parent)
        self.label, self.color = label, color
        self.fraction, self.text = 0.0, "—"
        self.marker: float | None = None
        self.setMinimumHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, fraction: float, text: str, marker: float | None = None) -> None:
        self.fraction = max(0.0, min(1.0, fraction))
        self.text, self.marker = text, marker
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        lab_w, val_w = 92, 118
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(qc(T.MUTED))
        p.drawText(QRectF(0, 0, lab_w, h), Qt.AlignVCenter | Qt.AlignLeft, self.label)
        track = QRectF(lab_w, h / 2 - 5, max(10, w - lab_w - val_w - 8), 10)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(T.BG2))
        p.drawRoundedRect(track, 5, 5)
        if self.fraction > 0:
            p.setBrush(qc(self.color))
            p.drawRoundedRect(QRectF(track.left(), track.top(),
                                     max(10, track.width() * self.fraction), 10), 5, 5)
        if self.marker is not None:
            mx = track.left() + track.width() * max(0.0, min(1.0, self.marker))
            p.setPen(QPen(qc(T.TEXT, 200), 2))
            p.drawLine(QPointF(mx, track.top() - 3), QPointF(mx, track.bottom() + 3))
        p.setPen(qc(T.TEXT))
        p.drawText(QRectF(w - val_w, 0, val_w, h), Qt.AlignVCenter | Qt.AlignRight, self.text)
        p.end()


class BodypartBars(QWidget):
    """Per-keypoint confidence: bar = % of frames above p-cutoff."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.names: list[str] = []
        self.frac: list[float] = []
        self.mean: list[float] = []
        self.setMinimumHeight(60)

    def set_data(self, names: list[str], frac: list[float], mean: list[float]) -> None:
        self.names, self.frac, self.mean = list(names), list(frac), list(mean)
        self.setMinimumHeight(max(60, 21 * len(self.names) + 6))
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        if not self.names:
            p.setPen(qc(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "keypoint confidence will appear here")
            p.end()
            return
        p.setFont(QFont("Segoe UI", 9))
        lab_w = min(150, max(p.fontMetrics().horizontalAdvance(n) for n in self.names) + 10)
        row = 21
        for i, name in enumerate(self.names):
            y = 3 + i * row
            f = self.frac[i] if i < len(self.frac) else 0.0
            p.setPen(qc(T.MUTED))
            p.drawText(QRectF(0, y, lab_w, row - 3), Qt.AlignVCenter | Qt.AlignLeft, name)
            track = QRectF(lab_w, y + 4, max(10, w - lab_w - 52), row - 11)
            p.setPen(Qt.NoPen)
            p.setBrush(qc(T.BG2))
            p.drawRoundedRect(track, 5, 5)
            p.setBrush(quality_color(f))
            p.drawRoundedRect(QRectF(track.left(), track.top(),
                                     max(8, track.width() * f), track.height()), 5, 5)
            p.setPen(qc(T.TEXT))
            p.drawText(QRectF(w - 50, y, 50, row - 3), Qt.AlignVCenter | Qt.AlignRight,
                       f"{f * 100:.1f}%")
        p.end()


class Confetti(QWidget):
    """Short celebratory overlay; ignores mouse input."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.parts: list[list] = []
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def burst(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        colors = [T.PINK, T.CYAN, T.YELLOW, T.GREEN, T.PURPLE, T.ORANGE]
        w = self.width()
        self.parts = [
            [random.uniform(0, w), random.uniform(-200, -10), random.uniform(-1.5, 1.5),
             random.uniform(2.5, 7), random.choice(colors), random.uniform(0, 6.3),
             random.uniform(5, 11)]
            for _ in range(170)
        ]
        self._t0 = time.time()
        self.show()
        self.raise_()
        self._timer.start()

    def _tick(self) -> None:
        for pt in self.parts:
            pt[0] += pt[2]
            pt[1] += pt[3]
            pt[3] += 0.08
            pt[5] += 0.2
        if time.time() - self._t0 > 4.0:
            self._timer.stop()
            self.hide()
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        fade = max(0.0, min(1.0, 4.0 - (time.time() - self._t0)))
        p.setPen(Qt.NoPen)
        for x, y, _vx, _vy, col, rot, size in self.parts:
            p.save()
            p.translate(x, y)
            p.rotate(math.degrees(rot))
            p.setBrush(qc(col, int(230 * fade)))
            p.drawRoundedRect(QRectF(-size / 2, -size / 4, size, size / 2), 1.5, 1.5)
            p.restore()
        p.end()


class BarChart(QWidget):
    """Labeled bar chart (used for calibration trials)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[tuple[str, float, str]] = []  # label, value, color
        self.baseline: float | None = None
        self.setMinimumHeight(170)

    def set_items(self, items, baseline=None) -> None:
        self.items, self.baseline = list(items), baseline
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not self.items:
            p.setPen(qc(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "trial results will appear here")
            p.end()
            return
        top = max(v for _, v, _ in self.items) * 1.18 or 1.0
        left, bottom, n = 8, h - 34, len(self.items)
        slot = (w - 2 * left) / n
        bar_w = min(54, slot * 0.68)
        p.setFont(QFont("Segoe UI", 8))
        for i, (label, value, color) in enumerate(self.items):
            cx = left + slot * (i + 0.5)
            bh = (value / top) * (bottom - 16)
            rect = QRectF(cx - bar_w / 2, bottom - bh, bar_w, bh)
            p.setPen(Qt.NoPen)
            p.setBrush(qc(color))
            p.drawRoundedRect(rect, 5, 5)
            p.setPen(qc(T.TEXT))
            p.drawText(QRectF(cx - slot / 2, bottom - bh - 16, slot, 14), Qt.AlignCenter,
                       f"{value:.0f}")
            p.setPen(qc(T.MUTED))
            p.drawText(QRectF(cx - slot / 2, bottom + 3, slot, 30),
                       Qt.AlignHCenter | Qt.AlignTop, label)
        if self.baseline:
            y = bottom - (self.baseline / top) * (bottom - 16)
            p.setPen(QPen(qc(T.MUTED), 1, Qt.DashLine))
            p.drawLine(QPointF(left, y), QPointF(w - left, y))
        p.end()


class Heatmap(QWidget):
    """Videos x keypoints, colored by % of confident frames."""

    row_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[str] = []
        self.cols: list[str] = []
        self.values = None
        self.selected = -1
        self.setMouseTracking(True)
        self.setMinimumHeight(120)

    def set_data(self, rows, cols, values) -> None:
        self.rows, self.cols, self.values = list(rows), list(cols), values
        self.setMinimumHeight(64 + 19 * len(self.rows))
        self.update()

    def set_selected(self, row: int) -> None:
        self.selected = row
        self.update()

    def _geom(self):
        fm = self.fontMetrics()
        lab_w = min(190, max((fm.horizontalAdvance(r) for r in self.rows), default=60) + 14)
        head_h = 54
        cw = max(18.0, (self.width() - lab_w - 6) / max(1, len(self.cols)))
        return lab_w, head_h, cw, 19

    def mousePressEvent(self, ev) -> None:
        if not self.rows:
            return
        _lab_w, head_h, _cw, rh = self._geom()
        row = int((ev.position().y() - head_h) // rh)
        if 0 <= row < len(self.rows):
            self.row_clicked.emit(row)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.rows or self.values is None:
            p.setPen(qc(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "analyze results to see the heatmap")
            p.end()
            return
        lab_w, head_h, cw, rh = self._geom()
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(qc(T.MUTED))
        for j, col in enumerate(self.cols):
            p.save()
            p.translate(lab_w + cw * (j + 0.5), head_h - 6)
            p.rotate(-32)
            p.drawText(QRectF(0, -12, 140, 14), Qt.AlignLeft | Qt.AlignVCenter, col)
            p.restore()
        for i, row in enumerate(self.rows):
            y = head_h + i * rh
            if i == self.selected:
                p.setPen(Qt.NoPen)
                p.setBrush(qc(T.PURPLE, 70))
                p.drawRoundedRect(QRectF(0, y, self.width(), rh), 4, 4)
            p.setPen(qc(T.TEXT if i == self.selected else T.MUTED))
            p.drawText(QRectF(4, y, lab_w - 8, rh), Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(row, Qt.ElideMiddle, lab_w - 10))
            for j in range(len(self.cols)):
                v = float(self.values[i][j])
                p.setPen(Qt.NoPen)
                p.setBrush(quality_color(v))
                p.drawRoundedRect(QRectF(lab_w + j * cw + 1.5, y + 1.5, cw - 3, rh - 3), 4, 4)
                if cw >= 34:
                    p.setPen(qc("#16121f"))
                    p.drawText(QRectF(lab_w + j * cw, y, cw, rh), Qt.AlignCenter,
                               f"{v * 100:.0f}")
        p.end()


class Timeline(QWidget):
    """Confidence over time for one video; click to pick a frame."""

    frame_picked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.conf = None
        self.bad = None
        self.n_frames = 0
        self.fps = 0.0
        self.cursor = -1
        self.pcutoff = 0.6
        self.setMinimumHeight(120)
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, conf, bad, n_frames: int, fps: float, pcutoff: float) -> None:
        self.conf, self.bad, self.n_frames, self.fps = conf, bad, n_frames, fps
        self.pcutoff = pcutoff
        self.cursor = -1
        self.update()

    def set_cursor_frame(self, frame: int) -> None:
        self.cursor = frame
        self.update()

    def _pick(self, ev) -> None:
        if self.n_frames <= 0:
            return
        frac = (ev.position().x() - 6) / max(1, self.width() - 12)
        frame = int(max(0.0, min(1.0, frac)) * (self.n_frames - 1))
        self.cursor = frame
        self.update()
        self.frame_picked.emit(frame)

    mousePressEvent = _pick

    def mouseMoveEvent(self, ev) -> None:
        if ev.buttons() & Qt.LeftButton:
            self._pick(ev)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.conf is None or len(self.conf) == 0:
            p.setPen(qc(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "select a video to see its timeline")
            p.end()
            return
        left, right, top, bottom = 6, w - 6, 8, h - 18
        n = len(self.conf)
        bw = (right - left) / n
        p.setPen(Qt.NoPen)
        for i in range(n):  # red bars = share of bad frames in this slice
            b = float(self.bad[i])
            if b > 0.005:
                p.setBrush(qc(T.RED, 60 + int(170 * min(1.0, b * 2))))
                bh = (bottom - top) * min(1.0, b)
                p.drawRect(QRectF(left + i * bw, bottom - bh, max(1.0, bw), bh))
        yc = bottom - self.pcutoff * (bottom - top)
        p.setPen(QPen(qc(T.MUTED, 150), 1, Qt.DashLine))
        p.drawLine(QPointF(left, yc), QPointF(right, yc))
        path = QPainterPath()
        for i in range(n):
            pt = QPointF(left + (i + 0.5) * bw, bottom - float(self.conf[i]) * (bottom - top))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QPen(qc(T.CYAN), 1.8))
        p.drawPath(path)
        if self.cursor >= 0 and self.n_frames > 1:
            cx = left + (self.cursor / (self.n_frames - 1)) * (right - left)
            p.setPen(QPen(qc(T.YELLOW), 2))
            p.drawLine(QPointF(cx, top - 4), QPointF(cx, bottom + 2))
        p.setPen(qc(T.MUTED))
        p.setFont(QFont("Segoe UI", 8))
        total = (f"{self.n_frames / self.fps / 60:.1f} min" if self.fps > 0
                 else f"{self.n_frames} frames")
        p.drawText(QRectF(left, h - 15, right - left - 110, 14), Qt.AlignLeft,
                   "cyan: confidence · red: bad frames · dashed: p-cutoff")
        p.drawText(QRectF(right - 120, h - 15, 120, 14), Qt.AlignRight, total)
        p.end()


class HardwareMonitor(QObject):
    """Samples GPU / CPU load on a background thread."""

    sample = Signal(dict)

    def __init__(self, interval: float = 1.5, parent=None):
        super().__init__(parent)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="hw-monitor")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        hardware.sample_system()  # prime cpu_percent
        while not self._stop.wait(self.interval):
            try:
                data = hardware.sample_system()
                data["gpus"] = hardware.sample_gpus()
                self.sample.emit(data)
            except Exception:
                pass
