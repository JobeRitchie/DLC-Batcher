"""Run manager: spawns worker processes, hands out videos, aggregates progress."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .dlc_env import console_python
from .worker import SENTINEL

PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass
class RunConfig:
    config: str
    engine: str = "pytorch"
    shuffle: int = 1
    trainingsetindex: int = 0
    snapshot_index: object = None  # None -> project default
    detector_snapshot_index: object = None
    device: str = "auto"
    batch_size: int = 8
    detector_batch_size: int = 0  # 0 -> project default
    workers: int = 1
    prefetch: bool = True
    amp: bool = False
    cudnn_benchmark: bool = False
    cpu_threads: int = 0  # 0 -> automatic
    destfolder: str = ""  # "" -> next to each video (DLC default)
    save_as_csv: bool = False
    overwrite: bool = False
    dynamic: tuple = (False, 0.5, 10)
    cropping: list | None = None
    robust_nframes: bool = False
    auto_track: bool = True
    pcutoff: float = 0.6
    longest_first: bool = True
    python_exe: str = ""
    # calibration only
    frame_limit: int = 0
    loop: bool = False

    def init_command(self, cpu_threads: int) -> dict:
        return dict(
            cmd="init",
            config=self.config,
            engine=self.engine,
            shuffle=self.shuffle,
            trainingsetindex=self.trainingsetindex,
            snapshot_index=self.snapshot_index,
            detector_snapshot_index=self.detector_snapshot_index,
            device=self.device,
            batch_size=self.batch_size,
            detector_batch_size=self.detector_batch_size,
            prefetch=self.prefetch,
            amp=self.amp,
            cudnn_benchmark=self.cudnn_benchmark,
            cpu_threads=cpu_threads,
        )

    def analyze_command(self, video: str) -> dict:
        return dict(
            cmd="analyze",
            video=video,
            destfolder=self.destfolder or None,
            save_as_csv=self.save_as_csv,
            overwrite=self.overwrite,
            dynamic=list(self.dynamic),
            cropping=self.cropping,
            robust_nframes=self.robust_nframes,
            auto_track=self.auto_track,
            pcutoff=self.pcutoff,
            frame_limit=self.frame_limit,
            loop=self.loop,
        )


@dataclass
class VideoItem:
    path: str
    n_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    status: str = "queued"  # queued|loading|running|saving|done|skipped|error
    frames_done: int = 0
    worker: int = -1
    seconds: float = 0.0
    speed: float = 0.0  # frames / seconds since the video started
    steady: float = 0.0  # the same without the first batch (and detector pass)
    has_detector: bool = False
    mean_conf: list = field(default_factory=list)
    frac_conf: list = field(default_factory=list)
    lost_frac: float = 0.0
    message: str = ""
    attempts: int = 0

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def default_python() -> str:
    """Interpreter used for workers when none was chosen: the one running the app."""
    return console_python(sys.executable)


class WorkerProc(QObject):
    """One worker subprocess speaking the JSON-lines protocol."""

    event = Signal(int, dict)
    log = Signal(int, str)
    exited = Signal(int, int)

    def __init__(self, index: int, python_exe: str, cpu_threads: int = 0, parent=None):
        super().__init__(parent)
        self.index = index
        self.ready = False
        self.busy = False
        self.current: str | None = None
        self.info: dict = {}
        self._buf = b""
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("DLClight", "True")
        existing = env.value("PYTHONPATH", "")
        env.insert("PYTHONPATH", PACKAGE_ROOT + (os.pathsep + existing if existing else ""))
        if cpu_threads > 0:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env.insert(var, str(cpu_threads))
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setProgram(python_exe or default_python())
        self.proc.setArguments(["-u", "-m", "dlc_batcher.worker"])
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._error)

    def start(self) -> None:
        self.proc.start()

    def send(self, cmd: dict) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.proc.write((json.dumps(cmd, default=str) + "\n").encode("utf-8"))

    def alive(self) -> bool:
        return self.proc.state() != QProcess.NotRunning

    def kill(self) -> None:
        if self.alive():
            self.proc.kill()
            self.proc.waitForFinished(3000)

    def quit(self) -> None:
        self.send({"cmd": "quit"})

    def _read(self) -> None:
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            if line.startswith(SENTINEL):
                try:
                    ev = json.loads(line[len(SENTINEL):])
                except ValueError:
                    continue
                if ev.get("event") == "ready":
                    self.ready = True
                    self.info = ev
                self.event.emit(self.index, ev)
            elif line.strip():
                self.log.emit(self.index, line)

    def _finished(self, code: int, _status) -> None:
        self.ready = False
        self.exited.emit(self.index, code)

    def _error(self, err) -> None:
        if err == QProcess.FailedToStart:
            self.log.emit(self.index, f"Could not start worker: {self.proc.errorString()}")
            self.exited.emit(self.index, -1)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def auto_cpu_threads(workers: int) -> int:
    cores = os.cpu_count() or 4
    return max(1, min(8, cores // max(1, workers)))


class RunManager(QObject):
    """Drives a batch run. Poll ``snapshot()`` for dashboard numbers."""

    log = Signal(str)
    video_changed = Signal(str)
    worker_changed = Signal(int)
    finished = Signal(dict)
    started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg: RunConfig | None = None
        self.videos: list[VideoItem] = []
        self.by_path: dict[str, VideoItem] = {}
        self.workers: list[WorkerProc] = []
        self.queue: deque[VideoItem] = deque()
        self.running = False
        self.stopping = False
        self.t_start = 0.0
        self.t_end = 0.0
        self.history: deque = deque(maxlen=600)  # (t, total frames done)
        self.speed_series: list[float] = []
        self.worker_windows: dict[int, list] = {}
        self.worker_vram: dict[int, int] = {}
        self._reset_rates()
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._sample)

    # ------------------------------------------------------------------ run
    def start(self, cfg: RunConfig, videos: list[VideoItem]) -> None:
        if self.running:
            return
        self.cfg = cfg
        self.videos = videos
        self.by_path = {v.path: v for v in videos}
        for v in videos:
            v.status, v.frames_done, v.worker, v.message = "queued", 0, -1, ""
            v.attempts = 0
            v.seconds = v.speed = v.steady = 0.0
            v.mean_conf, v.frac_conf, v.lost_frac = [], [], 0.0
        order = list(videos)
        if cfg.longest_first:
            order.sort(key=lambda v: -v.n_frames)
        self.queue = deque(order)
        self.running, self.stopping = True, False
        self.t_start, self.t_end = time.time(), 0.0
        self.history.clear()
        self.speed_series = []
        self.worker_windows, self.worker_vram = {}, {}
        self._reset_rates()
        n = max(1, min(cfg.workers, len(videos)))
        threads = cfg.cpu_threads or auto_cpu_threads(n)
        self.workers = []
        for i in range(n):
            w = WorkerProc(i, cfg.python_exe, threads, self)
            w.event.connect(self._on_event)
            w.log.connect(lambda idx, line: self.log.emit(f"[worker {idx + 1}] {line}"))
            w.exited.connect(self._on_exit)
            self.workers.append(w)
            w.start()
            w.send(cfg.init_command(threads))
        self.log.emit(
            f"Starting {n} worker(s) · batch size {cfg.batch_size} · {len(videos)} video(s)"
        )
        self._tick.start()
        self.started.emit()

    def stop(self, graceful: bool = False) -> None:
        if not self.running:
            return
        self.stopping = True
        self.queue.clear()
        if graceful:
            self.log.emit("Finishing the videos in progress, then stopping…")
            for w in self.workers:
                if not w.busy:
                    w.quit()
            self._check_done()
            return
        self.log.emit("Stopping now.")
        for w in self.workers:
            w.kill()
        for v in self.videos:
            if v.status in ("loading", "running", "saving"):
                v.status, v.frames_done, v.worker = "queued", 0, -1
                v.message = "cancelled"
                self.video_changed.emit(v.path)
        self._finish()

    # --------------------------------------------------------------- events
    def _assign(self, w: WorkerProc) -> None:
        if not w.ready or w.busy or not self.running:
            return
        if not self.queue:
            w.quit()
            return
        v = self.queue.popleft()
        v.status, v.worker = "loading", w.index
        v.attempts += 1
        w.busy, w.current = True, v.path
        w.send(self.cfg.analyze_command(v.path))
        self.video_changed.emit(v.path)
        self.worker_changed.emit(w.index)

    def _on_event(self, idx: int, ev: dict) -> None:
        w = self.workers[idx]
        name = ev.get("event")
        v = self.by_path.get(ev.get("video", ""))
        if name == "ready":
            dev = ev.get("gpu_name") or "CPU"
            self.log.emit(f"[worker {idx + 1}] ready on {dev} (DLC {ev.get('dlc_version')})")
            self._assign(w)
        elif name == "video_start" and v:
            v.status = "running"
            if ev.get("n_frames"):
                v.n_frames = int(ev["n_frames"])
            if ev.get("fps"):
                v.fps = float(ev["fps"])
            if ev.get("width"):
                v.width, v.height = int(ev["width"]), int(ev["height"])
            v.has_detector = bool(ev.get("has_detector"))
            self.video_changed.emit(v.path)
        elif name == "progress" and v:
            v.frames_done = int(ev.get("frames_done", 0))
            if ev.get("phase") == "save":
                v.status = "saving"
            elif ev.get("phase") == "detect":
                v.message = f"detector {ev.get('detect_done', 0)}/{v.n_frames}"
            if "mean_conf" in ev:
                v.mean_conf, v.frac_conf = ev["mean_conf"], ev["frac_conf"]
                v.lost_frac = float(ev.get("lost_frac", 0.0))
            if "window_conf" in ev:
                self.worker_windows[idx] = ev["window_conf"]
            if "vram_peak_mb" in ev:
                self.worker_vram[idx] = int(ev["vram_peak_mb"])
            el = float(ev.get("elapsed") or 0)
            if el > 0:
                v.speed = v.frames_done / el
            if ev.get("steady_fps"):
                v.steady = float(ev["steady_fps"])
            self.video_changed.emit(v.path)
        elif name == "video_done" and v:
            w.busy, w.current = False, None
            v.seconds = float(ev.get("seconds", 0))
            if ev.get("skipped"):
                v.status, v.frames_done = "skipped", 0
                v.message = "already analyzed"
            else:
                v.status = "done"
                v.frames_done = int(ev.get("frames") or v.n_frames)
                v.n_frames = max(v.n_frames, v.frames_done) if not self.cfg.frame_limit else v.frames_done
                v.speed = float(ev.get("fps") or v.speed)
                v.message = ""
                if "mean_conf" in ev:
                    v.mean_conf, v.frac_conf = ev["mean_conf"], ev["frac_conf"]
                    v.lost_frac = float(ev.get("lost_frac", 0.0))
            self.video_changed.emit(v.path)
            self.worker_changed.emit(idx)
            if self.stopping:
                w.quit()
            else:
                self._assign(w)
            self._check_done()
        elif name == "video_error" and v:
            w.busy, w.current = False, None
            self.log.emit(f"[worker {idx + 1}] ERROR on {v.name}: {ev.get('message')}")
            self.log.emit(ev.get("traceback", ""))
            retry = v.attempts < 2 and not ev.get("oom") and not self.stopping
            if retry:
                # transient GPU hiccups happen, especially on a shared GPU
                v.status, v.frames_done, v.worker = "queued", 0, -1
                v.message = "retrying after an error"
                self.queue.append(v)
                self.log.emit(f"Will retry {v.name} once.")
            else:
                v.status = "error"
                v.message = ev.get("message", "error")
                if ev.get("oom"):
                    v.message = "GPU out of memory - lower the batch size or worker count"
            self.video_changed.emit(v.path)
            self.worker_changed.emit(idx)
            if self.stopping:
                w.quit()
            else:
                self._assign(w)
            self._check_done()
        elif name == "fatal":
            self.log.emit(f"[worker {idx + 1}] FATAL: {ev.get('message')}\n{ev.get('traceback', '')}")

    def _on_exit(self, idx: int, code: int) -> None:
        if not self.running:
            return
        w = self.workers[idx]
        if w.current:
            v = self.by_path.get(w.current)
            if v and v.status in ("loading", "running", "saving"):
                v.status = "error"
                v.message = f"worker crashed (exit code {code})"
                self.video_changed.emit(v.path)
            w.busy, w.current = False, None
        self.worker_changed.emit(idx)
        self._check_done()

    def _check_done(self) -> None:
        if not self.running:
            return
        busy = any(w.busy for w in self.workers)
        alive = any(w.alive() for w in self.workers)
        if not busy and (not self.queue or not alive):
            for v in self.queue:
                v.message = "not processed (no worker available)"
            self._finish()

    def _finish(self) -> None:
        if not self.running:
            return
        self.running = False
        self.t_end = time.time()
        self._tick.stop()
        self._sample()
        for w in self.workers:
            w.quit()
        QTimer.singleShot(4000, self._reap)
        self.finished.emit(self.summary())

    def _reap(self) -> None:
        if self.running:
            return
        for w in self.workers:
            if w.alive():
                w.kill()

    # ---------------------------------------------------------------- stats
    def _reset_rates(self) -> None:
        self._worker_rate: dict[int, float] = {}
        self._phase: dict[str, tuple[str, float]] = {}
        self._overhead: dict[str, list[float]] = {"loading": [], "saving": []}
        self._loaded_once: set[int] = set()
        self._rates_t = self._eta_age = 0.0
        self._eta: float | None = None

    def _frames_done(self) -> int:
        return sum(v.frames_done for v in self.videos if v.status != "skipped")

    def _frames_total(self) -> int:
        return sum(v.n_frames for v in self.videos if v.status != "skipped")

    def _sample(self) -> None:
        now = time.time()
        done = self._frames_done()
        self.history.append((now, done))
        self._update_rates(now, done)
        self.speed_series.append(self.steady_speed())
        if len(self.speed_series) > 3600:
            self.speed_series = self.speed_series[::2]

    def _update_rates(self, now: float, done: int) -> None:
        """Once a second: refresh the speed readout and the time left.

        Frames stop arriving while a worker opens the next video or writes
        results, so anything computed from "frames in the last few seconds"
        drops between videos. Both numbers are built from per-video figures
        instead:

        * speed is the sum of each worker's rate on its current video. The rate
          of a video that has only just started is blended in over its first
          seconds, and it is kept while the worker is between videos;
        * time left is the remaining frames at that rate plus the measured
          load and save time of every video still to finish, shared between
          the workers. The displayed value counts down in real time and is
          pulled towards the estimate a little each second.
        """
        dt = now - self._rates_t if self._rates_t else 1.0
        self._rates_t = now
        if dt <= 0:
            return

        # how long each video has been in its current state; a video leaving
        # "loading" or "saving" gives one sample of that overhead
        for v in self.videos:
            prev = self._phase.get(v.path)
            if prev is not None and prev[0] == v.status:
                continue
            if prev is not None and prev[0] in ("loading", "saving"):
                first_load = prev[0] == "loading" and v.worker not in self._loaded_once
                if prev[0] == "loading":
                    self._loaded_once.add(v.worker)
                if not first_load:  # a worker's first video also loads the model
                    self._overhead[prev[0]].append(now - prev[1])
            self._phase[v.path] = (v.status, now)

        for w in self.workers:
            v = self.by_path.get(w.current or "")
            rate = (v.steady or v.speed) if v is not None and v.status == "running" else 0.0
            if rate > 0 and v.frames_done > 0:
                old = self._worker_rate.get(w.index)
                ran = now - self._phase[v.path][1]
                weight = 1.0 if old is None else min(1.0, ran / 10.0) ** 2
                self._worker_rate[w.index] = rate if old is None else old + weight * (rate - old)
            elif not w.alive():
                self._worker_rate.pop(w.index, None)
        if not self._worker_rate:
            return
        per_worker = sum(self._worker_rate.values()) / len(self._worker_rate)

        load = _median(self._overhead["loading"])
        save = _median(self._overhead["saving"])
        work = longest = 0.0
        for v in self.videos:
            spent = now - self._phase[v.path][1]
            rate = self._worker_rate.get(v.worker, per_worker)
            if v.status == "queued":
                left = load + v.n_frames / per_worker + save
            elif v.status == "loading":
                left = max(0.0, load - spent) + v.n_frames / per_worker + save
            elif v.status == "running":
                if v.has_detector or v.frames_done <= 0:
                    left = max(0.0, v.n_frames / rate - spent) + save
                else:
                    left = max(0, v.n_frames - v.frames_done) / (v.steady or rate) + save
            elif v.status == "saving":
                left = max(0.0, save - spent)
            else:
                continue
            work += left
            longest = max(longest, left)
        active = max(1, sum(1 for w in self.workers if w.alive()))
        target = max(work / active, longest)
        self._eta_age += dt
        if self._eta is None or self._eta_age < 15:  # first estimates are rough
            self._eta = target
            return
        ticked = max(0.0, self._eta - dt)
        self._eta = ticked + min(1.0, 0.05 * dt) * (target - ticked)

    def steady_speed(self) -> float:
        """Frames per second while analyzing, summed over the workers."""
        return float(sum(self._worker_rate.values()))

    def speed(self, window: float = 20.0) -> float:
        """Frames per second across all workers over the last ``window`` s."""
        if len(self.history) < 2:
            return 0.0
        t1, f1 = self.history[-1]
        for t0, f0 in self.history:
            if t1 - t0 <= window:
                break
        if t1 - t0 < 0.5:
            return 0.0
        return max(0.0, (f1 - f0) / (t1 - t0))

    def elapsed(self) -> float:
        if not self.t_start:
            return 0.0
        return (self.t_end or time.time()) - self.t_start

    def eta_seconds(self) -> float | None:
        if not self.running:
            return 0.0
        return self._eta

    def confidence(self) -> dict:
        """Frame-weighted confidence metrics across analyzed frames."""
        tot = 0
        mean = frac = None
        lost = 0.0
        for v in self.videos:
            if not v.mean_conf or v.frames_done <= 0:
                continue
            n = v.frames_done
            if mean is None or len(mean) != len(v.mean_conf):
                mean = [0.0] * len(v.mean_conf)
                frac = [0.0] * len(v.mean_conf)
                tot, lost = 0, 0.0
            for i, (m, f) in enumerate(zip(v.mean_conf, v.frac_conf)):
                mean[i] += m * n
                frac[i] += f * n
            lost += v.lost_frac * n
            tot += n
        if not tot:
            return {}
        mean = [m / tot for m in mean]
        frac = [f / tot for f in frac]
        return {
            "mean": mean,
            "frac": frac,
            "overall_mean": sum(mean) / len(mean),
            "overall_frac": sum(frac) / len(frac),
            "lost": lost / tot,
            "frames": tot,
        }

    def counts(self) -> dict:
        c = {"queued": 0, "active": 0, "done": 0, "skipped": 0, "error": 0}
        for v in self.videos:
            if v.status in ("loading", "running", "saving"):
                c["active"] += 1
            else:
                c[v.status] = c.get(v.status, 0) + 1
        return c

    def snapshot(self) -> dict:
        total, done = self._frames_total(), self._frames_done()
        return {
            "running": self.running,
            "elapsed": self.elapsed(),
            "eta": self.eta_seconds(),
            "speed": self.steady_speed() or self.speed(),
            "speed_now": self.speed(window=3.0),
            "frames_done": done,
            "frames_total": total,
            "fraction": (done / total) if total else 0.0,
            "counts": self.counts(),
            "confidence": self.confidence(),
            "speed_series": self.speed_series,
            "vram_mb": sum(self.worker_vram.values()),
        }

    def summary(self) -> dict:
        snap = self.snapshot()
        el = snap["elapsed"]
        snap["avg_speed"] = snap["frames_done"] / el if el > 0 else 0.0
        video_seconds = sum(v.duration for v in self.videos if v.status == "done")
        snap["realtime_factor"] = video_seconds / el if el > 0 else 0.0
        snap["stopped"] = self.stopping
        return snap
