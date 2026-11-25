"""Auto-tune: time the user's own model on this machine and pick the fastest
settings that fit in memory.

Steps:
1. Time DLC's default behavior (the project's batch size, one process). Every
   other result is reported relative to this baseline.
2. Sweep the batch size with one worker.
3. Sweep the number of parallel workers at the best batch size.
4. Re-test the batch size at the chosen worker count.
5. Time mixed precision at the chosen settings (reported separately).

Shared GPUs: before every step that needs more GPU memory, the free VRAM is
re-read and the step is skipped unless a safety margin remains, so a training
job on the same GPU is not pushed out of memory. If the GPU is busy during
calibration the profile is marked ``shared``; its speeds are then a lower
bound and the app suggests re-tuning when the GPU is idle.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from . import hardware
from .engine import RunConfig, WorkerProc, auto_cpu_threads

TARGET_SECONDS = 6.0  # length of one timed trial
GAIN = 1.05  # a step must be 5% faster to count as an improvement
TIE = 0.97  # settings within 3% of the best are considered equal


def bundled_clip() -> Path:
    return Path(__file__).resolve().parent / "assets" / "calibration" / "calibration_clip.mp4"


class Calibrator(QObject):
    progress = Signal(str, float)  # message, fraction 0..1
    trial_done = Signal(dict)
    log = Signal(str)
    finished = Signal(dict)  # profile ({} when cancelled / failed)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workers: list[WorkerProc] = []
        self.trials: list[dict] = []
        self.running = False
        self._gen = None
        self._pending = 0
        self._results: list[dict] = []
        self._errors: list[dict] = []
        self._tmp: Path | None = None
        self._samples: list[tuple] = []
        self._sampler = QTimer(self)
        self._sampler.setInterval(700)
        self._sampler.timeout.connect(self._take_sample)
        self._steps_done = 0
        self._steps_total = 1
        self.notes: list[str] = []

    # ------------------------------------------------------------- control
    def start(self, cfg: RunConfig, clip: str, hw: hardware.Hardware,
              default_batch: int, max_workers: int, device_index: int = 0) -> None:
        if self.running:
            return
        self.cfg, self.clip, self.hw = cfg, clip, hw
        self.default_batch = max(1, int(default_batch))
        self.max_workers = max(1, int(max_workers))
        self.device_index = device_index
        self.use_gpu = hw.has_gpu and cfg.device != "cpu"
        self.trials, self.notes = [], []
        self.workers = []
        self.running = True
        self._tmp = Path(tempfile.mkdtemp(prefix="dlc_batcher_cal_"))
        self.baseline_load = self._gpu_load()
        self.shared = hardware.gpu_is_shared(self.baseline_load) if self.use_gpu else False
        self._steps_done, self._steps_total = 0, 12
        self._gen = self._plan()
        self._advance()

    def cancel(self) -> None:
        if not self.running:
            return
        self.notes.append("Cancelled.")
        self._cleanup()
        self.finished.emit({})

    def _cleanup(self) -> None:
        self.running = False
        self._sampler.stop()
        self._gen = None
        for w in self.workers:
            w.kill()
        self.workers = []
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)

    # ----------------------------------------------------- generator driver
    def _advance(self) -> None:
        if not self.running or self._gen is None:
            return
        try:
            next(self._gen)
        except StopIteration:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self.log.emit(f"Calibration failed: {exc!r}")
            self._cleanup()
            self.finished.emit({})

    def _wait(self, n: int) -> None:
        self._pending = n
        self._results, self._errors = [], []

    def _one_done(self) -> None:
        self._pending -= 1
        if self._pending <= 0:
            QTimer.singleShot(0, self._advance)

    def _on_event(self, idx: int, ev: dict) -> None:
        name = ev.get("event")
        if name in ("ready", "set_ok"):
            self._one_done()
        elif name == "video_done":
            self._results.append(ev)
            self._one_done()
        elif name in ("video_error", "fatal"):
            self._errors.append(ev)
            self.log.emit(f"[cal worker {idx + 1}] {ev.get('message')}")
            self._one_done()

    def _on_exit(self, idx: int, code: int) -> None:
        if self.running and self._pending > 0:
            self._errors.append({"message": f"worker exited ({code})"})
            self._one_done()

    # ------------------------------------------------------------- helpers
    def _gpu_load(self):
        if not self.hw.has_gpu:
            return None
        loads = hardware.sample_gpus()
        for ld in loads:
            if ld.index == self.device_index:
                return ld
        return loads[0] if loads else None

    def _margin_mb(self) -> int:
        total = self.baseline_load.total_mb if self.baseline_load else 0
        return int(max(1024, 0.08 * total))

    def _headroom(self, extra_mb: float) -> bool:
        """Is there room for ``extra_mb`` more VRAM while keeping the margin?"""
        if not self.use_gpu:
            return True
        load = self._gpu_load()
        if load is None:
            return True
        return load.free_mb - extra_mb >= self._margin_mb()

    def _take_sample(self) -> None:
        load = self._gpu_load()
        sysload = hardware.sample_system()
        self._samples.append(
            (load.util if load else 0, load.used_mb if load else 0, sysload["cpu"])
        )

    def _spawn(self, total: int):
        new = []
        threads = auto_cpu_threads(total)
        while len(self.workers) < total:
            w = WorkerProc(len(self.workers), self.cfg.python_exe, threads, self)
            w.event.connect(self._on_event)
            w.log.connect(lambda i, line: None)
            w.exited.connect(self._on_exit)
            self.workers.append(w)
            new.append(w)
        if new:
            self._wait(len(new))
            for w in new:
                w.start()
                w.send(self.cfg.init_command(threads))
            yield

    def _set_all(self, n: int, **opts):
        self._wait(n)
        for w in self.workers[:n]:
            w.send(dict(cmd="set", **opts))
        yield

    def _run_all(self, n: int, frames: int):
        self._wait(n)
        for w in self.workers[:n]:
            cmd = self.cfg.analyze_command(self.clip)
            dest = self._tmp / f"w{w.index}"
            dest.mkdir(exist_ok=True)
            cmd.update(destfolder=str(dest), overwrite=True, save_as_csv=False,
                       frame_limit=int(frames), loop=True, dynamic=[False, 0.5, 10])
            w.send(cmd)
        yield

    def _step(self, message: str) -> None:
        self._steps_done += 1
        frac = min(0.97, self._steps_done / max(self._steps_total, self._steps_done + 1))
        self.progress.emit(message, frac)

    def _trial(self, batch: int, workers: int, prefetch: bool, label: str, amp: bool = False):
        """Warm up, then time ``workers`` processes at ``batch``. Returns the
        trial dict through ``self._last`` (None on failure)."""
        self._last = None
        self._step(f"Testing {label}: batch {batch} × {workers} worker(s)…")
        yield from self._spawn(workers)
        if self._errors:
            self.notes.append("A worker failed to start; stopped adding workers.")
            return
        yield from self._set_all(workers, batch_size=batch, prefetch=prefetch, amp=amp,
                                 cpu_threads=auto_cpu_threads(workers))
        yield from self._run_all(workers, max(96, 4 * batch))  # warm-up
        if self._errors:
            self._note_error(batch, workers)
            return
        warm = [r.get("steady_fps") or r.get("fps") or 50.0 for r in self._results]
        per_worker = min(warm) if warm else 50.0
        frames = int(min(12000, max(300, per_worker * TARGET_SECONDS)))
        frames = max(frames, 6 * batch)
        self._samples = []
        self._sampler.start()
        t0 = time.perf_counter()
        yield from self._run_all(workers, frames)
        wall = time.perf_counter() - t0
        self._sampler.stop()
        if self._errors or len(self._results) < workers:
            self._note_error(batch, workers)
            return
        fps = sum(float(r.get("fps") or 0.0) for r in self._results)
        peak = max(int(r.get("vram_peak_mb") or 0) for r in self._results)
        utils = [s[0] for s in self._samples] or [0]
        used = [s[1] for s in self._samples] or [0]
        cpus = [s[2] for s in self._samples] or [0]
        base_used = self.baseline_load.used_mb if self.baseline_load else 0
        trial = dict(
            label=label,
            batch_size=batch,
            workers=workers,
            prefetch=prefetch,
            amp=amp,
            fps=round(fps, 1),
            frames=frames * workers,
            seconds=round(wall, 2),
            worker_vram_mb=peak,
            vram_added_mb=max(0, max(used) - base_used),
            gpu_util=round(sum(utils) / len(utils)),
            cpu_util=round(sum(cpus) / len(cpus)),
        )
        self.trials.append(trial)
        self._last = trial
        self.trial_done.emit(trial)
        self.log.emit(
            f"  {label}: batch {batch} × {workers} → {fps:.0f} fps "
            f"(GPU {trial['gpu_util']}%, CPU {trial['cpu_util']}%)"
        )

    def _note_error(self, batch: int, workers: int) -> None:
        msg = "; ".join(str(e.get("message", "")) for e in self._errors)[:300]
        if any(e.get("oom") for e in self._errors) or "out of memory" in msg.lower():
            self.notes.append(
                f"Batch {batch} × {workers} worker(s) ran out of GPU memory; backed off."
            )
        else:
            self.notes.append(f"Batch {batch} × {workers} worker(s) failed: {msg}")
        # A failed worker may be in a bad state: restart from scratch
        for w in self.workers:
            w.kill()
        self.workers = []

    def _per_worker_mb(self, trial: dict) -> float:
        """VRAM one more worker is expected to need (model + CUDA context)."""
        measured = trial["vram_added_mb"] / max(1, trial["workers"])
        return max(measured, trial["worker_vram_mb"] + 350)

    # ---------------------------------------------------------------- plan
    def _plan(self):
        if self.shared and self.baseline_load is not None:
            self.notes.append(
                f"The GPU was busy during tuning ({self.baseline_load.used_mb / 1024:.1f} GB "
                f"and ~{self.baseline_load.util}% in use by other programs). Speeds are a "
                "lower bound. Re-tune when the GPU is idle."
            )

        # 1) DLC default: project batch size, one process, no background decoding
        yield from self._trial(self.default_batch, 1, False, "DLC default")
        base = self._last
        if base is None:
            self.notes.append("Could not run the baseline; calibration aborted.")
            self._done(None, None)
            return

        # 2) batch size sweep with one worker
        if self.use_gpu:
            total_gb = (self.baseline_load.total_mb if self.baseline_load else 8000) / 1024
            cands = [8, 16, 32, 64, 128, 256] if total_gb >= 6 else [2, 4, 8, 16, 32, 64]
        else:
            cands = [1, 2, 4, 8, 16]
        single: list[dict] = []
        misses = 0
        for b in cands:
            if single and not self._headroom(single[-1]["worker_vram_mb"]):
                self.notes.append(
                    f"Stopped at batch {single[-1]['batch_size']}: not enough free GPU "
                    "memory to safely test a larger batch right now."
                )
                break
            yield from self._trial(b, 1, True, "batch size")
            if self._last is None:
                break
            best_before = max((t["fps"] for t in single), default=0.0)
            single.append(self._last)
            misses = misses + 1 if self._last["fps"] < best_before * GAIN else 0
            if misses >= 2:
                break
        if not single:
            self._done(base, base)
            return
        top = max(t["fps"] for t in single)
        best1 = next(t for t in single if t["fps"] >= top * TIE)  # smallest tie
        batch = best1["batch_size"]

        # 3) worker sweep at that batch size
        multi = [best1]
        for wn in [2, 3, 4, 6, 8, 12, 16]:
            if wn > self.max_workers:
                break
            last = multi[-1]
            add = wn - last["workers"]
            if not self._headroom(add * self._per_worker_mb(last)):
                self.notes.append(
                    f"Stopped at {last['workers']} worker(s): not enough free GPU memory "
                    "to safely add more right now."
                )
                break
            yield from self._trial(batch, wn, True, "workers")
            if self._last is None:
                break
            best_before = max(t["fps"] for t in multi)
            multi.append(self._last)
            if self._last["fps"] < best_before * GAIN:
                break
        top = max(t["fps"] for t in multi)
        best = next(t for t in multi if t["fps"] >= top * TIE)

        # 4) refine the batch size at the chosen worker count
        if best["workers"] > 1 and self.workers:
            wn = best["workers"]
            for b in (batch // 2, batch * 2):
                if b < 1 or b > 512:
                    continue
                if b > batch and not self._headroom(wn * best["worker_vram_mb"]):
                    continue
                yield from self._trial(b, wn, True, "fine-tune")
                if self._last is not None and self._last["fps"] > best["fps"] * 1.03:
                    best = self._last
                if self._last is None:
                    break

        # 5) mixed precision at the chosen settings (reported separately
        #    because it changes the predictions slightly)
        amp_trial = None
        if self.use_gpu and self.cfg.engine == "pytorch":
            yield from self._trial(best["batch_size"], best["workers"], True,
                                   "mixed precision", amp=True)
            amp_trial = self._last
            # 16-bit shifts the limit from Python overhead to the GPU, so the
            # best batch size is often larger than in 32-bit: test upwards.
            from .hardware import default_settings

            cap = int(default_settings(self.hw).get("batch_size", 8))
            for b in (16, 32, 64):
                if amp_trial is None or b <= best["batch_size"] or b > cap:
                    continue
                yield from self._trial(b, best["workers"], True, "mixed precision", amp=True)
                if self._last is None:
                    break
                if self._last["fps"] > amp_trial["fps"] * 1.03:
                    amp_trial = self._last
                else:
                    break
        self._done(base, best, amp_trial)

    def _done(self, base: dict | None, best: dict | None, amp_trial: dict | None = None) -> None:
        profile: dict = {}
        if base and best:
            if best["fps"] <= base["fps"]:
                best = base
            profile = dict(
                batch_size=best["batch_size"],
                workers=best["workers"],
                prefetch=bool(best["prefetch"]),
                fps=best["fps"],
                baseline_fps=base["fps"],
                baseline_batch=base["batch_size"],
                speedup=round(best["fps"] / base["fps"], 2) if base["fps"] else 1.0,
                amp_fps=amp_trial["fps"] if amp_trial else 0.0,
                amp_batch=amp_trial["batch_size"] if amp_trial else 0,
                shared=self.shared,
                other_gpu_use_mb=self.baseline_load.used_mb if self.baseline_load else 0,
                other_gpu_util=self.baseline_load.util if self.baseline_load else 0,
                trials=self.trials,
                notes=self.notes,
                clip=str(self.clip),
                device=self.cfg.device,
                date=time.strftime("%Y-%m-%d %H:%M"),
                hardware=self.hw.to_dict(),
            )
        self.progress.emit("Done", 1.0)
        self._cleanup()
        self.finished.emit(profile)


def estimate_speedup(profile: dict | None, batch_size: int, workers: int,
                     amp: bool = False) -> dict | None:
    """Speed-up vs. DLC defaults for the given settings, from measured trials.

    Returns ``{"speedup", "fps", "baseline_fps", "exact"}`` or None when there
    is no calibration to base an estimate on.
    """
    if not profile or not profile.get("baseline_fps"):
        return None
    base = float(profile["baseline_fps"])
    if amp:
        for t in profile.get("trials", []):
            if (t.get("amp") and t.get("batch_size") == batch_size
                    and t.get("workers") == workers and t.get("fps")):
                return dict(speedup=float(t["fps"]) / base, fps=float(t["fps"]),
                            baseline_fps=base, exact=True)
        est = estimate_speedup(profile, batch_size, workers, amp=False)
        gain = float(profile.get("amp_fps") or 0) / float(profile.get("fps") or 1)
        if est is None or gain <= 0:
            return est
        est.update(speedup=est["speedup"] * gain, fps=est["fps"] * gain,
                   exact=est["exact"] and batch_size == profile.get("batch_size")
                   and workers == profile.get("workers"))
        return est
    trials = [t for t in profile.get("trials", []) if t.get("prefetch") and not t.get("amp")]
    for t in trials:
        if t["batch_size"] == batch_size and t["workers"] == workers:
            return dict(speedup=t["fps"] / base, fps=t["fps"], baseline_fps=base, exact=True)
    if not trials:
        return None
    # Estimate: batch effect measured with one worker x worker scaling measured
    # at the tuned batch size. Marked as not exact.
    single = sorted((t for t in trials if t["workers"] == 1), key=lambda t: t["batch_size"])
    if not single:
        return None
    nearest_b = min(single, key=lambda t: abs(t["batch_size"] - batch_size))
    by_w: dict[int, float] = {}
    for t in trials:
        by_w[t["workers"]] = max(by_w.get(t["workers"], 0.0), t["fps"])
    one = by_w.get(1) or nearest_b["fps"]
    known = sorted(by_w)
    lo = max((w for w in known if w <= workers), default=known[0])
    hi = min((w for w in known if w >= workers), default=known[-1])
    if lo == hi:
        scale = by_w[lo] / one
    else:
        f = (workers - lo) / (hi - lo)
        scale = (by_w[lo] + f * (by_w[hi] - by_w[lo])) / one
    fps = nearest_b["fps"] * scale
    return dict(speedup=fps / base, fps=fps, baseline_fps=base, exact=False)
