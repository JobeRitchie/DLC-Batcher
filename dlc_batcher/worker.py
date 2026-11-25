"""Analysis worker process.

One worker = one Python process that keeps a DLC model loaded and analyzes the
videos it is handed, one after the other. Several workers run side by side to
use the whole machine.

Protocol (JSON lines):
  stdin  <- {"cmd": "init", ...} | {"cmd": "analyze", ...} | {"cmd": "quit"}
  stdout -> lines starting with SENTINEL carry a JSON event; every other line
            is ordinary DLC console output and is treated as log text.

The worker calls DeepLabCut's own ``analyze_videos`` so the files it writes are
exactly what DLC would write. It only hooks a few internals to (1) report live
progress and confidence metrics, (2) keep the model loaded between videos and
(3) on DeepLabCut older than 3.0.2, prepare frames on a second thread and
decode keypoints a batch at a time (see install_pytorch_hooks).
"""
from __future__ import annotations

import copy
import gc
import json
import os
import queue
import sys
import threading
import time
import traceback

SENTINEL = "@@DLCB@@"
_OUT = sys.stdout


def emit(event: str, **data) -> None:
    data["event"] = event
    try:
        _OUT.write(SENTINEL + json.dumps(data, default=str) + "\n")
        _OUT.flush()
    except Exception:
        pass


class State:
    """Mutable state shared between the hooks and the command loop."""

    def __init__(self) -> None:
        self.engine = "pytorch"
        self.init: dict = {}
        self.runner_cache: dict = {}
        self.reset_video({})

    def reset_video(self, opts: dict) -> None:
        self.opts = opts
        self.video = opts.get("video", "")
        self.started = False
        self.n_frames = 0
        self.frames_done = 0
        self.detect_done = 0
        self.phase = "load"
        self.t_start = 0.0
        self.t_first = 0.0
        self.frames_at_first = 0
        self.has_detector = False
        self.t_last = 0.0
        self.last_emit = 0.0
        self.infer_seconds = 0.0
        self.bodyparts: list[str] = []
        self.conf_sum = None
        self.conf_hits = None
        self.conf_frames = 0
        self.lost_frames = 0
        self.win_sum = None
        self.win_frames = 0


STATE = State()


# --------------------------------------------------------------------------
# Frame feeding helpers
# --------------------------------------------------------------------------
class _SyncFeeder:
    """Frame limit / looping for calibration runs."""

    def __init__(self, it, limit: int, loop: bool) -> None:
        self.it, self.limit, self.loop, self.n = it, limit, loop, 0

    def get(self):
        if self.limit and self.n >= self.limit:
            return None
        frame = self.it.read_frame(crop=self.it._crop)
        if frame is None and self.loop and self.limit and self.n > 0:
            self.it.reset()
            frame = self.it.read_frame(crop=self.it._crop)
        if frame is None:
            return None
        self.n += 1
        return frame.copy()


class _Prepared:
    """A frame that was already decoded *and* preprocessed off-thread."""

    __slots__ = ("inputs", "context")

    def __init__(self, inputs, context) -> None:
        self.inputs, self.context = inputs, context


def _background_prepare(runner, images, depth: int):
    """Decode + preprocess on a background thread, in order.

    DLC decodes, preprocesses and runs the network on one thread, so the GPU
    idles while the CPU prepares the next frame. OpenCV / NumPy / PyTorch
    release the GIL, so moving that work to a thread overlaps it with the GPU.
    """
    from pathlib import Path

    import numpy as np

    q: queue.Queue = queue.Queue(maxsize=max(16, depth))
    stop = threading.Event()
    # Capture the preprocessor once: the consumer thread temporarily replaces
    # ``runner.preprocessor`` while it files prepared frames, so this thread
    # must never read that attribute again.
    preprocess = runner.preprocessor
    DONE = object()

    def put(item) -> None:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def work() -> None:
        try:
            for data in images:
                if stop.is_set():
                    return
                if isinstance(data, (str, Path, np.ndarray)):
                    inputs, context = data, {}
                else:
                    inputs, context = data
                inputs, context = preprocess(inputs, context)
                put(_Prepared(inputs, context))
            put(DONE)
        except BaseException as exc:
            put(exc)

    thread = threading.Thread(target=work, daemon=True, name="dlcb-prepare")
    thread.start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


# --------------------------------------------------------------------------
# PyTorch engine hooks
# --------------------------------------------------------------------------
def _progress(force: bool = False) -> None:
    st = STATE
    now = time.perf_counter()
    if not force and now - st.last_emit < 0.25:
        return
    st.last_emit = now
    data = dict(
        video=st.video,
        phase=st.phase,
        frames_done=st.frames_done,
        detect_done=st.detect_done,
        n_frames=st.n_frames,
        elapsed=now - st.t_start if st.t_start else 0.0,
    )
    if st.phase == "pose" and st.t_last - st.t_first > 1.0 and st.frames_done > st.frames_at_first:
        # rate since the first batch, which carries the one-off warm-up
        rate = (st.frames_done - st.frames_at_first) / (st.t_last - st.t_first)
        if st.has_detector and st.n_frames:
            # the detector ran over the whole video first: give the rate for
            # the video as a whole, assuming the pose pass keeps this speed
            rate = st.n_frames / ((st.t_first - st.t_start) + st.n_frames / rate)
        data["steady_fps"] = round(rate, 2)
    if st.win_frames:
        data["window_conf"] = [round(float(x), 4) for x in st.win_sum / st.win_frames]
        st.win_sum[:] = 0
        st.win_frames = 0
    if st.conf_frames:
        data["mean_conf"] = [round(float(x), 4) for x in st.conf_sum / st.conf_frames]
        data["frac_conf"] = [round(float(x), 4) for x in st.conf_hits / st.conf_frames]
        data["lost_frac"] = round(st.lost_frames / st.conf_frames, 4)
    try:
        import torch

        if torch.cuda.is_available():
            data["vram_mb"] = int(torch.cuda.memory_reserved() / 2**20)
            data["vram_peak_mb"] = int(torch.cuda.max_memory_reserved() / 2**20)
    except Exception:
        pass
    emit("progress", **data)


def _accumulate(results: list) -> None:
    """Fold a few frames of final predictions into the running metrics."""
    import numpy as np

    st = STATE
    pcut = float(st.opts.get("pcutoff", 0.6))
    for r in results:
        bp = r.get("bodyparts") if isinstance(r, dict) else None
        if bp is None:
            continue
        arr = np.asarray(bp, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] < 3 or arr.shape[0] == 0:
            continue
        conf = np.nan_to_num(arr[..., 2], nan=0.0, neginf=0.0, posinf=1.0)
        conf = np.clip(conf, 0.0, 1.0).max(axis=0)  # best assembly per bodypart
        if st.conf_sum is None or len(st.conf_sum) != len(conf):
            st.conf_sum = np.zeros(len(conf), dtype=np.float64)
            st.conf_hits = np.zeros(len(conf), dtype=np.float64)
            st.win_sum = np.zeros(len(conf), dtype=np.float64)
            st.conf_frames = st.lost_frames = st.win_frames = 0
        st.conf_sum += conf
        st.win_sum += conf
        hits = conf >= pcut
        st.conf_hits += hits
        st.conf_frames += 1
        st.win_frames += 1
        if not hits.any():
            st.lost_frames += 1


def install_pytorch_hooks() -> None:
    import torch
    import deeplabcut.pose_estimation_pytorch.apis.utils as U
    import deeplabcut.pose_estimation_pytorch.apis.videos as V
    import deeplabcut.pose_estimation_pytorch.runners.inference as I

    st = STATE
    # DLC 3.0.2 and later build batches from a list and preprocess on their own
    # thread (InferenceRunner._preprocessing_worker). Older releases keep a
    # growing ``_batch`` tensor and do everything on one thread.
    legacy_runner = not hasattr(I.InferenceRunner, "_preprocessing_worker")

    # --- silence tqdm: progress is reported through events instead ----------
    V.tqdm = lambda iterable=None, *a, **k: iterable

    # --- live progress + metrics --------------------------------------------
    orig_extract = I.InferenceRunner._extract_results

    def extract_results(self, shelf_writer):
        # Count the images this call will finish. On DLC 3.0.2 the preprocessing
        # thread appends to _image_batch_sizes while this runs, so the length
        # before and after cannot be compared; the front of the list is stable.
        have, done = len(self._predictions), 0
        for size in list(self._image_batch_sizes):
            if size > have:
                break
            have -= size
            done += 1
        results = orig_extract(self, shelf_writer)
        if done <= 0 or not st.started:
            return results
        now = time.perf_counter()
        if isinstance(self, I.DetectorInferenceRunner):
            st.phase = "detect"
            st.detect_done += done
        else:
            st.phase = "pose"
            if st.t_first == 0.0:
                st.t_first = now
                st.frames_at_first = done
            st.frames_done += done
            st.t_last = now
            try:
                _accumulate(results)
            except Exception:
                pass
        _progress()
        return results

    I.InferenceRunner._extract_results = extract_results

    # --- faster batching -----------------------------------------------------
    # DLC grows the pending batch with torch.cat() once per frame, which copies
    # the whole batch every time (cost grows with the batch size squared).
    # Collect the per-frame tensors in a list and concatenate once per batch.
    orig_prepare = I.InferenceRunner._prepare_inputs
    orig_waiting = I.InferenceRunner._inputs_waiting_for_processing
    orig_inference = I.InferenceRunner.inference
    _passthrough = lambda inputs, context: (inputs, context)  # noqa: E731

    def _merge(self) -> None:
        chunks = self.__dict__.get("_dlcb_chunks")
        if chunks:
            parts = ([self._batch] if self._batch is not None else []) + chunks
            self._batch = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            self.__dict__["_dlcb_chunks"] = []
            self.__dict__["_dlcb_n"] = 0

    def prepare_inputs(self, data):
        held, self._batch = self._batch, None
        try:
            if isinstance(data, _Prepared):
                pre, self.preprocessor = self.preprocessor, _passthrough
                try:
                    orig_prepare(self, (data.inputs, data.context))
                finally:
                    self.preprocessor = pre
            else:
                orig_prepare(self, data)
        finally:
            new, self._batch = self._batch, held
        if new is not None:
            self.__dict__.setdefault("_dlcb_chunks", []).append(new)
            self.__dict__["_dlcb_n"] = self.__dict__.get("_dlcb_n", 0) + len(new)
            have = (len(held) if held is not None else 0) + self.__dict__["_dlcb_n"]
            if have >= self.batch_size:
                _merge(self)

    def inputs_waiting(self):
        _merge(self)
        return orig_waiting(self)

    def inference(self, images, shelf_writer=None):
        self.__dict__["_dlcb_chunks"] = []
        self.__dict__["_dlcb_n"] = 0
        plain = type(self) in (I.PoseInferenceRunner, I.DetectorInferenceRunner)
        if (
            plain
            and st.init.get("prefetch", True)
            and self.preprocessor is not None
            and getattr(self, "dynamic", None) is None
        ):
            images = _background_prepare(self, images, 3 * int(self.batch_size))
        return orig_inference(self, images, shelf_writer)

    if legacy_runner and all(
        hasattr(I.InferenceRunner, n) for n in ("_process_full_batches", "_process_batch")
    ):
        I.InferenceRunner._prepare_inputs = prepare_inputs
        I.InferenceRunner._inputs_waiting_for_processing = inputs_waiting
        I.InferenceRunner.inference = inference

    # --- optional half precision ---------------------------------------------
    orig_predict = I.PoseInferenceRunner.predict

    def predict(self, inputs, **kwargs):
        want_half = bool(st.init.get("amp")) and str(self.device).startswith("cuda")
        if want_half != self.__dict__.get("_dlcb_half", False):
            # The whole model in 16-bit. torch.autocast was slower than 32-bit
            # with this model (145 vs 153 frames/s on an RTX 4070 Laptop GPU).
            self.model.half() if want_half else self.model.float()
            self.__dict__["_dlcb_half"] = want_half
        if not want_half:
            return orig_predict(self, inputs, **kwargs)
        import numpy as np

        if torch.is_tensor(inputs):
            inputs = inputs.to(self.device, non_blocking=True).half()
        preds = orig_predict(self, inputs, **kwargs)
        for p in preds:
            for head in p.values():
                for k, v in head.items():
                    if isinstance(v, np.ndarray) and v.dtype == np.float16:
                        head[k] = v.astype(np.float32)
        return preds

    I.PoseInferenceRunner.predict = predict

    # --- vectorised keypoint decoding ---------------------------------------
    # DLC reads each keypoint's score and offset in a Python loop over
    # (frame, keypoint), one GPU round trip each (~0.65 ms per frame with 7
    # keypoints). Indexing all of them at once gives the same values.
    try:
        from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
            HeatmapPredictor,
        )

        orig_pose_prediction = HeatmapPredictor.get_pose_prediction

        def get_pose_prediction(self, heatmap, locref, scale_factors):
            if not st.init.get("prefetch", True) or heatmap.dim() != 4:
                return orig_pose_prediction(self, heatmap, locref, scale_factors)
            y, x = self.get_top_values(heatmap)
            batch_size, num_joints = x.shape
            b = torch.arange(batch_size, device=heatmap.device)[:, None]
            j = torch.arange(num_joints, device=heatmap.device)[None, :]
            pose = torch.zeros((batch_size, 1, num_joints, 3), device=heatmap.device)
            pose[:, 0, :, 0] = x * scale_factors[1] + 0.5 * scale_factors[1]
            pose[:, 0, :, 1] = y * scale_factors[0] + 0.5 * scale_factors[0]
            if locref is not None:
                pose[:, 0, :, :2] += locref[b, y, x, j, :]
            pose[:, 0, :, 2] = heatmap[b, y, x, j]
            return pose

        if legacy_runner:  # 3.0.2 already decodes a whole batch at once
            HeatmapPredictor.get_pose_prediction = get_pose_prediction
    except Exception:  # predictor moved or changed in this DLC version
        pass

    # --- keep models loaded between videos ----------------------------------
    def _cached(kind: str, orig):
        def getter(*args, **kwargs):
            special = (
                args
                or kwargs.get("dynamic") is not None
                or kwargs.get("cond_provider") is not None
                or kwargs.get("ctd_tracking")
                or kwargs.get("transform") is not None
            )
            if special:
                return orig(*args, **kwargs)
            cfg = kwargs.get("model_config") or {}
            key = (
                str(kwargs.get("snapshot_path")),
                kwargs.get("batch_size"),
                kwargs.get("max_individuals"),
                str(cfg.get("device")),
            )
            hit = st.runner_cache.get(kind)
            if hit is not None and hit[0] == key:
                return hit[1]
            st.runner_cache.pop(kind, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            runner = orig(*args, **kwargs)
            st.runner_cache[kind] = (key, runner)
            return runner

        return getter

    U.get_pose_inference_runner = _cached("pose", U.get_pose_inference_runner)
    U.get_detector_inference_runner = _cached("detector", U.get_detector_inference_runner)

    # --- video iteration: frame limit, looping, background decoding ---------
    VI = V.VideoIterator
    orig_next = VI.__next__
    orig_nframes = VI.get_n_frames

    def get_n_frames(self, robust=False):
        n = orig_nframes(self, robust=robust)
        limit = int(st.opts.get("frame_limit") or 0)
        if limit:
            return limit if st.opts.get("loop") else min(limit, n)
        return n

    def __next__(self):
        limit = int(st.opts.get("frame_limit") or 0)
        if not limit:
            return orig_next(self)
        feeder = self.__dict__.get("_dlcb_feeder")
        if feeder is None:
            feeder = _SyncFeeder(self, limit, bool(st.opts.get("loop")))
            self.__dict__["_dlcb_feeder"] = feeder
        frame = feeder.get()
        if frame is None:
            self.__dict__.pop("_dlcb_feeder", None)
            self._index = 0
            self.reset()
            raise StopIteration
        if self._context is None:
            self._index += 1
            return frame
        context = copy.deepcopy(self._context[self._index])
        self._index += 1
        return frame, context

    VI.get_n_frames = get_n_frames
    VI.__next__ = __next__

    # --- video start / timing -----------------------------------------------
    orig_video_inference = V.video_inference

    def video_inference(video, pose_runner, detector_runner=None, **kwargs):
        st.started = True
        st.has_detector = detector_runner is not None
        st.phase = "detect" if st.has_detector else "pose"
        try:
            st.n_frames = int(video.get_n_frames())
            w, h = video.dimensions
            fps = float(video.fps)
        except Exception:
            w = h = 0
            fps = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        emit(
            "video_start",
            video=st.video,
            n_frames=st.n_frames,
            fps=fps,
            width=w,
            height=h,
            has_detector=detector_runner is not None,
            batch_size=getattr(pose_runner, "batch_size", None),
        )
        st.t_start = time.perf_counter()
        preds = orig_video_inference(video, pose_runner, detector_runner, **kwargs)
        st.infer_seconds = time.perf_counter() - st.t_start
        st.phase = "save"
        _progress(force=True)
        return preds

    V.video_inference = video_inference


def analyze_pytorch(cmd: dict) -> None:
    import deeplabcut.pose_estimation_pytorch.apis.videos as V

    init = STATE.init
    dyn = cmd.get("dynamic") or [False, 0.5, 10]
    batch_size = int(init.get("batch_size") or 8)
    if dyn[0]:
        batch_size = 1  # DLC requires batch size 1 for dynamic cropping
    kwargs = dict(
        shuffle=int(init.get("shuffle", 1)),
        trainingsetindex=int(init.get("trainingsetindex", 0)),
        save_as_csv=bool(cmd.get("save_as_csv", False)),
        destfolder=cmd.get("destfolder") or None,
        batch_size=batch_size,
        dynamic=(bool(dyn[0]), float(dyn[1]), int(dyn[2])),
        overwrite=bool(cmd.get("overwrite", False)),
        cropping=cmd.get("cropping") or None,
        robust_nframes=bool(cmd.get("robust_nframes", False)),
        auto_track=bool(cmd.get("auto_track", True)),
    )
    if init.get("device") and init["device"] != "auto":
        kwargs["device"] = init["device"]
    if init.get("snapshot_index") is not None:
        kwargs["snapshot_index"] = init["snapshot_index"]
    if init.get("detector_snapshot_index") is not None:
        kwargs["detector_snapshot_index"] = init["detector_snapshot_index"]
    if init.get("detector_batch_size"):
        kwargs["detector_batch_size"] = int(init["detector_batch_size"])
    if cmd.get("n_tracks"):
        kwargs["n_tracks"] = int(cmd["n_tracks"])
    V.analyze_videos(init["config"], [cmd["video"]], **kwargs)


# --------------------------------------------------------------------------
# TensorFlow engine (best effort: progress only, no live confidence metrics)
# --------------------------------------------------------------------------
class _TFProgress:
    """Minimal tqdm stand-in that reports progress through events."""

    def __init__(self, iterable=None, total=None, *a, **k):
        self.iterable = iterable
        st = STATE
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = 0
        if not st.started:
            st.started = True
            st.n_frames = int(total or 0)
            st.phase = "pose"
            st.t_start = st.t_first = time.perf_counter()
            emit("video_start", video=st.video, n_frames=st.n_frames, fps=0.0,
                 width=0, height=0, has_detector=False, batch_size=None)

    def update(self, n=1):
        st = STATE
        st.frames_done = min(st.n_frames or 10**12, st.frames_done + int(n))
        st.t_last = time.perf_counter()
        _progress()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)

    def close(self):
        STATE.infer_seconds = time.perf_counter() - STATE.t_start
        _progress(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_description(self, *a, **k):
        pass


def install_tensorflow_hooks() -> None:
    import deeplabcut.pose_estimation_tensorflow.predict_videos as PV

    PV.tqdm = _TFProgress


def analyze_tensorflow(cmd: dict) -> None:
    import deeplabcut.pose_estimation_tensorflow as tfapi

    init = STATE.init
    dyn = cmd.get("dynamic") or [False, 0.5, 10]
    gpu = None
    dev = str(init.get("device") or "auto")
    if dev.startswith("cuda:"):
        gpu = int(dev.split(":")[1])
    tfapi.analyze_videos(
        init["config"],
        [cmd["video"]],
        shuffle=int(init.get("shuffle", 1)),
        trainingsetindex=int(init.get("trainingsetindex", 0)),
        gputouse=gpu,
        save_as_csv=bool(cmd.get("save_as_csv", False)),
        destfolder=cmd.get("destfolder") or None,
        batchsize=int(init.get("batch_size") or 8),
        cropping=cmd.get("cropping") or None,
        dynamic=(bool(dyn[0]), float(dyn[1]), int(dyn[2])),
        robust_nframes=bool(cmd.get("robust_nframes", False)),
        allow_growth=True,
        auto_track=bool(cmd.get("auto_track", True)),
    )


# --------------------------------------------------------------------------
# Command loop
# --------------------------------------------------------------------------
def do_init(cmd: dict) -> None:
    st = STATE
    st.init = cmd
    st.engine = str(cmd.get("engine") or "pytorch").lower()
    threads = int(cmd.get("cpu_threads") or 0)
    info: dict = {"engine": st.engine, "pid": os.getpid()}

    import deeplabcut  # noqa: F401  (slow import - done once per worker)

    info["dlc_version"] = getattr(deeplabcut, "__version__", "?")
    if st.engine == "pytorch":
        import torch

        if threads > 0:
            torch.set_num_threads(threads)
            try:
                import cv2

                cv2.setNumThreads(threads)
            except Exception:
                pass
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            torch.backends.cudnn.benchmark = bool(cmd.get("cudnn_benchmark", False))
            idx = 0
            dev = str(cmd.get("device") or "auto")
            if dev.startswith("cuda:"):
                idx = int(dev.split(":")[1])
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            info["gpu_total_mb"] = int(torch.cuda.get_device_properties(idx).total_memory / 2**20)
        install_pytorch_hooks()
    else:
        install_tensorflow_hooks()
    emit("ready", **info)


def do_analyze(cmd: dict) -> None:
    st = STATE
    st.reset_video(cmd)
    t0 = time.perf_counter()
    emit("video_load", video=st.video)
    try:
        if st.engine == "pytorch":
            analyze_pytorch(cmd)
        else:
            analyze_tensorflow(cmd)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        st.runner_cache.clear()  # the runner may hold a half-processed batch
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        msg = f"{type(exc).__name__}: {exc}"
        emit(
            "video_error",
            video=st.video,
            message=msg,
            oom="out of memory" in msg.lower(),
            traceback=traceback.format_exc(limit=12),
        )
        return

    total = time.perf_counter() - t0
    data = dict(
        video=st.video,
        skipped=not st.started,
        frames=st.frames_done,
        n_frames=st.n_frames,
        seconds=total,
        infer_seconds=st.infer_seconds,
    )
    if st.t_last > st.t_first > 0 and st.frames_done > st.frames_at_first:
        data["steady_fps"] = (st.frames_done - st.frames_at_first) / (st.t_last - st.t_first)
    if st.infer_seconds > 0:
        data["fps"] = st.frames_done / st.infer_seconds
    if st.conf_frames:
        data["mean_conf"] = [round(float(x), 4) for x in st.conf_sum / st.conf_frames]
        data["frac_conf"] = [round(float(x), 4) for x in st.conf_hits / st.conf_frames]
        data["lost_frac"] = round(st.lost_frames / st.conf_frames, 4)
    try:
        import torch

        if torch.cuda.is_available():
            data["vram_peak_mb"] = int(torch.cuda.max_memory_reserved() / 2**20)
    except Exception:
        pass
    emit("video_done", **data)


def main() -> int:
    global _OUT
    os.environ.setdefault("DLClight", "True")  # no GUI imports inside DLC
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
    _OUT = sys.stdout
    emit("hello", pid=os.getpid(), python=sys.executable)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            cmd = json.loads(raw)
        except ValueError:
            emit("log", message=f"bad command: {raw[:200]}")
            continue
        name = cmd.get("cmd")
        if name == "quit":
            break
        try:
            if name == "init":
                do_init(cmd)
            elif name == "analyze":
                do_analyze(cmd)
            elif name == "set":  # change options (e.g. batch_size) between videos
                STATE.init.update({k: v for k, v in cmd.items() if k != "cmd"})
                if int(cmd.get("cpu_threads") or 0) > 0 and STATE.engine == "pytorch":
                    import torch

                    torch.set_num_threads(int(cmd["cpu_threads"]))
                emit("set_ok", **{k: v for k, v in cmd.items() if k != "cmd"})
            else:
                emit("log", message=f"unknown command: {name}")
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            emit(
                "fatal",
                message=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=12),
            )
            return 1
    emit("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
