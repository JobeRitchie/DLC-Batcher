"""Inference-quality metrics across analyzed videos (pure NumPy / pandas).

There is no ground truth for a freshly analyzed video, so quality is judged
from signals that tend to go wrong when a model struggles:

* Confidence: how often each keypoint is above the project's p-cutoff.
* Jumps: a confident keypoint moving implausibly far between two frames.
* Odd poses: distances between keypoints far from their usual value.
* Dropouts: stretches where the animal is not found at all.

Each video gets a 0-100 score and is compared with the rest of the batch, so
one bad recording stands out even when the model is good overall.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .project import VIDEO_EXTENSIONS, ProjectInfo

JUMP_FRACTION = 0.30  # a jump = moving > 30% of body length in one frame
TIMELINE_BINS = 240


@dataclass
class VideoQuality:
    h5_path: str
    video_name: str  # video stem
    video_path: str | None
    scorer: str
    n_frames: int
    fps: float
    labels: list[str]  # keypoint labels (individual/bodypart when multi-animal)
    frac_conf: np.ndarray  # per keypoint, fraction of frames >= pcutoff
    mean_conf: np.ndarray  # per keypoint
    jump_rate: np.ndarray  # per keypoint, jumps per 1000 frames
    confident: float  # mean of frac_conf
    lost: float  # fraction of frames with no keypoint above cutoff
    complete: float  # fraction of frames with all keypoints above cutoff
    jumps_per_1k: float
    odd_pose: float  # fraction of frames with implausible geometry
    jitter: float  # median step of confident points, in body lengths
    body_length: float  # px
    longest_gap_s: float  # longest stretch with < half the keypoints confident
    timeline_conf: np.ndarray  # binned mean confidence
    timeline_bad: np.ndarray  # binned fraction of "bad" frames
    worst_frame: int
    in_training: bool | None = None
    resolution: tuple | None = None
    score: float = 0.0
    status: str = "good"  # good | check | problem
    reasons: list[str] = field(default_factory=list)
    outlier: bool = False


def scorer_of(h5: Path) -> str | None:
    m = re.search(r"(DLC_.+?)(_el|_filtered|_sk|_bx)?$", h5.stem)
    return m.group(1) if m else None


def find_results(folders: list[str | Path]) -> dict[str, list[Path]]:
    """Map scorer name -> result .h5 files found in the given folders."""
    out: dict[str, list[Path]] = {}
    seen = set()
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for h5 in sorted(folder.glob("*DLC_*.h5")):
            stem = h5.stem
            if stem.endswith(("_filtered", "_sk", "_bx")) or h5 in seen:
                continue
            sc = scorer_of(h5)
            if sc:
                seen.add(h5)
                out.setdefault(sc, []).append(h5)
    # prefer stitched multi-animal tracks (_el) over raw detections when both exist
    for sc, files in out.items():
        stems = {f.stem for f in files}
        out[sc] = [f for f in files if not (f.stem + "_el") in stems]
    return out


def _video_for(h5: Path, scorer: str) -> tuple[str, Path | None]:
    stem = h5.stem
    name = stem[: stem.index(scorer)] if scorer in stem else stem
    for ext in VIDEO_EXTENSIONS:
        cand = h5.parent / (name + ext)
        if cand.is_file():
            return name, cand
    return name, None


def _meta(h5: Path) -> dict:
    import pickle

    stem = h5.stem
    for suffix in ("_el",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    pkl = h5.parent / f"{stem}_meta.pickle"
    try:
        with open(pkl, "rb") as f:
            return pickle.load(f).get("data", {})
    except Exception:
        return {}


def load_predictions(h5: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (xy [T, K, 2], likelihood [T, K], labels[K])."""
    import pandas as pd

    df = pd.read_hdf(h5)
    cols = df.columns
    names = list(cols.names)
    has_ind = "individuals" in names
    labels: list[str] = []
    seen = []
    for col in cols:
        key = col[:-1]
        if key not in seen:
            seen.append(key)
    for key in seen:
        bp = key[names.index("bodyparts")]
        if has_ind:
            ind = key[names.index("individuals")]
            labels.append(f"{ind}/{bp}" if ind != "single" else str(bp))
        else:
            labels.append(str(bp))
    arr = df.to_numpy(dtype=np.float32)
    k = len(seen)
    arr = arr[:, : k * 3].reshape(len(arr), k, 3)
    xy = arr[..., :2]
    lik = np.nan_to_num(arr[..., 2], nan=0.0)
    return xy, lik, labels


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


def _bin_mean(values: np.ndarray, bins: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    bins = max(1, min(bins, n))
    edges = np.linspace(0, n, bins + 1).astype(int)
    sums = np.add.reduceat(values.astype(np.float64), edges[:-1])
    counts = np.maximum(np.diff(edges), 1)
    return (sums / counts).astype(np.float32)


def analyze_file(h5: str | Path, pcutoff: float = 0.6, scorer: str | None = None) -> VideoQuality:
    h5 = Path(h5)
    scorer = scorer or scorer_of(h5) or ""
    xy, lik, labels = load_predictions(h5)
    T, K = lik.shape
    meta = _meta(h5)
    fps = float(meta.get("fps") or 0.0)
    name, video = _video_for(h5, scorer)

    ok = lik >= pcutoff
    frac_conf = ok.mean(axis=0) if T else np.zeros(K)
    mean_conf = lik.mean(axis=0) if T else np.zeros(K)
    n_ok = ok.sum(axis=1)
    lost = float((n_ok == 0).mean()) if T else 1.0
    complete = float((n_ok == K).mean()) if T else 0.0

    # Body length: spread of confident keypoints, robust median over frames
    xy_ok = np.where(ok[..., None], xy, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        span = np.nanmax(xy_ok, axis=1) - np.nanmin(xy_ok, axis=1)
        size = np.hypot(span[:, 0], span[:, 1])
        size[n_ok < max(2, K // 2)] = np.nan
        body = float(np.nanmedian(size)) if np.isfinite(size).any() else 0.0
    if not body or not np.isfinite(body):
        body = 1.0

    # Jumps and jitter between consecutive confident detections
    if T > 1:
        step = np.hypot(*(np.diff(xy, axis=0).transpose(2, 0, 1)))
        both = ok[1:] & ok[:-1]
        jump = both & (step > JUMP_FRACTION * body)
        jump_rate = jump.sum(axis=0) / max(1, T) * 1000.0
        valid_steps = step[both]
        jitter = float(np.median(valid_steps) / body) if valid_steps.size else 0.0
        jump_any = np.concatenate([[False], jump.any(axis=1)])
    else:
        jump_rate = np.zeros(K)
        jitter = 0.0
        jump_any = np.zeros(T, dtype=bool)

    # Odd poses: a pairwise distance far from its usual value, or body size off
    odd = np.zeros(T, dtype=bool)
    if K >= 2 and T > 10:
        with np.errstate(all="ignore"):
            for i in range(K):
                for j in range(i + 1, K):
                    d = np.hypot(xy[:, i, 0] - xy[:, j, 0], xy[:, i, 1] - xy[:, j, 1])
                    pair_ok = ok[:, i] & ok[:, j]
                    if pair_ok.sum() < 20:
                        continue
                    med = float(np.median(d[pair_ok]))
                    floor = 0.15 * body
                    odd |= pair_ok & (d > max(3.0 * med, med + floor * 3))
            odd |= np.nan_to_num(size, nan=0.0) > 2.2 * body
    odd_pose = float(odd.mean()) if T else 0.0

    weak = n_ok < np.ceil(K / 2)
    gap = _longest_run(weak)
    longest_gap_s = gap / fps if fps > 0 else float(gap)

    bad = weak | jump_any | odd
    frame_conf = lik.mean(axis=1) if T else np.zeros(0)
    tl_conf = _bin_mean(frame_conf, TIMELINE_BINS)
    tl_bad = _bin_mean(bad.astype(np.float32), TIMELINE_BINS)
    worst_frame = 0
    if T:
        worst_bin = int(np.argmax(tl_bad - 1e-3 * tl_conf))
        lo = int(worst_bin * T / len(tl_bad))
        hi = max(lo + 1, int((worst_bin + 1) * T / len(tl_bad)))
        worst_frame = lo + int(np.argmin(frame_conf[lo:hi]))

    res = meta.get("frame_dimensions")
    return VideoQuality(
        h5_path=str(h5),
        video_name=name,
        video_path=str(video) if video else None,
        scorer=scorer,
        n_frames=T,
        fps=fps,
        labels=labels,
        frac_conf=np.asarray(frac_conf, dtype=np.float32),
        mean_conf=np.asarray(mean_conf, dtype=np.float32),
        jump_rate=np.asarray(jump_rate, dtype=np.float32),
        confident=float(np.mean(frac_conf)) if K else 0.0,
        lost=lost,
        complete=complete,
        jumps_per_1k=float(jump_any.sum() / max(1, T) * 1000.0),
        odd_pose=odd_pose,
        jitter=jitter,
        body_length=body,
        longest_gap_s=longest_gap_s,
        timeline_conf=tl_conf,
        timeline_bad=tl_bad,
        worst_frame=worst_frame,
        resolution=tuple(res) if res else None,
    )


def _score(q: VideoQuality) -> float:
    """0-100. Confidence carries half the weight; the rest penalizes jumps,
    impossible poses and frames where the animal is lost."""
    conf = q.confident
    jumps = 1.0 - min(1.0, q.jumps_per_1k / 50.0)  # 50 jump-frames per 1000 = 0
    poses = 1.0 - min(1.0, q.odd_pose / 0.10)  # 10% odd frames = 0
    found = 1.0 - min(1.0, q.lost / 0.20)  # 20% lost frames = 0
    return 100.0 * (0.50 * conf + 0.20 * jumps + 0.15 * poses + 0.15 * found)


def assess(videos: list[VideoQuality], project: ProjectInfo | None = None) -> None:
    """Fill score / status / reasons, flagging outliers within the batch."""
    if not videos:
        return
    for q in videos:
        q.score = _score(q)
        if project is not None:
            q.in_training = q.video_name in project.training_videos
    scores = np.array([q.score for q in videos])
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826
    res_counts: dict = {}
    for q in videos:
        if q.resolution:
            res_counts[q.resolution] = res_counts.get(q.resolution, 0) + 1
    for q in videos:
        q.reasons = []
        q.outlier = len(videos) >= 5 and q.score < med - max(3.0 * mad, 5.0)
        if q.confident < 0.80:
            worst = [q.labels[i] for i in np.argsort(q.frac_conf)[:2] if q.frac_conf[i] < 0.8]
            q.reasons.append(
                f"only {q.confident:.0%} of keypoints confident"
                + (f" (worst: {', '.join(worst)})" if worst else "")
            )
        if q.lost > 0.02:
            q.reasons.append(f"animal not found in {q.lost:.1%} of frames")
        if q.jumps_per_1k > 5:
            q.reasons.append(f"{q.jumps_per_1k:.0f} jumps per 1000 frames")
        if q.odd_pose > 0.02:
            q.reasons.append(f"implausible pose in {q.odd_pose:.1%} of frames")
        if q.longest_gap_s > 2 and q.fps > 0:
            q.reasons.append(f"tracking lost for {q.longest_gap_s:.1f} s at one point")
        if q.outlier:
            q.reasons.append("clearly worse than the other videos")
        if q.score >= 85 and not q.outlier and q.lost <= 0.02:
            q.status = "good"
        elif q.score >= 65:
            q.status = "check"
        else:
            q.status = "problem"


@dataclass
class Advice:
    level: str  # good | info | warn | bad
    title: str
    detail: str
    command: str = ""


def recommendations(
    videos: list[VideoQuality], project: ProjectInfo | None = None, pcutoff: float = 0.6
) -> list[Advice]:
    """Plain-language guidance for improving the model."""
    out: list[Advice] = []
    if not videos:
        return out
    cfg = str(project.config_path) if project else "path/to/config.yaml"
    problems = [q for q in videos if q.status == "problem"]
    checks = [q for q in videos if q.status == "check"]
    n = len(videos)

    # ---- overall verdict
    med_score = float(np.median([q.score for q in videos]))
    if not problems and not checks:
        out.append(Advice("good", "The model handles this batch well",
                          f"All {n} videos score ≥ 85 (median {med_score:.0f}). "
                          "No retraining needed for these recordings."))
    elif len(problems) + len(checks) <= max(1, n // 4):
        out.append(Advice("info", "Most videos are fine; a few need attention",
                          f"{n - len(problems) - len(checks)} of {n} videos look good "
                          f"(median score {med_score:.0f}). The issues are specific to a "
                          "few recordings rather than the model as a whole."))
    else:
        out.append(Advice("bad", "The model struggles on much of this batch",
                          f"{len(problems)} problem and {len(checks)} borderline videos "
                          f"out of {n} (median score {med_score:.0f}). This points to the "
                          "model itself: more (and more varied) labeled frames, and "
                          "possibly longer training, are the usual fix."))

    # ---- keypoints that are weak everywhere
    labels = videos[0].labels
    same = [q for q in videos if q.labels == labels]
    if same:
        frac = np.stack([q.frac_conf for q in same])
        jumps = np.stack([q.jump_rate for q in same])
        med_frac = np.median(frac, axis=0)
        overall = float(np.median(med_frac))
        weak = [i for i in np.argsort(med_frac) if med_frac[i] < min(0.85, overall - 0.07)]
        if weak:
            names = ", ".join(f"{labels[i]} ({med_frac[i]:.0%})" for i in weak[:4])
            out.append(Advice(
                "warn", f"Weak keypoint{'s' if len(weak) > 1 else ''}: "
                + ", ".join(labels[i] for i in weak[:4]),
                f"Confident in a typical video: {names}, versus {overall:.0%} for a typical "
                "keypoint. If the part is genuinely hidden in those frames this is expected. "
                "Otherwise label more frames where it is visible but hard (odd angles, "
                "partial occlusion) and re-check existing labels for consistency."))
        jumpy = [i for i in np.argsort(-np.median(jumps, axis=0))
                 if np.median(jumps[:, i]) > 3 and med_frac[i] >= 0.85]
        if jumpy:
            out.append(Advice(
                "warn", "Confident but jumpy: " + ", ".join(labels[i] for i in jumpy[:4]),
                "These keypoints are reported with high confidence yet jump between "
                "frames, typically from left/right swaps or a similar-looking body part. Label frames "
                "from the moments where they jump (outlier extraction with the 'jump' "
                "algorithm finds them)."))

    # ---- problem videos and what to do with them
    worst = sorted(problems + checks, key=lambda q: q.score)[:6]
    if worst:
        untrained = [q for q in worst if q.in_training is False]
        lines = [f"• {q.video_name} (score {q.score:.0f}): {'; '.join(q.reasons) or 'borderline'}"
                 for q in worst]
        detail = "\n".join(lines)
        if untrained:
            detail += (
                f"\n\n{len(untrained)} of these contributed no labeled frames to the "
                "training set, so the model has never seen them. Extracting outlier frames "
                "from them, correcting the labels and retraining is the usual fix."
            )
        else:
            detail += (
                "\n\nExtract outlier frames from these videos, correct them in the DLC "
                "labeling GUI, merge and retrain."
            )
        paths = [q.video_path or q.video_name for q in worst[:3]]
        jump_dom = sum(q.jumps_per_1k > 5 for q in worst) > sum(q.confident < 0.8 for q in worst)
        algo = "jump" if jump_dom else "uncertain"
        vids = ", ".join(f'r"{p}"' for p in paths)
        cmd = (
            "import deeplabcut\n"
            f'deeplabcut.extract_outlier_frames(r"{cfg}", [{vids}], '
            f'outlieralgorithm="{algo}")\n'
            f'deeplabcut.refine_labels(r"{cfg}")  # then merge_datasets + retrain'
        )
        out.append(Advice("bad" if problems else "warn",
                          "Videos to refine the model on", detail, cmd))

    # ---- issues concentrated in time
    for q in sorted(videos, key=lambda q: q.score)[:3]:
        tl = q.timeline_bad
        if len(tl) < 20 or tl.sum() <= 0 or q.status == "good":
            continue
        k = max(1, len(tl) // 10)
        top = np.sort(tl)[-k:].sum() / tl.sum()
        if top > 0.6:
            peak = int(np.argmax(tl))
            when = peak / len(tl) * q.n_frames
            stamp = f"{when / q.fps / 60:.1f} min" if q.fps > 0 else f"frame {int(when)}"
            out.append(Advice(
                "info", f"{q.video_name}: problems are concentrated around {stamp}",
                f"{top:.0%} of the bad frames fall within 10% of the video. That usually "
                "means an event (hand in the arena, lighting change, animal out of view) "
                "rather than a general model weakness. Click the timeline to look at it."))

    # ---- resolution mismatch
    res = {}
    for q in videos:
        if q.resolution:
            res.setdefault(q.resolution, []).append(q)
    if len(res) > 1:
        major = max(res.values(), key=len)
        others = [q for r, qs in res.items() if qs is not major for q in qs]
        worse = [q for q in others if q.score < np.median([m.score for m in major]) - 5]
        if worse:
            out.append(Advice(
                "warn", "Different frame size, lower quality",
                f"{len(worse)} video(s) with a different resolution than the rest score "
                "lower (" + ", ".join(q.video_name for q in worse[:4]) + "). "
                "Make sure this camera / crop is represented in the training set."))

    # ---- training set size
    if project is not None:
        counts = project.labeled_frame_counts()
        total = sum(counts.values())
        if total:
            if total < 150:
                out.append(Advice(
                    "warn", f"Small training set ({total} labeled frames)",
                    f"{total} labeled frames from {len(counts)} videos. DeepLabCut projects typically "
                    "need about 200 or more diverse frames for robust tracking. Outlier frames "
                    "from the lowest-scoring videos are a good place to start."))
            else:
                out.append(Advice(
                    "info", f"Training set: {total} labeled frames from {len(counts)} videos",
                    "The size is reasonable. Further gains are more likely from labeling "
                    "hard cases than from adding more frames of the same kind."))

    # ---- noise floor
    jit = float(np.median([q.jitter for q in videos]))
    if jit > 0.04:
        out.append(Advice(
            "info", "Predictions are jittery",
            f"Confident keypoints move a median of ~{jit:.1%} of a body length per frame. "
            "deeplabcut.filterpredictions() (median filter) smooths this for "
            "downstream analysis."))
    return out


def analyze_folder(folders: list[str | Path], pcutoff: float = 0.6,
                   scorer: str | None = None, project: ProjectInfo | None = None,
                   progress=None) -> tuple[list[VideoQuality], str | None, list[str]]:
    """Analyze all result files for one scorer. Returns (videos, scorer, all scorers)."""
    found = find_results(folders)
    if not found:
        return [], None, []
    scorers = sorted(found, key=lambda s: -max(f.stat().st_mtime for f in found[s]))
    scorer = scorer if scorer in found else scorers[0]
    files = found[scorer]
    videos = []
    for i, h5 in enumerate(files):
        if progress:
            progress(i, len(files), h5.name)
        try:
            videos.append(analyze_file(h5, pcutoff, scorer))
        except Exception as exc:  # unreadable file: skip, keep going
            if progress:
                progress(i, len(files), f"skipped {h5.name}: {exc}")
    assess(videos, project)
    return videos, scorer, scorers
