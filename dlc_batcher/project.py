"""Read-only inspection of a DeepLabCut project (config, shuffles, snapshots).

Nothing in this module imports deeplabcut, so it is fast and safe to call from
the GUI thread.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".flv", ".m4v")


@dataclass
class ModelInfo:
    """One trained shuffle found in the project."""

    engine: str  # "pytorch" | "tensorflow"
    iteration: int
    shuffle: int
    train_fraction: float
    trainingset_index: int
    folder: Path
    net_type: str = "?"
    method: str = "?"  # bu | td | ctd (pytorch only)
    snapshots: list[str] = field(default_factory=list)  # in DLC index order
    detector_snapshots: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        pct = int(round(self.train_fraction * 100))
        eng = "PyTorch" if self.engine == "pytorch" else "TensorFlow"
        return (
            f"shuffle {self.shuffle} · {self.net_type} · {eng} · "
            f"trainset {pct}% · iteration {self.iteration}"
        )

    @property
    def is_top_down(self) -> bool:
        return self.method == "td"


@dataclass
class ProjectInfo:
    config_path: Path
    project_path: Path
    task: str
    scorer: str
    date: str
    multianimal: bool
    engine: str
    bodyparts: list[str]
    individuals: list[str]
    pcutoff: float
    batch_size: int
    iteration: int
    snapshot_index: object
    cropping: list[int] | None  # [x1, x2, y1, y2] when enabled in config
    training_fractions: list[float]
    training_videos: list[str]  # stems of videos that contributed labeled frames
    models: list[ModelInfo]
    raw: dict

    def labeled_frame_counts(self) -> dict[str, int]:
        """Number of labeled frames per video folder in labeled-data."""
        counts: dict[str, int] = {}
        root = self.project_path / "labeled-data"
        if not root.is_dir():
            return counts
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name.endswith("_labeled"):
                continue
            n = 0
            for csv in folder.glob("CollectedData_*.csv"):
                try:
                    with open(csv, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [ln for ln in f if ln.strip()]
                    header = 4 if self.multianimal else 3
                    n = max(n, len(lines) - header)
                except OSError:
                    pass
            counts[folder.name] = n
        return counts


def _sorted_pt_snapshots(train_dir: Path, prefix: str) -> list[str]:
    """Mimic DLC's TorchSnapshotManager ordering: by epoch, 'best' last."""
    regular, best = [], []
    for p in train_dir.glob(f"{prefix}-*.pt"):
        m = re.fullmatch(rf"{re.escape(prefix)}-(best-)?(\d+)", p.stem)
        if not m:
            continue
        (best if m.group(1) else regular).append((int(m.group(2)), p.name))
    return [n for _, n in sorted(regular)] + [n for _, n in sorted(best)]


def _sorted_tf_snapshots(train_dir: Path) -> list[str]:
    found = []
    for p in train_dir.glob("snapshot-*.index"):
        m = re.fullmatch(r"snapshot-(\d+)", p.stem)
        if m:
            found.append((int(m.group(1)), p.stem))
    return [n for _, n in sorted(found)]


def _scan_models(project_path: Path, fractions: list[float]) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for root_name, engine in (("dlc-models-pytorch", "pytorch"), ("dlc-models", "tensorflow")):
        root = project_path / root_name
        if not root.is_dir():
            continue
        for it_dir in sorted(root.glob("iteration-*")):
            try:
                iteration = int(it_dir.name.split("-")[-1])
            except ValueError:
                continue
            for folder in sorted(it_dir.iterdir()):
                m = re.search(r"-trainset(\d+)shuffle(\d+)$", folder.name)
                train_dir = folder / "train"
                if not m or not train_dir.is_dir():
                    continue
                pct, shuffle = int(m.group(1)), int(m.group(2))
                ts_index = 0
                for i, frac in enumerate(fractions):
                    if int(round(float(frac) * 100)) == pct:
                        ts_index = i
                        break
                info = ModelInfo(
                    engine=engine,
                    iteration=iteration,
                    shuffle=shuffle,
                    train_fraction=pct / 100.0,
                    trainingset_index=ts_index,
                    folder=folder,
                )
                if engine == "pytorch":
                    info.snapshots = _sorted_pt_snapshots(train_dir, "snapshot")
                    info.detector_snapshots = _sorted_pt_snapshots(
                        train_dir, "snapshot-detector"
                    )
                    cfg_file = train_dir / "pytorch_config.yaml"
                    if cfg_file.is_file():
                        try:
                            with open(cfg_file, "r", encoding="utf-8") as f:
                                mcfg = yaml.safe_load(f) or {}
                            info.net_type = str(mcfg.get("net_type", "?"))
                            info.method = str(mcfg.get("method", "?")).lower()
                        except Exception:
                            pass
                else:
                    info.snapshots = _sorted_tf_snapshots(train_dir)
                    cfg_file = train_dir / "pose_cfg.yaml"
                    if cfg_file.is_file():
                        try:
                            with open(cfg_file, "r", encoding="utf-8") as f:
                                mcfg = yaml.safe_load(f) or {}
                            info.net_type = str(mcfg.get("net_type", "?"))
                        except Exception:
                            pass
                if info.snapshots:
                    models.append(info)
    # Newest iteration first, then shuffle
    models.sort(key=lambda mi: (-mi.iteration, mi.shuffle, mi.engine))
    return models


def load_project(config_path: str | Path) -> ProjectInfo:
    """Parse a DLC ``config.yaml`` (or a project folder containing one)."""
    config_path = Path(config_path)
    if config_path.is_dir():
        config_path = config_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.yaml found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "Task" not in cfg or "TrainingFraction" not in cfg:
        raise ValueError(f"{config_path} does not look like a DeepLabCut config.yaml")

    # Trust the file location over the (often stale) project_path entry
    project_path = config_path.parent
    multi = bool(cfg.get("multianimalproject", False))
    if multi:
        bodyparts = list(cfg.get("multianimalbodyparts") or [])
        individuals = list(cfg.get("individuals") or [])
    else:
        bodyparts = list(cfg.get("bodyparts") or [])
        individuals = ["animal"]

    cropping = None
    if cfg.get("cropping"):
        try:
            cropping = [int(cfg["x1"]), int(cfg["x2"]), int(cfg["y1"]), int(cfg["y2"])]
        except (KeyError, TypeError, ValueError):
            cropping = None

    fractions = [float(x) for x in (cfg.get("TrainingFraction") or [0.95])]
    video_sets = cfg.get("video_sets") or {}
    training_videos = sorted({Path(str(v)).stem for v in video_sets})

    return ProjectInfo(
        config_path=config_path,
        project_path=project_path,
        task=str(cfg.get("Task", "")),
        scorer=str(cfg.get("scorer", "")),
        date=str(cfg.get("date", "")),
        multianimal=multi,
        engine=str(cfg.get("engine") or "tensorflow").lower(),
        bodyparts=[str(b) for b in bodyparts],
        individuals=[str(i) for i in individuals],
        pcutoff=float(cfg.get("pcutoff", 0.6) or 0.6),
        batch_size=int(cfg.get("batch_size", 8) or 8),
        iteration=int(cfg.get("iteration", 0) or 0),
        snapshot_index=cfg.get("snapshotindex", -1),
        cropping=cropping,
        training_fractions=fractions,
        training_videos=training_videos,
        models=_scan_models(project_path, fractions),
        raw=cfg,
    )


def list_videos(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Videos in a folder, skipping DLC's own labeled-video outputs."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    out = []
    for p in it:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not is_dlc_output_video(p):
            out.append(p)
    return sorted(out)


def is_dlc_output_video(path: Path) -> bool:
    stem = path.stem
    return bool(re.search(r"DLC_.+_(labeled|full)$", stem)) or stem.endswith("_labeled")
