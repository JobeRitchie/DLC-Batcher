"""Tests that need neither a GPU nor DeepLabCut (run with: python -m pytest)."""
import numpy as np
import pandas as pd
import pytest

from dlc_batcher import hardware, quality
from dlc_batcher.calibration import bundled_clip, estimate_speedup
from dlc_batcher.project import is_dlc_output_video, load_project

SCORER = "DLC_Resnet50_demoJan1shuffle1_snapshot_100"
PARTS = ["nose", "ear", "tail"]


def write_h5(folder, name, xy, lik):
    cols = pd.MultiIndex.from_product([[SCORER], PARTS, ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    data = np.concatenate([xy, lik[..., None]], axis=2).reshape(len(xy), -1)
    path = folder / f"{name}{SCORER}.h5"
    pd.DataFrame(data, columns=cols).to_hdf(path, key="df_with_missing", mode="w")
    return path


def clean_track(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    centre = np.cumsum(rng.normal(0, 0.4, (n, 1, 2)), axis=0) + 200
    offsets = np.array([[30, 0], [15, 10], [-30, 0]], dtype=float)[None]
    xy = centre + offsets + rng.normal(0, 0.3, (n, 3, 2))
    lik = np.full((n, 3), 0.97)
    return xy.astype(np.float32), lik.astype(np.float32)


def test_quality_flags_the_bad_video(tmp_path):
    for i in range(5):
        xy, lik = clean_track(seed=i)
        write_h5(tmp_path, f"good{i}", xy, lik)
    xy, lik = clean_track(seed=9)
    lik[600:1400, 2] = 0.1  # tail lost for a long stretch
    lik[900:1100] = 0.05  # animal lost entirely
    xy[300:400:5, 0] += 120  # nose teleports
    write_h5(tmp_path, "bad", xy, lik)

    videos, scorer, scorers = quality.analyze_folder([tmp_path], pcutoff=0.6)
    assert scorer == SCORER and scorers == [SCORER] and len(videos) == 6
    by = {v.video_name: v for v in videos}
    assert all(by[f"good{i}"].status == "good" for i in range(5))
    bad = by["bad"]
    assert bad.status != "good" and bad.outlier
    assert bad.score < min(by[f"good{i}"].score for i in range(5)) - 10
    assert bad.jumps_per_1k > 5 and bad.lost > 0.05
    assert bad.frac_conf[2] < bad.frac_conf[1]
    advice = quality.recommendations(videos)
    assert any("bad" in a.detail and a.command for a in advice)


def test_load_project_and_models(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "Task: demo\nscorer: me\ndate: Jan1\nmultianimalproject: false\nengine: pytorch\n"
        "bodyparts: [nose, ear, tail]\nTrainingFraction: [0.95]\niteration: 0\n"
        "snapshotindex: -1\npcutoff: 0.6\nbatch_size: 8\ncropping: false\n"
        "video_sets:\n  C:\\vids\\a.mp4:\n    crop: 0, 10, 0, 10\n")
    train = tmp_path / "dlc-models-pytorch" / "iteration-0" / "demoJan1-trainset95shuffle3" / "train"
    train.mkdir(parents=True)
    for name in ("snapshot-100.pt", "snapshot-050.pt", "snapshot-best-070.pt",
                 "snapshot-detector-020.pt"):
        (train / name).write_bytes(b"")
    (train / "pytorch_config.yaml").write_text("net_type: resnet_50\nmethod: td\n")
    proj = load_project(tmp_path)
    assert proj.task == "demo" and proj.bodyparts == PARTS and proj.cropping is None
    assert proj.training_videos == ["a"]
    (model,) = proj.models
    assert (model.shuffle, model.net_type, model.is_top_down) == (3, "resnet_50", True)
    # DLC order: by epoch, "best" last
    assert model.snapshots == ["snapshot-050.pt", "snapshot-100.pt", "snapshot-best-070.pt"]
    assert model.detector_snapshots == ["snapshot-detector-020.pt"]
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nope")


def test_defaults_ignore_current_load():
    hw = hardware.Hardware("cpu", 24, 24, 64.0, [hardware.GpuInfo(0, "Big GPU", 24564)])
    d = hardware.default_settings(hw)
    assert d == {"batch_size": 64, "workers": 1, "device": "auto"}
    small = hardware.Hardware("cpu", 8, 8, 16.0, [hardware.GpuInfo(0, "Small", 4096)])
    assert hardware.default_settings(small)["batch_size"] == 8
    assert hardware.default_settings(hardware.Hardware("cpu", 8, 8, 16.0))["device"] == "cpu"
    assert hardware.gpu_is_shared(hardware.GpuLoad(0, 6000, 24564, 5))
    assert not hardware.gpu_is_shared(hardware.GpuLoad(0, 900, 24564, 2))


def test_speedup_estimate():
    trial = lambda b, w, fps, **k: dict(batch_size=b, workers=w, fps=fps, prefetch=True, **k)
    prof = dict(baseline_fps=100.0, fps=400.0, batch_size=64, workers=1, amp_fps=500.0,
                trials=[dict(batch_size=8, workers=1, fps=100.0, prefetch=False),
                        trial(8, 1, 120.0), trial(64, 1, 400.0), trial(64, 2, 300.0),
                        trial(64, 1, 500.0, amp=True)])
    est = estimate_speedup(prof, 64, 1)
    assert est["exact"] and est["speedup"] == pytest.approx(4.0)
    assert estimate_speedup(prof, 64, 1, amp=True)["speedup"] == pytest.approx(5.0)
    assert not estimate_speedup(prof, 32, 1)["exact"]
    assert estimate_speedup(None, 8, 1) is None


def test_misc():
    from pathlib import Path

    assert bundled_clip().is_file()
    assert is_dlc_output_video(Path("m1DLC_Resnet50_xshuffle1_snapshot_10_p60_labeled.mp4"))
    assert not is_dlc_output_video(Path("mouse1.mp4"))
