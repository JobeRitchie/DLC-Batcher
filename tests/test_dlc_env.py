"""Finding the DeepLabCut environment, on fake environment folders."""
import os
import sys

import pytest

from dlc_batcher import dlc_env

WIN = os.name == "nt"


def make_env(root, version=None, kind="venv"):
    """Create a fake environment folder and return its interpreter path."""
    if WIN:
        exe = root / ("Scripts/python.exe" if kind == "venv" else "python.exe")
        site = root / "Lib" / "site-packages"
    else:
        exe = root / "bin" / "python"
        site = root / "lib" / "python3.10" / "site-packages"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    site.mkdir(parents=True, exist_ok=True)
    if version:
        (site / f"deeplabcut-{version}.dist-info").mkdir()
        (site / "deeplabcut").mkdir()
    return exe


@pytest.fixture
def home(tmp_path, monkeypatch):
    for var in ("ProgramData", "CONDA_PREFIX", "CONDA_ROOT", dlc_env.ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "home"


def find(home, tmp_path, **kw):
    kw.setdefault("running", str(tmp_path / "no" / "python.exe"))
    kw.setdefault("environ", {})
    return dlc_env.find_candidates(home=home, app_dir=tmp_path / "app", **kw)


def test_version_order():
    versions = ["3.0.2", "3.0.0rc8", "2.3.11", "3.0.0", "3.0.0rc13", "3.1"]
    assert sorted(versions, key=dlc_env.version_key) == [
        "2.3.11", "3.0.0rc8", "3.0.0rc13", "3.0.0", "3.0.2", "3.1"]


def test_version_read_from_dist_info(tmp_path):
    exe = make_env(tmp_path / "env", "3.0.2")
    assert dlc_env.dlc_version(exe) == "3.0.2"
    assert dlc_env.dlc_version(make_env(tmp_path / "plain")) is None
    # a folder that only looks similar must not count
    other = make_env(tmp_path / "other")
    (dlc_env.site_packages(other)[0] / "deeplabcut_live-1.0.dist-info").mkdir()
    (dlc_env.site_packages(other)[0] / "napari_deeplabcut-0.2.dist-info").mkdir()
    assert dlc_env.dlc_version(other) is None


def test_no_environment(home, tmp_path):
    make_env(home / "anaconda3" / "envs" / "other", kind="conda")
    assert find(home, tmp_path) == []
    assert dlc_env.resolve({}, candidates=[], environ={}) == (None, [])


def test_single_environment(home, tmp_path):
    exe = make_env(home / "miniconda3" / "envs" / "dlc", "3.0.0rc8", kind="conda")
    (cand,) = find(home, tmp_path)
    assert (cand.python, cand.version, cand.source) == (str(exe), "3.0.0rc8", "conda")
    assert cand.tested


def test_newest_version_is_the_default(home, tmp_path):
    make_env(home / "anaconda3" / "envs" / "DEEPLABCUT", "3.0.0rc8", kind="conda")
    venv = make_env(home / "Desktop" / "DeepLabCut" / "venv", "3.0.2")
    make_env(tmp_path / "app" / ".venv", "2.3.9")
    cands = find(home, tmp_path)
    assert [c.version for c in cands] == ["3.0.2", "3.0.0rc8", "2.3.9"]
    choice, _ = dlc_env.resolve({}, candidates=cands, environ={})
    assert choice.python == str(venv) and choice.source == "venv"
    assert not cands[-1].tested


def test_running_interpreter_and_env_var_are_candidates(home, tmp_path):
    running = make_env(tmp_path / "here", "3.0.2")
    forced = make_env(tmp_path / "elsewhere", "3.0.0rc8", kind="conda")
    cands = find(home, tmp_path, running=str(running), environ={dlc_env.ENV_VAR: str(forced)})
    assert {c.source for c in cands} == {"app", "env var"}
    # the same interpreter found twice is listed once
    again = find(home, tmp_path, running=str(running), environ={dlc_env.ENV_VAR: str(running)})
    assert len(again) == 1


def test_resolve_order(home, tmp_path):
    new = make_env(home / "Desktop" / "DeepLabCut" / "venv", "3.0.2")
    old = make_env(home / "anaconda3" / "envs" / "old", "3.0.0rc8", kind="conda")
    forced = make_env(tmp_path / "forced", "3.0.0rc8")
    cands = find(home, tmp_path)
    # a saved choice wins over the newest version
    store = {"python_exe": str(old)}
    assert dlc_env.resolve(store, candidates=cands, environ={})[0].python == str(old)
    # the environment variable wins over both
    env = {dlc_env.ENV_VAR: str(forced)}
    assert dlc_env.resolve(store, candidates=cands, environ=env)[0].python == str(forced)
    # an automatic choice is not sticky, and a saved path that is gone is ignored
    auto = {"python_exe": str(old), "python_exe_picked": False}
    assert dlc_env.resolve(auto, candidates=cands, environ={})[0].python == str(new)
    gone = {"python_exe": str(tmp_path / "gone" / "python.exe")}
    assert dlc_env.resolve(gone, candidates=cands, environ={})[0].python == str(new)


def test_validate(tmp_path):
    good = make_env(tmp_path / "good", "3.0.2")
    cand, why = dlc_env.validate(str(good), run=False)
    assert cand.version == "3.0.2" and why == ""
    # the environment folder is accepted too
    assert dlc_env.validate(str(tmp_path / "good"), run=False)[0].python == str(good)
    for bad in ("", str(tmp_path / "missing" / "python.exe"),
                str(make_env(tmp_path / "plain"))):
        cand, why = dlc_env.validate(bad, run=False)
        assert cand is None and why
    not_python = tmp_path / "good" / "notes.txt"
    not_python.write_text("x")
    assert dlc_env.validate(str(not_python), run=False)[0] is None
    # an interpreter that cannot start is rejected when it is run
    assert dlc_env.validate(str(good), run=True)[0] is None


def test_validate_runs_the_real_interpreter(tmp_path):
    cand, why = dlc_env.validate(sys.executable)
    if dlc_env.dlc_version(sys.executable) is None:
        assert cand is None and "not installed" in why
    else:
        assert cand is not None


def test_remember_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DLC_BATCHER_HOME", str(tmp_path / "cfg"))
    monkeypatch.delenv(dlc_env.ENV_VAR, raising=False)
    exe = make_env(tmp_path / "env", "3.0.2")
    store = {}
    dlc_env.remember(store, dlc_env.Candidate(str(exe), "3.0.2", "picked"), picked=True)
    assert store == {"python_exe": str(exe), "python_exe_picked": True, "dlc_version": "3.0.2"}
    from dlc_batcher import settings

    settings.save(store)
    assert dlc_env.main(["--resolve"]) == 0
    assert capsys.readouterr().out.strip() == str(exe)
    assert dlc_env.main(["--save", str(tmp_path / "nope.exe")]) == 1
