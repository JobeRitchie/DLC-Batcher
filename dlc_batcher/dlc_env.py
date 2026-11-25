"""Find the Python environment that has DeepLabCut installed.

Nothing here imports deeplabcut (that takes many seconds and loads torch).
An environment counts when its site-packages holds a ``deeplabcut-<version>.dist-info``
folder, and the version is read from that folder name.

Also usable from the launcher:
    python -m dlc_batcher.dlc_env --resolve       print the interpreter to use
    python -m dlc_batcher.dlc_env --save PATH     check PATH and remember it
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import settings

ENV_VAR = "DLC_BATCHER_PYTHON"
TESTED_VERSIONS = ("3.0.0rc8", "3.0.2")

_DIST_INFO = re.compile(r"^deeplabcut-(?P<version>[^-]+)\.dist-info$", re.IGNORECASE)
_WINDOWS = os.name == "nt"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class Candidate:
    python: str
    version: str
    source: str  # "app", "env var", "conda", "venv", "saved", "picked"

    @property
    def tested(self) -> bool:
        return self.version in TESTED_VERSIONS

    @property
    def label(self) -> str:
        return f"DeepLabCut {self.version} at {self.python}"


def version_key(version: str) -> tuple:
    """Sort key for versions such as 3.0.0rc8 < 3.0.0 < 3.0.2."""
    m = re.match(r"^(\d+(?:\.\d+)*)(?:\.?(a|b|rc|dev)\.?(\d*))?", version.strip().lower())
    if not m:
        return ((0,), -1, 0)
    release = tuple(int(p) for p in m.group(1).split("."))
    release += (0,) * (4 - len(release))
    stage = {"dev": 0, "a": 1, "b": 2, "rc": 3, None: 4}[m.group(2)]
    return (release, stage, int(m.group(3) or 0))


def env_root(python_exe: str | os.PathLike) -> Path:
    """The environment folder for an interpreter path.

    conda on Windows keeps python.exe in the environment folder itself; venvs
    and everything on Linux/macOS keep it in Scripts/ or bin/.
    """
    parent = Path(python_exe).parent
    if parent.name.lower() in ("scripts", "bin"):
        return parent.parent
    return parent


def site_packages(python_exe: str | os.PathLike) -> list[Path]:
    root = env_root(python_exe)
    found = [root / "Lib" / "site-packages"]
    found += sorted((root / "lib").glob("python*/site-packages"))
    return [p for p in found if p.is_dir()]


def dlc_version(python_exe: str | os.PathLike) -> str | None:
    """DeepLabCut version installed for this interpreter, or None."""
    versions = []
    for sp in site_packages(python_exe):
        try:
            names = os.listdir(sp)
        except OSError:
            continue
        for name in names:
            m = _DIST_INFO.match(name)
            if m:
                versions.append(m.group("version"))
    return max(versions, key=version_key) if versions else None


def python_in(env: str | os.PathLike) -> Path | None:
    """The interpreter inside an environment folder, if there is one."""
    env = Path(env)
    names = ("python.exe", "Scripts/python.exe") if _WINDOWS else ("bin/python", "bin/python3")
    for name in names:
        exe = env / name
        if exe.is_file():
            return exe
    return None


def console_python(exe: str) -> str:
    """pythonw.exe has no console streams, and the workers need stdio."""
    if os.path.basename(exe).lower() == "pythonw.exe":
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def conda_roots(home: Path | None = None) -> list[Path]:
    home = home or Path.home()
    roots = [home / n for n in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda")]
    program_data = os.environ.get("ProgramData")
    if program_data:
        roots += [Path(program_data) / n for n in ("anaconda3", "miniconda3", "miniforge3")]
    for var in ("CONDA_PREFIX", "CONDA_ROOT"):
        value = os.environ.get(var)
        if value:
            p = Path(value)
            # an active env lives in <root>/envs/<name>
            roots.append(p.parent.parent if p.parent.name == "envs" else p)
    return roots


def venv_spots(home: Path | None = None, app_dir: Path | None = None) -> list[Path]:
    home = home or Path.home()
    app_dir = app_dir or Path(__file__).resolve().parent.parent
    spots = [
        home / "Desktop" / "DeepLabCut" / "venv",
        home / "Desktop" / "DeepLabCut" / ".venv",
        home / "DeepLabCut" / "venv",
        home / "DeepLabCut" / ".venv",
        app_dir / ".venv",
        app_dir / "venv",
    ]
    return spots


def find_candidates(home: Path | None = None, app_dir: Path | None = None,
                    running: str | None = None, environ: dict | None = None) -> list[Candidate]:
    """Every interpreter with DeepLabCut that can be found, newest version first."""
    environ = os.environ if environ is None else environ
    seen: dict[str, Candidate] = {}

    def consider(exe, source: str) -> None:
        if not exe:
            return
        exe = str(exe)
        key = os.path.normcase(os.path.abspath(exe))
        if key in seen or not os.path.isfile(exe):
            return
        version = dlc_version(exe)
        if version:
            seen[key] = Candidate(os.path.abspath(exe), version, source)

    consider(console_python(running if running is not None else sys.executable), "app")
    consider(environ.get(ENV_VAR, "").strip().strip('"'), "env var")
    for root in conda_roots(home):
        consider(python_in(root), "conda")
        envs = root / "envs"
        if envs.is_dir():
            for env in sorted(envs.iterdir()):
                consider(python_in(env), "conda")
    for spot in venv_spots(home, app_dir):
        consider(python_in(spot), "venv")
    return sorted(seen.values(), key=lambda c: version_key(c.version), reverse=True)


def validate(python_exe: str, run: bool = True) -> tuple[Candidate | None, str]:
    """Check a path the user picked. Returns (candidate, "") or (None, reason)."""
    exe = str(python_exe or "").strip().strip('"')
    if not exe:
        return None, "No path given."
    path = Path(exe)
    if path.is_dir():
        inside = python_in(path)
        if inside is None:
            return None, f"There is no Python interpreter in {path}."
        path = inside
    if not path.is_file():
        return None, f"{path} does not exist."
    if not path.name.lower().startswith("python"):
        return None, f"{path.name} is not a Python interpreter."
    path = Path(console_python(str(path)))
    version = dlc_version(path)
    if version is None:
        return None, f"DeepLabCut is not installed in {env_root(path)}."
    if run:
        try:
            out = subprocess.run([str(path), "-c", "print('ok')"], capture_output=True,
                                 text=True, timeout=30, creationflags=_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"{path} could not be started: {exc}"
        if out.returncode != 0 or "ok" not in out.stdout:
            return None, f"{path} did not run (exit code {out.returncode})."
    return Candidate(str(path.absolute()), version, "picked"), ""


def resolve(store: dict, candidates: list[Candidate] | None = None,
            environ: dict | None = None) -> tuple[Candidate | None, list[Candidate]]:
    """Pick the interpreter for the workers.

    Order: DLC_BATCHER_PYTHON, the choice saved in settings, then the newest
    DeepLabCut found. Returns (choice, all candidates).
    """
    environ = os.environ if environ is None else environ
    if candidates is None:
        candidates = find_candidates(environ=environ)
    forced = environ.get(ENV_VAR, "").strip().strip('"')
    if forced:
        cand, _ = validate(forced, run=False)
        if cand:
            return Candidate(cand.python, cand.version, "env var"), candidates
    saved = str(store.get("python_exe") or "")
    if saved and store.get("python_exe_picked", True):
        cand, _ = validate(saved, run=False)
        if cand:
            return Candidate(cand.python, cand.version, "saved"), candidates
    return (candidates[0] if candidates else None), candidates


def remember(store: dict, cand: Candidate, picked: bool) -> None:
    """Store the interpreter. ``picked`` marks a choice the user made, which is
    kept until they change it; an automatic choice is looked up again on every
    start so a newer DeepLabCut is noticed."""
    store["python_exe"] = cand.python
    store["python_exe_picked"] = bool(picked)
    store["dlc_version"] = cand.version


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--resolve"]:
        choice, _ = resolve(settings.load())
        if choice is None:
            return 1
        print(choice.python)
        return 0
    if args[:1] == ["--save"] and len(args) == 2:
        cand, why = validate(args[1])
        if cand is None:
            print(why)
            return 1
        store = settings.load()
        remember(store, cand, picked=True)
        settings.save(store)
        print(cand.python)
        return 0
    if args[:1] == ["--list"]:
        for c in find_candidates():
            print(f"{c.version}\t{c.source}\t{c.python}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
