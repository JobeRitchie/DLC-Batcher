"""Check the git remote for a newer version, fast-forward to it and restart.

The app is distributed as a git checkout, so "updating" is fetching
``origin/main`` and fast-forwarding. Local edits are never overwritten: if the
branch cannot be fast-forwarded the update stops and says why.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REMOTE = "origin"
BRANCH = "main"
GIT_MISSING = ("Git was not found. Install Git for Windows from "
               "https://git-scm.com/download/win and restart the app.")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class UpdateStatus:
    behind: int = 0  # commits the local branch is behind the remote
    current: str = ""  # local version
    latest: str = ""  # version on the remote, when it could be read
    error: str = ""

    @property
    def available(self) -> bool:
        return not self.error and self.behind > 0

    @property
    def summary(self) -> str:
        if self.latest and self.latest != self.current:
            return f"Version {self.latest} is available (you have {self.current})."
        n = self.behind
        return f"An update is available ({n} change{'s' if n != 1 else ''} since your copy)."


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_git() -> str | None:
    """Locate git even when it is not on this process's PATH. An app started
    from a shortcut keeps the PATH from before Git was installed."""
    found = shutil.which("git")
    if found or os.name != "nt":
        return found
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if not base:
            continue
        for root in (Path(base) / "Git", Path(base) / "Programs" / "Git"):
            for exe in (root / "cmd" / "git.exe", root / "bin" / "git.exe"):
                if exe.exists():
                    return str(exe)
    return None


class Updater:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else repo_root()
        self.git = find_git()

    def _run(self, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run([self.git, *args], cwd=str(self.root), capture_output=True,
                              text=True, timeout=timeout, creationflags=_NO_WINDOW)

    def _trust_folder(self) -> None:
        """Git refuses folders owned by another account (network shares) until
        they are listed as safe."""
        r = self._run("rev-parse", "--is-inside-work-tree", timeout=10)
        if r.returncode == 0 or "dubious ownership" not in (r.stderr or "").lower():
            return
        paths = {str(self.root), str(self.root).replace("\\", "/")}
        m = re.search(r"repository at '([^']+)'", r.stderr or "")
        if m:
            paths.add(m.group(1))
        for p in paths:
            self._run("config", "--global", "--add", "safe.directory", p, timeout=10)

    def _version_at(self, ref: str) -> str:
        r = self._run("show", f"{ref}:dlc_batcher/__init__.py", timeout=10)
        m = re.search(r'__version__\s*=\s*"([^"]+)"', r.stdout or "")
        return m.group(1) if r.returncode == 0 and m else ""

    def check(self) -> UpdateStatus:
        from . import __version__

        status = UpdateStatus(current=__version__)
        if not (self.root / ".git").exists():
            status.error = "This copy is not a git checkout, so it cannot update itself."
            return status
        if not self.git:
            status.error = GIT_MISSING
            return status
        try:
            self._trust_folder()
            if REMOTE not in self._run("remote", timeout=10).stdout.split():
                status.error = "This copy has no git remote to update from."
                return status
            fetch = self._run("fetch", REMOTE, BRANCH)
            if fetch.returncode != 0:
                status.error = (fetch.stderr or fetch.stdout or "git fetch failed").strip()
                return status
            count = self._run("rev-list", "--count", f"HEAD..{REMOTE}/{BRANCH}", timeout=15)
            if count.returncode != 0:
                status.error = (count.stderr or "git rev-list failed").strip()
                return status
            status.behind = int(count.stdout.strip() or 0)
            if status.behind:
                status.latest = self._version_at(f"{REMOTE}/{BRANCH}")
        except subprocess.TimeoutExpired:
            status.error = "Timed out. Check the network connection."
        except (OSError, ValueError) as exc:
            status.error = str(exc)
        return status

    def apply(self) -> tuple[bool, str]:
        """Fast-forward to the remote branch. Returns (ok, git's output)."""
        if not self.git:
            return False, GIT_MISSING
        try:
            fetch = self._run("fetch", REMOTE, BRANCH)
            if fetch.returncode != 0:
                return False, (fetch.stderr or fetch.stdout).strip()
            merge = self._run("merge", "--ff-only", f"{REMOTE}/{BRANCH}", timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        out = "\n".join(p.strip() for p in (merge.stdout, merge.stderr) if p and p.strip())
        if merge.returncode != 0:
            out += ("\n\nThe update was not applied. This usually means files in the app "
                    "folder were edited. Run 'git status' there to see which.")
        return merge.returncode == 0, out

    def relaunch(self) -> None:
        """Start a new copy of the app, through the launcher when there is one."""
        bat = self.root / "DLC Batcher.bat"
        if os.name == "nt" and bat.exists():
            subprocess.Popen(["cmd", "/c", str(bat)], cwd=str(self.root), close_fds=True,
                             creationflags=0x00000010)  # CREATE_NEW_CONSOLE
            return
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0  # detached, new group
        subprocess.Popen([sys.executable, "-m", "dlc_batcher"], cwd=str(self.root),
                         close_fds=True, creationflags=flags, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
