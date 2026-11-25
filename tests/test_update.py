"""The updater against a real local git remote."""
import subprocess

import pytest

from dlc_batcher import update

pytestmark = pytest.mark.skipif(update.find_git() is None, reason="git not installed")
INIT = 'APP_NAME = "DLC Batcher"\n__version__ = "{}"\n'


def git(cwd, *args):
    cmd = [update.find_git(), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def publish(upstream, version):
    (upstream / "dlc_batcher").mkdir(exist_ok=True)
    (upstream / "dlc_batcher" / "__init__.py").write_text(INIT.format(version))
    git(upstream, "add", "-A")
    git(upstream, "commit", "-m", f"version {version}")


@pytest.fixture
def repos(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    publish(upstream, "0.1.0")
    git(tmp_path, "clone", str(upstream), "local")
    return upstream, tmp_path / "local"


def test_up_to_date(repos):
    _upstream, local = repos
    status = update.Updater(local).check()
    assert status.error == "" and not status.available


def test_update_found_and_applied(repos):
    upstream, local = repos
    publish(upstream, "0.2.0")
    up = update.Updater(local)
    status = up.check()
    assert status.available and status.behind == 1 and status.latest == "0.2.0"
    assert "0.2.0" in status.summary
    ok, _out = up.apply()
    assert ok
    assert '"0.2.0"' in (local / "dlc_batcher" / "__init__.py").read_text()
    assert not up.check().available


def test_local_edits_are_not_overwritten(repos):
    upstream, local = repos
    publish(upstream, "0.2.0")
    mine = INIT.format("0.1.0") + "# my edit\n"
    (local / "dlc_batcher" / "__init__.py").write_text(mine)
    ok, out = update.Updater(local).apply()
    assert not ok and "not applied" in out
    assert (local / "dlc_batcher" / "__init__.py").read_text() == mine


def test_no_checkout_or_remote(tmp_path):
    assert "not a git checkout" in update.Updater(tmp_path).check().error
    git(tmp_path, "init", "-b", "main")
    assert "no git remote" in update.Updater(tmp_path).check().error
