"""Speed and time-left readouts over a simulated run with gaps between videos."""
import pytest

from dlc_batcher.engine import RunConfig, RunManager, VideoItem

FPS = 600.0  # analysis speed of one worker
LOAD = 4.0  # seconds to open a video before frames arrive
WARM = 1.5  # first batches of a video are slow
SAVE = 6.0  # seconds to write results after the last frame


class FakeWorker:
    def __init__(self, index):
        self.index, self.current = index, None

    def alive(self):
        return True


def simulate(n_videos=8, frames=36000, n_workers=1, slowdown_at=None):
    """Advance a fake run one second at a time. Returns a list of
    (t, speed tile, eta, true seconds left, in_gap)."""
    mgr = RunManager()
    mgr.cfg = RunConfig(config="x")
    mgr.videos = [VideoItem(path=f"v{i}", n_frames=frames, fps=30.0) for i in range(n_videos)]
    mgr.by_path = {v.path: v for v in mgr.videos}
    mgr.workers = [FakeWorker(i) for i in range(n_workers)]
    mgr.running = True
    queue = list(mgr.videos)
    state = {w.index: None for w in mgr.workers}  # index -> [video, phase, t_phase, elapsed]
    rows, t, dt = [], 0.0, 1.0
    while True:
        fps = FPS / n_workers ** 0.5  # workers share one GPU
        if slowdown_at is not None and t >= slowdown_at:
            fps *= 0.5
        gap = False
        for w in mgr.workers:
            st = state[w.index]
            if st is None:
                if not queue:
                    continue
                v = queue.pop(0)
                v.status, w.current, v.worker = "loading", v.path, w.index
                st = state[w.index] = [v, "load", 0.0, 0.0]
            v = st[0]
            st[2] += dt
            if st[1] == "load":
                gap = True
                if st[2] >= LOAD:
                    st[1], st[2], v.status = "run", 0.0, "running"
            elif st[1] == "run":
                st[3] += dt
                rate = fps * (0.1 if st[3] <= WARM else 1.0)
                v.frames_done = min(v.n_frames, int(v.frames_done + rate * dt))
                v.speed = v.frames_done / st[3]
                if len(st) == 4:  # like the worker: timed from the first batch
                    st += [st[3], v.frames_done]
                elif st[3] > st[4]:
                    v.steady = (v.frames_done - st[5]) / (st[3] - st[4])
                if v.frames_done >= v.n_frames:
                    st[1], st[2], v.status = "save", 0.0, "saving"
            elif st[1] == "save":
                gap = True
                if st[2] >= SAVE:
                    v.status, w.current, state[w.index] = "done", None, None
        t += dt
        done = mgr._frames_done()
        mgr._update_rates(t, done)
        rows.append([t, mgr.steady_speed(), mgr.eta_seconds(), None, gap])
        if all(v.status == "done" for v in mgr.videos):
            break
        assert t < 1e5
    for row in rows:
        row[3] = t - row[0]
    return rows


def test_speed_does_not_dip_between_videos():
    rows = simulate()
    settled = [r for r in rows if r[0] > 30]
    speeds = [r[1] for r in settled]
    assert min(speeds) > 0.93 * FPS and max(speeds) < 1.02 * FPS
    in_gaps = [r[1] for r in settled if r[4]]
    assert in_gaps and min(in_gaps) > 0.93 * FPS
    # one second to the next the tile moves by a few fps at most
    steps = [abs(b - a) for a, b in zip(speeds, speeds[1:])]
    assert max(steps) < 0.03 * FPS


def test_time_left_is_steady_and_close():
    rows = simulate()
    settled = [r for r in rows if r[0] > 60 and r[2] is not None]
    # no upward jumps: it counts down, with small corrections
    ups = [b[2] - a[2] for a, b in zip(settled, settled[1:])]
    assert max(ups) < 3.0
    # the one-second change stays near -1 s, also across a gap
    assert min(ups) > -0.03 * rows[0][3] - 2.0
    # load and save times are only known once the first videos have finished
    for t, _speed, eta, truth, _gap in settled:
        if truth > 60:
            assert eta == pytest.approx(truth, rel=0.12 if t > 150 else 0.25), (t, eta, truth)


def test_two_workers():
    rows = simulate(n_workers=2)
    settled = [r for r in rows if r[0] > 40 and r[3] > 30]
    total = 2 * FPS / 2 ** 0.5
    assert min(r[1] for r in settled) > 0.9 * total
    assert max(r[1] for r in settled) < 1.03 * total
    for _t, _speed, eta, truth, _gap in settled:
        if truth > 60:
            assert eta == pytest.approx(truth, rel=0.15)


def test_time_left_follows_a_real_slowdown():
    rows = simulate(slowdown_at=150.0)
    late = [r for r in rows if r[0] > 330 and r[3] > 60]
    for _t, speed, eta, truth, _gap in late:
        assert speed < 0.62 * FPS
        assert eta == pytest.approx(truth, rel=0.25)
