"""Generate the bundled calibration clip (synthetic, no real animal data).

A gray arena with a dark blob moving around: enough structure for a pose
network to do real work, small enough to ship in the repository. The tuner
loops it, so its length does not matter.

    python make_clip.py [width height fps frames]
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def make(path: Path, width=640, height=480, fps=30, frames=240, seed=7) -> None:
    rng = np.random.default_rng(seed)
    floor = rng.normal(150, 6, (height, width)).astype(np.float32)
    floor = cv2.GaussianBlur(floor, (0, 0), 3)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        t = 2 * np.pi * i / frames  # one seamless loop
        cx = width * (0.5 + 0.32 * np.sin(t))
        cy = height * (0.5 + 0.30 * np.sin(2 * t + 0.6))
        ang = np.degrees(np.arctan2(0.60 * np.cos(2 * t + 0.6), 0.32 * np.cos(t)))
        img = np.repeat((floor + rng.normal(0, 3, floor.shape))[..., None], 3, axis=2)
        img = np.clip(img, 0, 255).astype(np.uint8)
        s = height / 480
        cv2.ellipse(img, (int(cx), int(cy)), (int(46 * s), int(24 * s)), ang, 0, 360, (45, 45, 50), -1)
        hx = cx + 44 * s * np.cos(np.radians(ang))
        hy = cy + 44 * s * np.sin(np.radians(ang))
        cv2.circle(img, (int(hx), int(hy)), int(15 * s), (60, 60, 66), -1)
        for side in (-1, 1):
            ex = hx + 12 * s * np.cos(np.radians(ang + side * 70))
            ey = hy + 12 * s * np.sin(np.radians(ang + side * 70))
            cv2.circle(img, (int(ex), int(ey)), int(6 * s), (95, 80, 90), -1)
        tx = cx - 80 * s * np.cos(np.radians(ang))
        ty = cy - 80 * s * np.sin(np.radians(ang))
        cv2.line(img, (int(cx - 40 * s * np.cos(np.radians(ang))), int(cy - 40 * s * np.sin(np.radians(ang)))),
                 (int(tx), int(ty)), (70, 60, 65), max(1, int(4 * s)))
        writer.write(cv2.GaussianBlur(img, (0, 0), 0.8))
    writer.release()


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:5]]
    make(Path(__file__).with_name("calibration_clip.mp4"), *args)
    print("wrote calibration_clip.mp4")
