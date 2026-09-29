"""Synthetic test footage generator.

WHY THIS EXISTS
---------------
Milestone 1 needs to be testable *today*, before anyone has shot real footage.
This writes three short clips that stand in for three cameras, with vehicles
carrying legible number plates -- large enough to clear the ~80px plate-width
rule -- so the detection, tracking and ANPR plumbing of Milestones 2-5 can be
built and debugged without waiting on a camera shoot.

Deliberately, four vehicles appear in all three clips at increasing times, so
there are genuine cross-camera journeys to test Milestone 7 against.

THIS IS SCAFFOLDING, NOT DATA.
Synthetic footage proves the pipeline is wired correctly. It proves nothing
about real-world accuracy, it never appears in the demo, and no number derived
from it goes near the presentation. Real recorded footage replaces it.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger

log = get_logger("sample")

WIDTH, HEIGHT = 1280, 720
FPS = 25

ROAD = (52, 54, 58)
SHOULDER = (44, 46, 50)
KERB = (78, 82, 88)
LANE_PAINT = (188, 190, 192)

# plate,        body colour (BGR),  kind,    w,   h
ROSTER = [
    ("KA01AB1234", (196, 132, 58), "car", 152, 66),
    ("KA05MJ8842", (72, 86, 196), "car", 148, 64),
    ("MH12DE1433", (118, 178, 96), "truck", 198, 88),
    ("KA03NC7719", (206, 188, 96), "car", 150, 64),
    ("TN09BX2255", (92, 92, 210), "bus", 226, 96),
    ("KA51HH0091", (152, 122, 200), "car", 146, 62),
    ("KL07AZ6612", (96, 176, 214), "car", 148, 64),
    ("AP28TQ3390", (176, 148, 108), "truck", 192, 84),
]

# (roster index, entry second, lane 0=far..2=near, direction, px/sec)
SCRIPTS: dict[str, list[tuple[int, float, int, int, float]]] = {
    "cam01": [
        (0, 1.0, 2, 1, 330), (1, 2.6, 1, 1, 380), (2, 4.4, 0, 1, 265),
        (3, 6.5, 2, 1, 350), (4, 9.0, 1, 1, 240), (5, 11.4, 2, 1, 395),
        (6, 13.8, 0, 1, 300), (7, 16.5, 1, 1, 275), (0, 19.2, 2, 1, 345),
    ],
    "cam02": [
        (1, 1.4, 1, -1, 355), (0, 3.0, 2, -1, 320), (5, 5.2, 0, -1, 290),
        (2, 7.0, 1, -1, 250), (3, 9.6, 2, -1, 365), (4, 12.2, 0, -1, 235),
        (6, 14.8, 1, -1, 310), (1, 17.6, 2, -1, 340),
    ],
    "cam03": [
        (2, 1.8, 0, 1, 260), (3, 3.6, 2, 1, 340), (0, 5.8, 1, 1, 315),
        (7, 8.2, 0, 1, 280), (1, 10.8, 2, 1, 370), (3, 13.6, 1, 1, 330),
        (2, 16.4, 0, 1, 255),
    ],
}

LANES = [(300, 0.74), (432, 0.90), (566, 1.06)]  # (centre y, scale)


def _road_background(t: float) -> np.ndarray:
    """Asphalt with scrolling lane paint, so motion reads even in still lanes."""
    img = np.full((HEIGHT, WIDTH, 3), ROAD, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (WIDTH, 205), SHOULDER, -1)
    cv2.rectangle(img, (0, 660), (WIDTH, HEIGHT), SHOULDER, -1)
    cv2.line(img, (0, 205), (WIDTH, 205), KERB, 3)
    cv2.line(img, (0, 660), (WIDTH, 660), KERB, 3)

    dash, gap, speed = 62, 46, 210.0
    shift = int((t * speed) % (dash + gap))
    for y in (366, 500):
        x = -(dash + gap) + shift
        while x < WIDTH:
            cv2.line(img, (x, y), (x + dash, y), LANE_PAINT, 3, cv2.LINE_AA)
            x += dash + gap
    return img


def _shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * factor))) for c in colour)


def _draw(img, spec, x: float, lane: int) -> None:
    plate, colour, kind, bw, bh = spec
    cy, scale = LANES[lane]
    w, h = int(bw * scale), int(bh * scale)
    x0, y0 = int(x), int(cy - h / 2)
    x1, y1 = x0 + w, y0 + h

    if x1 < -20 or x0 > WIDTH + 20:
        return

    # shadow, body, roof
    cv2.ellipse(img, (x0 + w // 2, y1 + 4), (int(w * 0.46), 7), 0, 0, 360, (34, 35, 38), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), _shade(colour, 0.55), 2)

    roof_h = int(h * (0.34 if kind == "car" else 0.55))
    cv2.rectangle(img, (x0 + int(w * 0.18), y0 + 3),
                  (x1 - int(w * 0.18), y0 + roof_h), _shade(colour, 0.72), -1)
    cv2.rectangle(img, (x0 + int(w * 0.24), y0 + 6),
                  (x1 - int(w * 0.24), y0 + roof_h - 3), (168, 178, 186), -1)

    # number plate on the leading edge, sized to stay OCR-legible
    pw, ph = int(112 * scale), int(32 * scale)
    px = x1 - pw - int(8 * scale) if lane % 2 == 0 else x0 + int(8 * scale)
    px = max(x0 + 4, min(px, x1 - pw - 4))
    py = y1 - ph - int(6 * scale)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (248, 248, 245), -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (40, 40, 40), 1)

    # shrink to fit rather than overflow the patch
    fs = 0.46 * scale
    while fs > 0.12:
        (tw, th), _ = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        if tw <= pw - 6:
            break
        fs -= 0.02
    cv2.putText(img, plate, (px + (pw - tw) // 2, py + (ph + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)


def generate_clip(path: Path, script, seconds: float = 25.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")

    total = int(seconds * FPS)
    for i in range(total):
        t = i / FPS
        img = _road_background(t)
        for idx, t_enter, lane, direction, speed in script:
            if t < t_enter:
                continue
            spec = ROSTER[idx]
            travelled = (t - t_enter) * speed
            body_w = int(spec[3] * LANES[lane][1])
            x = -body_w + travelled if direction > 0 else WIDTH - travelled
            _draw(img, spec, x, lane)
        writer.write(img)

    writer.release()
    return path


def generate_sample_set(video_dir: Path, seconds: float = 25.0) -> list[Path]:
    written = []
    for name, script in SCRIPTS.items():
        out = video_dir / f"{name}.mp4"
        log.info("generating %s (%.0fs, %d vehicles)", out.name, seconds, len(script))
        written.append(generate_clip(out, script, seconds))
    return written
