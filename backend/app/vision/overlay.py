"""Drawing helpers for the on-screen HUD.

Kept separate from the pipeline because later milestones draw into the same
frame -- boxes in M2, track IDs in M3, plate crops in M5 -- and they should all
share one visual language rather than each inventing its own.
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR. A cool slate bar with an amber signal accent -- matches the project's
# command-centre palette.
BAR_BG = (28, 22, 18)
ACCENT = (60, 170, 235)
WHITE = (245, 240, 235)
MUTED = (170, 155, 140)

BAR_HEIGHT = 62


def draw_hud(
    image: np.ndarray,
    *,
    title: str,
    subtitle: str = "",
    stats: tuple[tuple[str, str], ...] = (),
    hint: str = "",
) -> np.ndarray:
    """Draw a translucent status bar across the top of a frame, in place."""
    h, w = image.shape[:2]

    band = image[0:BAR_HEIGHT, 0:w]
    cv2.addWeighted(np.full_like(band, BAR_BG, dtype=np.uint8), 0.78, band, 0.22, 0, band)
    cv2.line(image, (0, BAR_HEIGHT), (w, BAR_HEIGHT), ACCENT, 2, cv2.LINE_AA)

    cv2.putText(image, title, (16, 27), FONT, 0.62, WHITE, 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(image, subtitle, (16, 48), FONT, 0.42, MUTED, 1, cv2.LINE_AA)

    # Stats sit right-aligned, laid out from the right edge backwards.
    x = w - 18
    for label, value in reversed(stats):
        (vw, _), _ = cv2.getTextSize(value, FONT, 0.56, 1)
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.36, 1)
        block = max(vw, lw)
        x -= block
        cv2.putText(image, label, (x, 24), FONT, 0.36, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, value, (x, 48), FONT, 0.56, WHITE, 1, cv2.LINE_AA)
        x -= 28

    if hint:
        cv2.putText(image, hint, (16, h - 14), FONT, 0.40, MUTED, 1, cv2.LINE_AA)

    return image


# --- detection drawing -----------------------------------------------------
# One colour per vehicle class, kept here so every milestone that draws boxes
# (M2 detection, M3 tracks, M5 plate crops) speaks the same visual language.
# Values are BGR, chosen to stay legible against dark asphalt.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "car": (60, 175, 240),          # amber
    "motorcycle": (140, 205, 90),   # green
    "bus": (190, 110, 230),         # magenta
    "truck": (225, 200, 80),        # cyan
    "bicycle": (240, 150, 170),     # violet
}
DEFAULT_COLOUR = (200, 200, 200)


def class_colour(label: str) -> tuple[int, int, int]:
    return CLASS_COLOURS.get(label, DEFAULT_COLOUR)


def draw_detections(image: np.ndarray, detections, show_confidence: bool = True) -> np.ndarray:
    """Draw detection boxes with a label chip, in place."""
    h = image.shape[0]
    for det in detections:
        colour = class_colour(det.label)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        text = f"{det.label} {det.confidence:.2f}" if show_confidence else det.label
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.44, 1)
        chip_h = th + 10

        # Sit the chip above the box, or inside the top edge when there is no room.
        cy = y1 - chip_h
        inside = cy < BAR_HEIGHT + 2
        if inside:
            cy = min(y1, h - chip_h - 2)

        cv2.rectangle(image, (x1, cy), (x1 + tw + 12, cy + chip_h), colour, -1)
        cv2.putText(image, text, (x1 + 6, cy + chip_h - 7), FONT, 0.44,
                    (25, 25, 28), 1, cv2.LINE_AA)
    return image


def summarise(detections) -> str:
    """'car 3  truck 1' -- ASCII only; Hershey fonts have nothing else."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    if not counts:
        return "none"
    return "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))


# --- tracking drawing ------------------------------------------------------

def draw_tracks(image: np.ndarray, detections, tracks: dict) -> np.ndarray:
    """Draw tracked boxes with a stable ID, the voted class, and a motion trail.

    The ID is what proves tracking works to anyone watching the screen, so it
    leads the label.
    """
    h = image.shape[0]
    for det in detections:
        tid = getattr(det, "track_id", None)
        if tid is None:
            continue
        track = tracks.get(int(tid))
        colour = class_colour(track.label if track else det.label)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        if track is not None and len(track.centers) > 1:
            pts = np.array(track.centers, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], False, colour, 2, cv2.LINE_AA)

        label = track.label if track else det.label
        text = f"#{int(tid)} {label}"
        if track is not None:
            plate = getattr(track, "plate", None)
            if plate is not None and getattr(plate, "ok", False):
                text = f"#{int(tid)} {plate.text}"
            elif track.plate_readings:
                text = f"#{int(tid)} {label} [{len(track.plate_readings)} ocr]"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.46, 1)
        chip_h = th + 10
        cy = y1 - chip_h
        if cy < BAR_HEIGHT + 2:
            cy = min(y1, h - chip_h - 2)
        cv2.rectangle(image, (x1, cy), (x1 + tw + 12, cy + chip_h), colour, -1)
        cv2.putText(image, text, (x1 + 6, cy + chip_h - 7), FONT, 0.46,
                    (25, 25, 28), 1, cv2.LINE_AA)
    return image


def draw_counting_line(image: np.ndarray, counter) -> np.ndarray:
    """Draw the counting line with its per-direction tallies."""
    a, b = counter.a, counter.b
    cv2.line(image, a, b, (35, 38, 42), 6, cv2.LINE_AA)      # casing, for contrast
    cv2.line(image, a, b, ACCENT, 2, cv2.LINE_AA)
    for pt in (a, b):
        cv2.circle(image, pt, 5, ACCENT, -1, cv2.LINE_AA)

    y = max(a[1], b[1]) + 24
    x = a[0]
    for name, value in counter.counts.items():
        text = f"{name} {value}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.50, 1)
        cv2.rectangle(image, (x, y - th - 8), (x + tw + 14, y + 8), (28, 22, 18), -1)
        cv2.putText(image, text, (x + 7, y), FONT, 0.50, WHITE, 1, cv2.LINE_AA)
        x += tw + 24
    return image
