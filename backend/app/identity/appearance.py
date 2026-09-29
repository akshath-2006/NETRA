"""Cheap appearance features.

The second of the three signals. Deliberately not a learned re-ID embedding:
vehicle class, dominant colour and shape get most of the discriminating power
for a fraction of the effort, and they degrade honestly (a grey car at night
looks like every other grey car, and the score says so).

Upgrading to a real embedding later means replacing ``similarity`` and nothing
else -- the resolver only asks for a number between 0 and 1.
"""

from __future__ import annotations

import math

# Classes a detector genuinely confuses at a distance. A car read as a truck is
# a plausible same-vehicle pair; a car read as a motorcycle is not.
CONFUSABLE_CLASSES = {
    frozenset({"car", "truck"}),
    frozenset({"truck", "bus"}),
    frozenset({"car", "bus"}),
    frozenset({"bicycle", "motorcycle"}),
}


def parse_hex(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    v = value.strip().lstrip("#")
    if len(v) != 6:
        return None
    try:
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    except ValueError:
        return None


def to_hex(bgr) -> str:
    """OpenCV gives BGR; store RGB so the dashboard can use it directly."""
    b, g, r = (int(max(0, min(255, c))) for c in bgr)
    return f"{r:02X}{g:02X}{b:02X}"


def class_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.5                      # unknown, so neither evidence nor veto
    if a == b:
        return 1.0
    return 0.45 if frozenset({a, b}) in CONFUSABLE_CLASSES else 0.0


def colour_similarity(a: str | None, b: str | None) -> float:
    """1 - normalised euclidean distance in RGB.

    RGB rather than a perceptual space on purpose: the input is a crude mean
    over a vehicle crop under uncontrolled lighting, so a more principled
    colour space would be false precision.
    """
    ca, cb = parse_hex(a), parse_hex(b)
    if ca is None or cb is None:
        return 0.5
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb)))
    return max(0.0, 1.0 - dist / 441.67)          # 441.67 = sqrt(3*255^2)


def shape_similarity(a: float | None, b: float | None) -> float:
    """Compare box aspect ratios. A bus stays long; a hatchback stays square."""
    if not a or not b or a <= 0 or b <= 0:
        return 0.5
    ratio = min(a, b) / max(a, b)
    return float(ratio)


def similarity(a, b) -> float:
    """Overall appearance agreement between two sightings, 0-1.

    Class carries the most weight because it is the most reliable of the three;
    colour is informative but lighting-sensitive; shape is weakest because it
    changes with viewing angle.
    """
    cls = class_similarity(getattr(a, "vehicle_class", None),
                           getattr(b, "vehicle_class", None))
    col = colour_similarity(getattr(a, "colour_hex", None),
                            getattr(b, "colour_hex", None))
    shp = shape_similarity(getattr(a, "aspect", None), getattr(b, "aspect", None))

    # A hard class mismatch is close to disqualifying on its own.
    if cls == 0.0:
        return 0.10 * col + 0.05 * shp

    return 0.55 * cls + 0.30 * col + 0.15 * shp
