"""Virtual line counting.

A vehicle is counted when the line joining its previous position to its current
position crosses the camera's counting line. Two properties matter and both are
easy to get wrong:

* **Count once, never twice.** A vehicle idling on the line would otherwise
  tick the counter every frame. We latch a flag on the track instead.
* **Direction is signed.** Which side the vehicle ended up on tells us
  inbound vs outbound, which is what makes an origin-destination matrix
  possible in Milestone 9.

The geometry is a standard segment-intersection test. Collinear cases are
treated as "no crossing", which is the right answer for a vehicle sliding
exactly along the line.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import CountingLine
from app.core.logging import get_logger

Point = tuple[float, float]


def _side(p: Point, a: Point, b: Point) -> float:
    """>0, <0 or 0 depending on which side of line a->b the point p is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    """True when segment p1->p2 properly crosses segment a->b."""
    d1, d2 = _side(p1, a, b), _side(p2, a, b)
    d3, d4 = _side(a, p1, p2), _side(b, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass
class Crossing:
    track_id: int
    label: str
    direction: str
    at: object          # datetime
    point: tuple[int, int]


class LineCounter:
    """Counts tracks crossing one camera's counting line."""

    def __init__(self, line: CountingLine, width: int, height: int, camera_id: str = "") -> None:
        self.line = line
        self.a, self.b = line.pixels(width, height)
        self.counts: dict[str, int] = {line.positive_label: 0, line.negative_label: 0}
        self.by_class: dict[str, int] = {}
        self.crossings: list[Crossing] = []
        self.log = get_logger(f"count.{camera_id}" if camera_id else "count")

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def update(self, tracks) -> list[Crossing]:
        """Check every active track for a crossing this frame."""
        new: list[Crossing] = []

        for track in tracks:
            if track.counted or len(track.centers) < 2:
                continue
            prev, cur = track.centers[-2], track.centers[-1]
            if not segments_intersect(prev, cur, self.a, self.b):
                continue

            # Which side did it end up on? That is the direction.
            direction = (self.line.positive_label if _side(cur, self.a, self.b) > 0
                         else self.line.negative_label)

            track.counted = True
            track.crossing = direction
            self.counts[direction] = self.counts.get(direction, 0) + 1
            self.by_class[track.label] = self.by_class.get(track.label, 0) + 1

            crossing = Crossing(track_id=track.track_id, label=track.label,
                                direction=direction, at=track.last_seen, point=cur)
            self.crossings.append(crossing)
            new.append(crossing)
            self.log.info("track %d (%s) crossed %s", track.track_id, track.label, direction)

        return new

    def summary(self) -> str:
        parts = [f"{k} {v}" for k, v in self.counts.items()]
        return "  ".join(parts)
