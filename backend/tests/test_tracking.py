"""Logic tests for tracking and line counting.

These run without video, without a model and without a GPU, which is the point:
the bugs that actually bite in this part of the system are geometry and
bookkeeping bugs, and those are testable in milliseconds.

    make test          (or)      PYTHONPATH=backend python backend/tests/test_tracking.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import CountingLine          # noqa: E402
from app.vision.counting import LineCounter, segments_intersect  # noqa: E402
from app.vision.detector import Detection         # noqa: E402
from app.vision.tracker import TrackManager       # noqa: E402

T0 = datetime(2026, 9, 3, 10, 42, 0)

_passed, _failed = 0, []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def frame(index: int, seconds: float = 0.0):
    """Only .index and .timestamp are used by TrackManager."""
    return SimpleNamespace(index=index, timestamp=T0 + timedelta(seconds=seconds))


def det(x: float, y: float, tid: int, label: str = "car", conf: float = 0.9) -> Detection:
    return Detection(x1=x - 20, y1=y - 12, x2=x + 20, y2=y + 12,
                     confidence=conf, class_id=2, label=label, track_id=tid)


# ---------------------------------------------------------------------------
print("\ngeometry")

check("crossing segments intersect",
      segments_intersect((0, -10), (0, 10), (-10, 0), (10, 0)))
check("parallel segments do not",
      not segments_intersect((0, 5), (10, 5), (0, 0), (10, 0)))
check("segments that stop short do not",
      not segments_intersect((0, -10), (0, -1), (-10, 0), (10, 0)))
check("touching endpoint is not a crossing",
      not segments_intersect((0, -10), (0, 0), (-10, 0), (10, 0)))
check("beyond the line's extent does not count",
      not segments_intersect((50, -10), (50, 10), (-10, 0), (10, 0)))

# ---------------------------------------------------------------------------
print("\nnormalised coordinates")

line = CountingLine(start=(0.0, 0.5), end=(1.0, 0.5))
check("scales to 1280x720", line.pixels(1280, 720) == ((0, 360), (1280, 360)),
      str(line.pixels(1280, 720)))
check("same line at 640x480", line.pixels(640, 480) == ((0, 240), (640, 240)),
      str(line.pixels(640, 480)))

# ---------------------------------------------------------------------------
print("\nline counting")

def run_counter(path, tid=1, label="car"):
    """Feed a sequence of centre points through manager + counter."""
    mgr = TrackManager("TEST", min_frames=1, max_age_frames=5)
    cnt = LineCounter(CountingLine(start=(0.0, 0.5), end=(1.0, 0.5),
                                   positive_label="down", negative_label="up"),
                      100, 100, "TEST")
    for i, (x, y) in enumerate(path):
        mgr.update([det(x, y, tid, label)], frame(i, i * 0.1))
        cnt.update(mgr.stable_active)
    return mgr, cnt

# line sits at y=50. moving downward (increasing y) ends on the +side.
_, c = run_counter([(50, 20), (50, 40), (50, 60), (50, 80)])
check("counts a crossing once", c.total == 1, f"total={c.total}")
check("direction resolved", sum(c.counts.values()) == 1 and max(c.counts.values()) == 1)
down_label = [k for k, v in c.counts.items() if v == 1][0]

_, c = run_counter([(50, 80), (50, 60), (50, 40), (50, 20)])
up_label = [k for k, v in c.counts.items() if v == 1][0]
check("opposite travel gives the opposite direction", up_label != down_label,
      f"{up_label} vs {down_label}")

_, c = run_counter([(50, 20), (50, 60), (50, 62), (50, 64), (50, 66)])
check("never double-counts after crossing", c.total == 1, f"total={c.total}")

_, c = run_counter([(50, 20), (50, 30), (50, 40), (50, 45)])
check("no count when the line is never reached", c.total == 0, f"total={c.total}")

_, c = run_counter([(50, 20), (50, 60), (50, 20), (50, 60)])
check("a vehicle is counted at most once, even on a U-turn", c.total == 1,
      f"total={c.total}")

_, c = run_counter([(50, 20), (50, 80)], label="bus")
check("crossing is attributed to the voted class", c.by_class.get("bus") == 1,
      str(c.by_class))

# ---------------------------------------------------------------------------
print("\nclass voting")

mgr = TrackManager("TEST", min_frames=1, max_age_frames=5)
# 3 low-confidence 'truck' frames vs 2 high-confidence 'car' frames.
seq = [("truck", 0.40), ("truck", 0.42), ("truck", 0.38), ("car", 0.95), ("car", 0.93)]
for i, (label, conf) in enumerate(seq):
    mgr.update([det(10 + i, 10, 7, label, conf)], frame(i, i * 0.1))
track = mgr.active[7]
check("confidence-weighted vote beats a raw frame count",
      track.label == "car", f"got {track.label} votes={track.class_votes}")
check("label confidence is a proportion",
      0.0 < track.label_confidence < 1.0, str(track.label_confidence))

mgr = TrackManager("TEST", min_frames=1, max_age_frames=5)
for i in range(6):
    mgr.update([det(10 + i, 10, 8, "car", 0.9)], frame(i, i * 0.1))
check("a single ID stays one track across frames",
      mgr.total_created == 1 and mgr.active[8].frames == 6,
      f"created={mgr.total_created}")

# ---------------------------------------------------------------------------
print("\ntrack lifecycle")

mgr = TrackManager("TEST", min_frames=3, max_age_frames=2)
for i in range(5):
    mgr.update([det(10 + i, 10, 1, "car", 0.9)], frame(i, i * 0.1))
for i in range(5, 10):                      # vehicle gone
    mgr.update([], frame(i, i * 0.1))
done = mgr.drain_completed()
check("track closes after max_age_frames", len(done) == 1, f"{len(done)} closed")
check("closed track kept its history",
      done and done[0].frames == 5 and done[0].duration_s > 0)

mgr = TrackManager("TEST", min_frames=5, max_age_frames=2)
for i in range(2):                          # only 2 frames -- a flicker
    mgr.update([det(10, 10, 2, "car", 0.9)], frame(i, i * 0.1))
for i in range(2, 8):
    mgr.update([], frame(i, i * 0.1))
check("flickers below min_frames are discarded", len(mgr.drain_completed()) == 0)

mgr = TrackManager("TEST", min_frames=1, max_age_frames=50)
for i in range(4):
    mgr.update([det(10, 10, 3, "car", 0.9)], frame(i, i * 0.1))
check("flush closes tracks still active at end of stream",
      len(mgr.flush()) == 1 and not mgr.active)

mgr = TrackManager("TEST", min_frames=1, max_age_frames=5)
mgr.update([det(10, 10, None, "car", 0.9)], frame(0))
check("detections without a track id are ignored", mgr.total_created == 0)

mgr = TrackManager("TEST", min_frames=1, max_age_frames=5)
for i in range(3):
    mgr.update([det(10, 10, 1), det(50, 50, 2), det(90, 20, 3)], frame(i, i * 0.1))
check("multiple vehicles are tracked independently",
      mgr.total_created == 3 and len(mgr.active) == 3)

s = mgr.active[1].to_sighting()
check("sighting record carries the M4 contract",
      {"camera_id", "track_id", "vehicle_class", "first_seen", "last_seen",
       "duration_s", "frames", "crossing"} <= set(s), str(sorted(s)))

# ---------------------------------------------------------------------------
print()
if _failed:
    print(f"{_passed} passed, {len(_failed)} FAILED: {', '.join(_failed)}")
    raise SystemExit(1)
print(f"{_passed} passed, 0 failed")
