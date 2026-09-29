"""Tests for video preflight.

    PYTHONPATH=backend python backend/tests/test_preflight.py

Every fixture is generated here with OpenCV, so the suite needs no footage,
no network and no models. The point of these tests is that a WRONG verdict is
worse than no verdict: calling a good file corrupt sends someone re-downloading
footage that was fine, and calling a placeholder "ready" is exactly the failure
that cost an hour of processing and produced nothing.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                   # noqa: E402

from app.vision.preflight import (classify, preflight_sources,       # noqa: E402
                                  probe_file, summarise)

_passed, _failed = 0, []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))


def write_clip(path: Path, frames: int, w: int = 352, h: int = 288,
               fps: float = 25.0) -> Path:
    """A real, decodable H.264/MP4V clip with moving content."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    rng = np.random.default_rng(7)
    for i in range(frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i % w, 40), ((i % w) + 60, 120), (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return path


tmp = Path(tempfile.mkdtemp())

print("\nfixtures")
good = write_clip(tmp / "good.mp4", frames=250)
short = write_clip(tmp / "short.mp4", frames=25, w=64, h=48)     # tiny AND brief
truncated = tmp / "truncated.mp4"
shutil.copy(good, truncated)
with open(truncated, "r+b") as f:
    f.truncate(int(os.path.getsize(truncated) * 0.3))
notvideo = tmp / "notvideo.mp4"
notvideo.write_text("plain text wearing an .mp4 extension\n", encoding="utf-8")
print(f"  good {os.path.getsize(good)}B  short {os.path.getsize(short)}B  "
      f"truncated {os.path.getsize(truncated)}B")

# ---------------------------------------------------------------------------
print("\nthe five verdicts")
p = classify(probe_file(good, camera_id="CAM01"))
check("a clean clip is ready", p.verdict == "ready", f"{p.verdict}: {p.reason}")
check("it reports the real geometry", p.width == 352 and p.height == 288,
      f"{p.width}x{p.height}")
check("it reports the real frame rate", 24.0 <= p.fps <= 26.0, f"{p.fps}")
check("it reports a duration", 9.0 <= p.duration_s <= 11.0, f"{p.duration_s}")

p = classify(probe_file(short, camera_id="CAM02"))
check("a short tiny clip is a placeholder, not footage",
      p.verdict == "placeholder", f"{p.verdict}: {p.reason}")
check("the placeholder reason names the remedy", "re-capture" in p.reason.lower(),
      p.reason)

p = classify(probe_file(truncated, camera_id="CAM03"))
check("a truncated download is corrupt", p.verdict == "corrupt",
      f"{p.verdict}: {p.reason}")
# The two verdicts must lead somewhere different. A damaged file is fetched
# again; an offline camera is re-captured later. Telling someone to re-capture
# a file that merely arrived broken wastes a day waiting for a working camera.
check("a damaged file is never blamed on an offline camera",
      "re-capture" not in p.reason.lower() and "fetch it again" in p.reason.lower(),
      p.reason)

p = classify(probe_file(notvideo, camera_id="CAM04"))
check("a non-video is unsupported", p.verdict == "unsupported",
      f"{p.verdict}: {p.reason}")

p = classify(probe_file(tmp / "nope.mp4", camera_id="CAM05"))
check("an absent file is missing", p.verdict == "missing", p.verdict)

# ---------------------------------------------------------------------------
print("\nordering: corrupt must beat placeholder")
# A file that is short, small AND undecodable must be reported as corrupt --
# otherwise someone re-captures a camera that was never offline.
broken_tiny = tmp / "broken_tiny.mp4"
broken_tiny.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40)
p = classify(probe_file(broken_tiny, camera_id="CAM06"))
check("an undecodable stub is never called a placeholder",
      p.verdict != "placeholder", f"{p.verdict}: {p.reason}")

# ---------------------------------------------------------------------------
print("\nstreams are reported, not opened")
probes = preflight_sources([("CAM07", "rtsp://10.0.0.5/stream"),
                            ("CAM08", "0"),
                            ("CAM09", str(good))])
by_id = {q.camera_id: q for q in probes}
check("an rtsp source is a stream", by_id["CAM07"].verdict == "stream")
check("a device index is a stream", by_id["CAM08"].verdict == "stream")
check("a file alongside streams is still judged", by_id["CAM09"].verdict == "ready")

# ---------------------------------------------------------------------------
print("\nANPR reachability is arithmetic, not opinion")
p = classify(probe_file(good, camera_id="CAM10"), min_plate_width_px=175)
check("a 352px frame warns that a 175px plate needs half the picture",
      any("must fill" in w for w in p.warnings), str(p.warnings))

p = classify(probe_file(good, camera_id="CAM11"), min_plate_width_px=400)
check("a threshold wider than the frame is declared impossible",
      any("NO PLATES POSSIBLE" in w for w in p.warnings), str(p.warnings))

p = classify(probe_file(good, camera_id="CAM12"), min_plate_width_px=60)
check("a reachable threshold produces no plate warning",
      not any("plate" in w.lower() for w in p.warnings), str(p.warnings))

check("warnings never change the verdict",
      classify(probe_file(good), min_plate_width_px=400).verdict == "ready")

# ---------------------------------------------------------------------------
print("\nreporting")
counts = summarise([classify(probe_file(good)), classify(probe_file(short)),
                    classify(probe_file(tmp / "nope.mp4"))])
check("the summary counts each verdict", counts["ready"] == 1
      and counts["placeholder"] == 1 and counts["missing"] == 1, str(counts))
check("a probe serialises for the remote report",
      isinstance(classify(probe_file(good)).as_dict(), dict))

print(f"\n{_passed} passed, {len(_failed)} FAILED")
if _failed:
    for n in _failed:
        print("   -", n)
    raise SystemExit(1)
