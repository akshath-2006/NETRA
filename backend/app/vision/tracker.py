"""Multi-object tracking and the Track lifecycle.

Detection answers "what is in this frame". Tracking answers "is this the same
vehicle as last frame" -- and that second question is what everything
downstream is built on. Without stable IDs there is no such thing as a vehicle
*sighting*, only a pile of unrelated boxes: the plate consensus in M5 has
nothing to vote over, and the cross-camera association in M7 has nothing to
associate.

We use ByteTrack through ultralytics rather than writing our own. It is strong,
it is one argument, and hand-rolling a tracker would cost days and still lose.
Our own work is the layer on top -- the ``Track`` objects here accumulate the
history that becomes a persisted sighting in Milestone 4:

* **Class voting.** A vehicle flickers between "car" and "truck" across frames.
  We keep confidence-weighted votes and report the winner, so one bad frame
  cannot relabel a vehicle. This is the same principle as plate consensus in
  M5, introduced early because it matters just as much here.
* **Lifecycle.** Tracks are born, live, go unseen, and are closed out. A closed
  track is exactly one sighting record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging import get_logger


def _sample_colour(image, box) -> str | None:
    """Mean colour of the middle of the box, as RGB hex."""
    from app.identity.appearance import to_hex

    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = image.shape[:2]
    dx, dy = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
    x1, y1 = max(0, x1 + dx), max(0, y1 + dy)
    x2, y2 = min(w, x2 - dx), min(h, y2 - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    return to_hex(patch.reshape(-1, 3).mean(axis=0)) if patch.size else None


def _crop(image, box, pad_frac: float = 0.12):
    """Vehicle crop for the evidence snapshot, padded and copied.

    The copy is not optional: the overlay draws boxes and the HUD onto the
    same frame buffer a few lines later, so a view into it would become an
    annotated frame by the time we wrote it out.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = image.shape[:2]
    px, py = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


@dataclass
class Track:
    """One vehicle followed through one camera's field of view."""

    track_id: int
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    first_frame: int
    last_frame: int

    frames: int = 0
    class_votes: dict[str, float] = field(default_factory=dict)
    centers: deque = field(default_factory=lambda: deque(maxlen=32))
    last_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    best_confidence: float = 0.0
    best_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    colour_hex: str | None = None      # dominant colour, sampled on the best frame

    missing: int = 0
    counted: bool = False
    crossing: str | None = None

    # Milestone 5. Every OCR attempt made on this vehicle, and the single
    # answer voted out of them when the track closes.
    plate_readings: list = field(default_factory=list)
    plate: object | None = None

    # ANPR budget bookkeeping. OCR is the second most expensive thing we do,
    # and reading the same vehicle for the fortieth time tells us nothing the
    # first ten did not. We stop once the evidence is good enough, and reopen
    # only when a materially larger plate appears -- a bigger crop is better
    # evidence and can correct an earlier truncated read.
    anpr_good_reads: int = 0
    best_plate_width: int = 0

    # Evidence snapshot. One crop, taken at the frame where the detector was
    # most confident -- which is also the frame most worth showing a human.
    # Held in memory only while the track is open (bounded by active tracks,
    # ~60 KB each) and written to disk exactly once when the track closes.
    # Copied on capture because the overlay draws onto the frame in place.
    best_crop: object | None = None
    best_frame_index: int = 0
    best_seen_at: datetime | None = None
    snapshot: str | None = None        # filename once written

    @property
    def label(self) -> str:
        """Confidence-weighted majority class, not just the latest guess."""
        if not self.class_votes:
            return "unknown"
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def label_confidence(self) -> float:
        total = sum(self.class_votes.values())
        return (max(self.class_votes.values()) / total) if total else 0.0

    @property
    def duration_s(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def aspect(self) -> float:
        x1, y1, x2, y2 = self.best_box
        h = max(1, y2 - y1)
        return round((x2 - x1) / h, 3)

    def is_stable(self, min_frames: int) -> bool:
        return self.frames >= min_frames

    def to_sighting(self) -> dict:
        """The shape Milestone 4 will persist. Defined here so the contract
        between tracking and the event store is visible in one place."""
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "vehicle_class": self.label,
            "class_confidence": round(self.label_confidence, 3),
            "detection_confidence": round(self.best_confidence, 3),
            "first_seen": self.first_seen.isoformat(timespec="milliseconds"),
            "last_seen": self.last_seen.isoformat(timespec="milliseconds"),
            "duration_s": round(self.duration_s, 2),
            "frames": self.frames,
            "crossing": self.crossing,
            "best_box": list(self.best_box),
            "plate": getattr(self.plate, "text", None),
            "plate_confidence": getattr(self.plate, "confidence", None),
            "plate_reads": len(self.plate_readings),
            "colour_hex": self.colour_hex,
            "aspect": self.aspect,
        }


class TrackManager:
    """Owns the Track objects for one camera and closes them out when done."""

    def __init__(
        self,
        camera_id: str,
        min_frames: int = 5,
        max_age_frames: int = 30,
        trail_length: int = 32,
        keep_snapshots: bool = True,
        snapshot_pad: float = 0.12,
    ) -> None:
        self.camera_id = camera_id
        self.min_frames = int(min_frames)
        self.max_age_frames = int(max_age_frames)
        self.trail_length = int(trail_length)
        self.keep_snapshots = bool(keep_snapshots)
        self.snapshot_pad = float(snapshot_pad)

        self.active: dict[int, Track] = {}
        self._completed: list[Track] = []
        self.total_created = 0
        self.log = get_logger(f"tracks.{camera_id}")

    def update(self, detections, frame) -> dict[int, Track]:
        """Fold this frame's detections into the track set."""
        seen: set[int] = set()

        for det in detections:
            tid = getattr(det, "track_id", None)
            if tid is None:
                continue  # tracker had no id for this box; ignore it
            tid = int(tid)
            seen.add(tid)

            track = self.active.get(tid)
            if track is None:
                track = Track(
                    track_id=tid,
                    camera_id=self.camera_id,
                    first_seen=frame.timestamp,
                    last_seen=frame.timestamp,
                    first_frame=frame.index,
                    last_frame=frame.index,
                    centers=deque(maxlen=self.trail_length),
                )
                self.active[tid] = track
                self.total_created += 1

            track.last_seen = frame.timestamp
            track.last_frame = frame.index
            track.frames += 1
            track.missing = 0
            track.last_box = det.box
            track.centers.append(tuple(int(v) for v in det.center))
            track.class_votes[det.label] = track.class_votes.get(det.label, 0.0) + det.confidence
            if det.confidence > track.best_confidence:
                track.best_confidence = det.confidence
                track.best_box = det.box
                track.best_frame_index = frame.index
                track.best_seen_at = frame.timestamp
                # Sample colour from the centre of the box only: the edges are
                # mostly road and sky, which would wash every vehicle grey.
                image = getattr(frame, "image", None)
                if image is not None:
                    track.colour_hex = _sample_colour(image, det.box)
                    if self.keep_snapshots:
                        track.best_crop = _crop(image, det.box, self.snapshot_pad)

        # Age out anything not seen this frame.
        for tid, track in list(self.active.items()):
            if tid in seen:
                continue
            track.missing += 1
            if track.missing > self.max_age_frames:
                self.active.pop(tid)
                if track.is_stable(self.min_frames):
                    self._completed.append(track)
                    self.log.debug("closed track %d (%s, %d frames, %.1fs)",
                                   tid, track.label, track.frames, track.duration_s)

        return self.active

    def drain_completed(self) -> list[Track]:
        """Return newly-finished tracks and forget them."""
        done, self._completed = self._completed, []
        return done

    def flush(self) -> list[Track]:
        """Close every remaining track. Call at end of stream."""
        for track in self.active.values():
            if track.is_stable(self.min_frames):
                self._completed.append(track)
        self.active.clear()
        return self.drain_completed()

    @property
    def stable_active(self) -> list[Track]:
        return [t for t in self.active.values() if t.is_stable(self.min_frames)]


def build_track_manager(cfg, camera_id: str) -> TrackManager:
    sysd = cfg.raw.get("system", {}) or {}
    t = sysd.get("tracking", {}) or {}
    ev = sysd.get("evidence", {}) or {}
    return TrackManager(
        camera_id=camera_id,
        min_frames=t.get("min_frames", 5),
        max_age_frames=t.get("max_age_frames", 30),
        trail_length=t.get("trail_length", 32),
        keep_snapshots=ev.get("keep_snapshots", True),
        snapshot_pad=ev.get("snapshot_pad_frac", 0.12),
    )
