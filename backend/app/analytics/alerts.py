"""Alert rules.

Four rules, each derived from something the system observed rather than
something it assumed. Every alert carries the value that triggered it and the
threshold it crossed, so a controller can see *why* rather than being asked to
trust a red dot.

One rule deserves particular care. A watchlist hit is **flagged for review,
never asserted**. Matching is fuzzy because OCR misreads a character often
enough that exact matching would miss real hits -- but that means a hit is a
hypothesis. Every watchlist alert reports the similarity that produced it and
says a human should confirm. A system that announces "stolen vehicle detected"
on a 0.86 match is a system that gets someone stopped for nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from app.core.logging import get_logger
from app.vision.plate_grammar import normalise, similarity

log = get_logger("alerts")

SEVERITY_ORDER = {"critical": 3, "warning": 2, "info": 1}


@dataclass
class Alert:
    kind: str                    # congestion | overspeed | watchlist | coverage_gap
    severity: str                # critical | warning | info
    subject: str                 # camera id or "CAM01->CAM02"
    message: str
    detail: str = ""
    at: datetime | None = None
    value: float = 0.0
    threshold: float = 0.0
    plate: str | None = None
    confidence: float = 1.0
    needs_review: bool = False

    @property
    def rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)


@dataclass
class WatchlistEntry:
    plate: str
    reason: str = ""
    severity: str = "warning"


@dataclass
class Watchlist:
    entries: list[WatchlistEntry] = field(default_factory=list)
    min_similarity: float = 0.85

    def match(self, plate: str | None) -> tuple[WatchlistEntry, float] | None:
        """Best fuzzy match above the threshold, or None."""
        if not plate:
            return None
        target = normalise(plate)
        best, best_score = None, 0.0
        for entry in self.entries:
            score = similarity(target, entry.plate)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.min_similarity:
            return best, best_score
        return None


def load_watchlist(path: Path) -> Watchlist:
    if not Path(path).exists():
        return Watchlist()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = [
        WatchlistEntry(plate=normalise(str(e.get("plate", ""))),
                       reason=str(e.get("reason", "")),
                       severity=str(e.get("severity", "warning")))
        for e in (raw.get("plates") or []) if e.get("plate")
    ]
    return Watchlist(entries=entries,
                     min_similarity=float(raw.get("min_similarity", 0.85)))


class AlertEngine:
    def __init__(self, *, congestion_index: float = 1.5,
                 min_trips_for_congestion: int = 3,
                 coverage_gap_count: int = 2,
                 speed_limits: dict[str, float] | None = None,
                 watchlist: Watchlist | None = None) -> None:
        self.congestion_index = congestion_index
        self.min_trips_for_congestion = min_trips_for_congestion
        self.coverage_gap_count = coverage_gap_count
        self.speed_limits = speed_limits or {}
        self.watchlist = watchlist or Watchlist()

    # -- rules -------------------------------------------------------------

    def congestion(self, report) -> list[Alert]:
        out = []
        for corridor in report.corridors:
            # Refuse to call congestion from two trips. A threshold crossed by
            # a sample of two is not a finding.
            if corridor.trips < self.min_trips_for_congestion:
                continue
            if corridor.congestion_index < self.congestion_index:
                continue
            over = (corridor.median_travel_s - corridor.free_flow_travel_s) / 60.0
            out.append(Alert(
                kind="congestion",
                severity="critical" if corridor.congestion_index >= 2.0 else "warning",
                subject=corridor.label,
                message=f"{corridor.label} running {corridor.congestion_index:.1f}x "
                        f"slower than free flow ({corridor.level})",
                detail=f"median {corridor.median_travel_s/60:.1f} min against a "
                       f"{corridor.free_flow_travel_s/60:.1f} min baseline "
                       f"({'measured' if corridor.baseline_from_data else 'configured'}), "
                       f"+{over:.1f} min per vehicle over {corridor.trips} trips",
                value=corridor.congestion_index, threshold=self.congestion_index,
            ))
        return out

    def overspeed(self, hops) -> list[Alert]:
        """Legal-limit breaches, which are not the same as the plausibility
        ceiling. The resolver rejects journeys that are *impossible*; this flags
        journeys that are merely *illegal*."""
        out = []
        for hop in hops:
            key = f"{hop.from_camera}->{hop.to_camera}"
            limit = self.speed_limits.get(key)
            if not limit or hop.implied_speed_kmph <= limit:
                continue
            over = hop.implied_speed_kmph - limit
            out.append(Alert(
                kind="overspeed",
                severity="critical" if over > 20 else "warning",
                subject=key,
                message=f"{hop.implied_speed_kmph:.0f} km/h on {key} "
                        f"(limit {limit:.0f})",
                detail=f"{hop.distance_km:.1f} km covered in {hop.gap_s:.0f} s. "
                       f"Average over the segment, not an instantaneous reading.",
                at=getattr(hop, "arrived_at", None),
                value=hop.implied_speed_kmph, threshold=limit,
                plate=getattr(hop, "plate", None),
                needs_review=True,
            ))
        return out

    def watchlist_hits(self, sightings) -> list[Alert]:
        out = []
        seen: set[tuple[str, str]] = set()
        for s in sightings:
            match = self.watchlist.match(getattr(s, "plate_text", None))
            if match is None:
                continue
            entry, score = match
            key = (entry.plate, s.camera_id)
            if key in seen:
                continue
            seen.add(key)
            exact = normalise(s.plate_text) == entry.plate
            out.append(Alert(
                kind="watchlist",
                severity=entry.severity,
                subject=s.camera_id,
                message=f"possible watchlist match {entry.plate} at {s.camera_id}"
                        + ("" if exact else f" (read as {s.plate_text})"),
                detail=f"{entry.reason}. Plate similarity {score:.2f}"
                       + ("." if exact else " -- not an exact read.")
                       + " Flagged for human review, not confirmed.",
                at=s.first_seen, value=score,
                threshold=self.watchlist.min_similarity,
                plate=entry.plate, confidence=score, needs_review=True,
            ))
        return out

    def coverage_gaps(self, journeys) -> list[Alert]:
        counts: Counter[str] = Counter()
        for j in journeys:
            gaps = getattr(j, "coverage_gaps", None) or []
            if isinstance(gaps, str):
                gaps = [g for g in gaps.split(",") if g]
            counts.update(gaps)

        out = []
        for camera_id, n in counts.most_common():
            if n < self.coverage_gap_count:
                continue
            out.append(Alert(
                kind="coverage_gap",
                severity="warning" if n >= self.coverage_gap_count * 2 else "info",
                subject=camera_id,
                message=f"{camera_id} missed {n} vehicles that passed it",
                detail="These vehicles were seen either side of this camera on a "
                       "route that runs through it. Likely a detection, framing "
                       "or plate-readability problem rather than a traffic one.",
                value=float(n), threshold=float(self.coverage_gap_count),
            ))
        return out

    # -- all ---------------------------------------------------------------

    def evaluate(self, report, journeys, hops, sightings) -> list[Alert]:
        alerts = (self.congestion(report) + self.overspeed(hops)
                  + self.watchlist_hits(sightings) + self.coverage_gaps(journeys))
        alerts.sort(key=lambda a: (-a.rank, a.kind))
        log.info("%d alert(s): %s", len(alerts),
                 ", ".join(f"{k} {v}" for k, v in
                           Counter(a.kind for a in alerts).items()) or "none")
        return alerts
