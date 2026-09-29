"""Trajectory reconstruction.

Milestone 7 produces a *set* of sightings believed to be one vehicle, plus every
pairwise link it accepted. That is not yet a journey. Three things have to
happen:

**1. Chain, do not fan out.**
For a vehicle seen at CAM01, CAM02 and CAM03, the resolver accepts three links:
01->02, 02->03 and also 01->03. Only the first two are hops. A journey is the
chain of *consecutive* sightings in time order; the shortcut link is corroborating
evidence, not a leg of the trip.

**2. Route, do not just connect.**
Two consecutive sightings are joined by a road, and that road may pass cameras
that never saw the vehicle. Routing over the declared network recovers those,
and an unobserved camera on a route the vehicle certainly took is a **coverage
gap** worth reporting -- a miscalibrated camera, a missed detection, or a plate
the OCR could not read.

**3. Measure, do not just draw.**
Each hop carries distance, duration, implied speed and how that compares to the
corridor's typical speed. That ratio is the raw material for congestion
analytics in Milestone 9 -- a hop taken at half the usual speed is a delay,
observed rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.core.city_graph import CityGraph
from app.core.logging import get_logger

log = get_logger("trajectory")


@dataclass
class Hop:
    """One leg of a journey: between two consecutive observed cameras."""

    from_camera: str
    to_camera: str
    departed_at: datetime
    arrived_at: datetime
    gap_s: float
    distance_km: float
    implied_speed_kmph: float
    typical_speed_kmph: float
    confidence: float = 0.0
    via: list[str] = field(default_factory=list)     # unobserved cameras on the route
    routed: bool = False                             # distance came from the road network

    @property
    def delay_ratio(self) -> float:
        """How much slower than usual. 1.0 = typical, 2.0 = twice the time."""
        if self.implied_speed_kmph <= 0:
            return 0.0
        return round(self.typical_speed_kmph / self.implied_speed_kmph, 3)

    @property
    def congested(self) -> bool:
        return self.delay_ratio >= 1.5

    def describe(self) -> str:
        route = f" via {'/'.join(self.via)} (unobserved)" if self.via else ""
        return (f"{self.from_camera} -> {self.to_camera}{route}  "
                f"{self.gap_s/60:.1f} min  {self.distance_km:.1f} km  "
                f"{self.implied_speed_kmph:.0f} km/h")


@dataclass
class Journey:
    """One vehicle's reconstructed trip through the camera network."""

    identity_key: str
    plate: str | None
    plate_confidence: float
    sighting_ids: list[int]
    route: list[str]                                  # cameras that saw it
    hops: list[Hop] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)

    @property
    def started_at(self) -> datetime | None:
        return self.hops[0].departed_at if self.hops else None

    @property
    def ended_at(self) -> datetime | None:
        return self.hops[-1].arrived_at if self.hops else None

    @property
    def total_distance_km(self) -> float:
        return round(sum(h.distance_km for h in self.hops), 3)

    @property
    def total_duration_s(self) -> float:
        if not self.hops:
            return 0.0
        return round((self.ended_at - self.started_at).total_seconds(), 2)

    @property
    def average_speed_kmph(self) -> float:
        if self.total_duration_s <= 0:
            return 0.0
        return round(self.total_distance_km / (self.total_duration_s / 3600.0), 2)

    @property
    def confidence(self) -> float:
        """A journey is exactly as trustworthy as its weakest hop.

        Not rounded here: rounding belongs at the display and persistence
        edges, not in the value itself.
        """
        return min((h.confidence for h in self.hops), default=0.0)

    @property
    def full_route(self) -> list[str]:
        """Observed cameras with inferred ones folded in, in travel order."""
        if not self.hops:
            return list(self.route)
        out = [self.hops[0].from_camera]
        for hop in self.hops:
            out.extend(hop.via)
            out.append(hop.to_camera)
        return out

    @property
    def slowest_hop(self) -> Hop | None:
        return max(self.hops, key=lambda h: h.delay_ratio) if self.hops else None

    def summary(self) -> str:
        mins, secs = divmod(int(self.total_duration_s), 60)
        return (f"{' -> '.join(self.route)}  |  {self.total_distance_km:.1f} km  |  "
                f"{mins} min {secs:02d} s  |  {self.average_speed_kmph:.0f} km/h avg  |  "
                f"confidence {self.confidence:.2f}")


class TrajectoryEngine:
    def __init__(self, graph: CityGraph) -> None:
        self.graph = graph

    def build(self, identity, sightings_by_id: dict, link_scores: dict) -> Journey:
        """Turn one resolved identity into an ordered journey.

        ``link_scores`` maps ``(from_sighting_id, to_sighting_id)`` to the score
        the resolver gave that pair, so each hop keeps the confidence it earned.
        """
        members = [sightings_by_id[sid] for sid in identity.sighting_ids
                   if sid in sightings_by_id]
        members.sort(key=lambda s: s.first_seen)

        journey = Journey(
            identity_key=identity.key,
            plate=identity.plate,
            plate_confidence=round(identity.plate_confidence or 0.0, 4),
            sighting_ids=[m.id for m in members],
            route=[m.camera_id for m in members],
        )

        # Consecutive pairs only. The resolver's shortcut links corroborate the
        # journey; they are not legs of it.
        for a, b in zip(members, members[1:]):
            if a.camera_id == b.camera_id:
                continue
            gap_s = (b.first_seen - a.last_seen).total_seconds()

            # Prefer the real road route over a straight-line estimate.
            path = self.graph.shortest_path(a.camera_id, b.camera_id)
            edge = self.graph.edge(a.camera_id, b.camera_id)
            if path is not None:
                nodes, distance_km = path
                via = nodes[1:-1]
                routed = True
            else:
                nodes, distance_km = [a.camera_id, b.camera_id], (edge.distance_km if edge else 0.0)
                via, routed = [], False

            implied = (distance_km / (gap_s / 3600.0)) if gap_s > 0 else 0.0
            typical = edge.typical_speed_kmph if edge else self.graph.defaults.typical_speed_kmph

            journey.hops.append(Hop(
                from_camera=a.camera_id, to_camera=b.camera_id,
                departed_at=a.last_seen, arrived_at=b.first_seen,
                gap_s=round(gap_s, 2), distance_km=round(distance_km, 3),
                implied_speed_kmph=round(implied, 2),
                typical_speed_kmph=typical,
                confidence=link_scores.get((a.id, b.id),
                                           link_scores.get((b.id, a.id), 0.0)),
                via=via, routed=routed,
            ))
            journey.coverage_gaps.extend(via)

        # Same camera twice on one route is not a gap worth reporting twice.
        journey.coverage_gaps = list(dict.fromkeys(journey.coverage_gaps))
        return journey

    def build_all(self, identities, sightings_by_id: dict, verdicts) -> list[Journey]:
        scores = {(v.from_id, v.to_id): v.score for v in verdicts if v.accepted}
        journeys = [self.build(i, sightings_by_id, scores) for i in identities]
        journeys = [j for j in journeys if j.hops]
        journeys.sort(key=lambda j: (-len(j.hops), j.started_at or datetime.min))
        gaps = sum(len(j.coverage_gaps) for j in journeys)
        log.info("built %d journey(s), %d hop(s), %d coverage gap(s)",
                 len(journeys), sum(len(j.hops) for j in journeys), gaps)
        return journeys
