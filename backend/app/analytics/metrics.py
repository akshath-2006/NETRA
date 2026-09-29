"""Traffic analytics.

Everything here is computed from observations the system actually made. No
figure is modelled, extrapolated or filled in, and where there is not enough
data to say something the report says so rather than producing a confident
number from three samples.

THE ONE IDEA WORTH DEFENDING
Congestion is measured from **travel time**, not vehicle count. Forty vehicles
moving freely is not congestion; four vehicles crawling is. Counting cameras
can only tell you how many passed; a system that reconstructs journeys can tell
you how long they took, which is the thing a traffic controller actually needs.

The free-flow baseline is taken from the data itself -- the 15th percentile of
observed travel times on that corridor -- falling back to the configured
typical speed when there are too few trips to trust a percentile.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median

MIN_TRIPS_FOR_BASELINE = 5      # below this, a percentile is noise


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. q in 0..1."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass
class VolumeBucket:
    start: datetime
    end: datetime
    count: int = 0
    by_class: dict[str, int] = field(default_factory=dict)


@dataclass
class CameraStats:
    camera_id: str
    sightings: int = 0
    share: float = 0.0
    plates_read: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    by_class: dict[str, int] = field(default_factory=dict)

    @property
    def plate_rate(self) -> float:
        """Share of vehicles this camera got a plate from.

        A quality signal, not a traffic one: a camera reading 20% while its
        neighbours read 70% is badly sited or out of focus.
        """
        return self.plates_read / self.sightings if self.sightings else 0.0


@dataclass
class CorridorStats:
    from_camera: str
    to_camera: str
    trips: int = 0
    distance_km: float = 0.0
    median_travel_s: float = 0.0
    free_flow_travel_s: float = 0.0
    median_speed_kmph: float = 0.0
    free_flow_speed_kmph: float = 0.0
    baseline_from_data: bool = False

    @property
    def congestion_index(self) -> float:
        """Median travel time over free-flow. 1.0 = free-flow, 2.0 = twice as long."""
        if self.free_flow_travel_s <= 0:
            return 0.0
        return round(self.median_travel_s / self.free_flow_travel_s, 3)

    @property
    def level(self) -> str:
        idx = self.congestion_index
        if idx <= 0:
            return "unknown"
        if idx < 1.15:
            return "free flow"
        if idx < 1.50:
            return "moderate"
        if idx < 2.00:
            return "heavy"
        return "severe"

    @property
    def label(self) -> str:
        return f"{self.from_camera}->{self.to_camera}"


@dataclass
class AnalyticsReport:
    window_start: datetime | None = None
    window_end: datetime | None = None
    total_sightings: int = 0
    total_journeys: int = 0
    plates_read: int = 0
    buckets: list[VolumeBucket] = field(default_factory=list)
    cameras: list[CameraStats] = field(default_factory=list)
    corridors: list[CorridorStats] = field(default_factory=list)
    od_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    vehicle_mix: dict[str, int] = field(default_factory=dict)
    bucket_minutes: int = 5

    @property
    def plate_rate(self) -> float:
        return self.plates_read / self.total_sightings if self.total_sightings else 0.0

    @property
    def peak_bucket(self) -> VolumeBucket | None:
        return max(self.buckets, key=lambda b: b.count) if self.buckets else None

    @property
    def hotspots(self) -> list[CorridorStats]:
        """Corridors ranked by delay weighted by how many vehicles it affects.

        A corridor twice as slow as usual matters more when a hundred vehicles
        use it than when two do.
        """
        scored = [c for c in self.corridors if c.congestion_index > 1.0]
        return sorted(scored, key=lambda c: c.congestion_index * c.trips, reverse=True)

    @property
    def busiest_camera(self) -> CameraStats | None:
        return max(self.cameras, key=lambda c: c.sightings) if self.cameras else None


def _bucket_start(when: datetime, minutes: int) -> datetime:
    floor = (when.minute // minutes) * minutes
    return when.replace(minute=floor, second=0, microsecond=0)


def build_report(sightings, journeys, hops, bucket_minutes: int = 5,
                 typical_speeds: dict[str, float] | None = None) -> AnalyticsReport:
    """Compute every metric from observed sightings, journeys and hops.

    ``sightings`` needs camera_id, vehicle_class, plate_text, first_seen.
    ``hops`` needs from_camera, to_camera, gap_s, distance_km.
    ``journeys`` needs a route (list of cameras).
    """
    report = AnalyticsReport(bucket_minutes=bucket_minutes)
    typical_speeds = typical_speeds or {}

    if not sightings:
        return report

    times = [s.first_seen for s in sightings]
    report.window_start, report.window_end = min(times), max(times)
    report.total_sightings = len(sightings)
    report.total_journeys = len(journeys)
    report.plates_read = sum(1 for s in sightings if s.plate_text)
    report.vehicle_mix = dict(Counter(s.vehicle_class for s in sightings).most_common())

    # -- volume over time --------------------------------------------------
    grouped: dict[datetime, list] = defaultdict(list)
    for s in sightings:
        grouped[_bucket_start(s.first_seen, bucket_minutes)].append(s)

    cursor = _bucket_start(report.window_start, bucket_minutes)
    step = timedelta(minutes=bucket_minutes)
    last = _bucket_start(report.window_end, bucket_minutes)
    while cursor <= last:                      # include empty buckets: a gap is data
        members = grouped.get(cursor, [])
        report.buckets.append(VolumeBucket(
            start=cursor, end=cursor + step, count=len(members),
            by_class=dict(Counter(m.vehicle_class for m in members)),
        ))
        cursor += step

    # -- per camera --------------------------------------------------------
    by_camera: dict[str, list] = defaultdict(list)
    for s in sightings:
        by_camera[s.camera_id].append(s)
    for camera_id, members in sorted(by_camera.items()):
        report.cameras.append(CameraStats(
            camera_id=camera_id, sightings=len(members),
            share=len(members) / len(sightings),
            plates_read=sum(1 for m in members if m.plate_text),
            first_seen=min(m.first_seen for m in members),
            last_seen=max(m.first_seen for m in members),
            by_class=dict(Counter(m.vehicle_class for m in members)),
        ))

    # -- corridors ---------------------------------------------------------
    by_corridor: dict[tuple[str, str], list] = defaultdict(list)
    for h in hops:
        by_corridor[(h.from_camera, h.to_camera)].append(h)

    for (a, b), legs in sorted(by_corridor.items()):
        travel = [h.gap_s for h in legs if h.gap_s > 0]
        if not travel:
            continue
        distance = median([h.distance_km for h in legs])
        med = median(travel)

        # Free flow from the data where there is enough of it; otherwise from
        # the corridor's configured typical speed. Which one was used is
        # reported, because a baseline from four trips is not a baseline.
        if len(travel) >= MIN_TRIPS_FOR_BASELINE:
            free_flow_s = percentile(travel, 0.15)
            from_data = True
        else:
            typical = typical_speeds.get(f"{a}->{b}", 0.0)
            free_flow_s = (distance / typical) * 3600.0 if typical > 0 else med
            from_data = False

        report.corridors.append(CorridorStats(
            from_camera=a, to_camera=b, trips=len(legs),
            distance_km=round(distance, 3),
            median_travel_s=round(med, 2),
            free_flow_travel_s=round(free_flow_s, 2),
            median_speed_kmph=round(distance / (med / 3600.0), 2) if med > 0 else 0.0,
            free_flow_speed_kmph=round(distance / (free_flow_s / 3600.0), 2)
            if free_flow_s > 0 else 0.0,
            baseline_from_data=from_data,
        ))

    # -- origin / destination ----------------------------------------------
    od: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for j in journeys:
        route = getattr(j, "route", None) or []
        if isinstance(route, str):
            route = [r for r in route.split(">") if r]
        if len(route) >= 2:
            od[route[0]][route[-1]] += 1
    report.od_matrix = {k: dict(v) for k, v in od.items()}

    return report
