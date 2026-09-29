"""The camera network as a graph, and the physics gate built on it.

This is the third of the three signals in Milestone 7, and the only one that
can *veto*. Plate similarity and appearance are evidence; travel time is a
constraint. A vehicle cannot cross 3.1 km in eight seconds, and no amount of
plate agreement makes that link real.

Being able to say "we rejected this pair because it implies 214 km/h" is worth
more in front of a judge than any accepted link, because it shows the system
reasons rather than pattern-matches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.logging import get_logger

log = get_logger("citygraph")

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance. The floor under any road distance."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    distance_km: float
    typical_speed_kmph: float
    max_speed_kmph: float
    min_speed_kmph: float
    speed_limit_kmph: float = 0.0     # legal limit; 0 = none configured
    derived: bool = False          # True when estimated from coordinates

    def travel_seconds(self, kmph: float) -> float:
        return (self.distance_km / max(kmph, 0.1)) * 3600.0

    @property
    def fastest_s(self) -> float:
        return self.travel_seconds(self.max_speed_kmph)

    @property
    def typical_s(self) -> float:
        return self.travel_seconds(self.typical_speed_kmph)

    @property
    def slowest_s(self) -> float:
        return self.travel_seconds(self.min_speed_kmph)


@dataclass
class GraphDefaults:
    detour_factor: float = 1.35
    max_speed_kmph: float = 80.0
    min_speed_kmph: float = 4.0
    typical_speed_kmph: float = 28.0
    speed_limit_kmph: float = 50.0
    max_gap_minutes: float = 30.0


class CityGraph:
    """Camera nodes with weighted edges, plus the plausibility test."""

    def __init__(self, cameras, defaults: GraphDefaults, auto_edges: bool = True) -> None:
        self.defaults = defaults
        self.auto_edges = auto_edges
        self.nodes = {c.id.upper(): c for c in cameras}
        self._edges: dict[tuple[str, str], Edge] = {}

    # -- construction ------------------------------------------------------

    def add_edge(self, a: str, b: str, distance_km: float,
                 typical_speed_kmph: float | None = None,
                 max_speed_kmph: float | None = None,
                 min_speed_kmph: float | None = None,
                 speed_limit_kmph: float | None = None,
                 bidirectional: bool = True, derived: bool = False) -> None:
        d = self.defaults
        edge = Edge(
            from_id=a.upper(), to_id=b.upper(), distance_km=float(distance_km),
            typical_speed_kmph=float(typical_speed_kmph or d.typical_speed_kmph),
            max_speed_kmph=float(max_speed_kmph or d.max_speed_kmph),
            min_speed_kmph=float(min_speed_kmph or d.min_speed_kmph),
            speed_limit_kmph=float(speed_limit_kmph or d.speed_limit_kmph),
            derived=derived,
        )
        self._edges[(edge.from_id, edge.to_id)] = edge
        if bidirectional:
            self._edges[(edge.to_id, edge.from_id)] = Edge(
                from_id=edge.to_id, to_id=edge.from_id, distance_km=edge.distance_km,
                typical_speed_kmph=edge.typical_speed_kmph,
                max_speed_kmph=edge.max_speed_kmph, min_speed_kmph=edge.min_speed_kmph,
                speed_limit_kmph=edge.speed_limit_kmph, derived=derived)

    def edge(self, a: str, b: str) -> Edge | None:
        """The edge from a to b, deriving one from coordinates if allowed."""
        key = (a.upper(), b.upper())
        if key in self._edges:
            return self._edges[key]
        if not self.auto_edges:
            return None

        na, nb = self.nodes.get(key[0]), self.nodes.get(key[1])
        if na is None or nb is None or key[0] == key[1]:
            return None

        straight = haversine_km(na.location.lat, na.location.lon,
                                nb.location.lat, nb.location.lon)
        self.add_edge(key[0], key[1], straight * self.defaults.detour_factor,
                      bidirectional=True, derived=True)
        return self._edges[key]

    # -- the gate ----------------------------------------------------------

    def check(self, from_camera: str, to_camera: str, gap_s: float) -> tuple[bool, float, str]:
        """Is this journey physically possible?

        Returns ``(possible, implied_speed_kmph, reason)``. The reason is
        written to be readable straight off a dashboard.
        """
        if from_camera.upper() == to_camera.upper():
            return False, 0.0, "same camera"
        if gap_s <= 0:
            return False, 0.0, "arrival is not after departure"
        if gap_s > self.defaults.max_gap_minutes * 60:
            return False, 0.0, (f"gap of {gap_s/60:.0f} min exceeds the "
                                f"{self.defaults.max_gap_minutes:.0f} min window")

        edge = self.edge(from_camera, to_camera)
        if edge is None:
            return False, 0.0, f"no route from {from_camera} to {to_camera}"

        implied = edge.distance_km / (gap_s / 3600.0)
        if implied > edge.max_speed_kmph:
            return False, implied, (f"implies {implied:.0f} km/h over "
                                    f"{edge.distance_km:.1f} km, above the "
                                    f"{edge.max_speed_kmph:.0f} km/h ceiling")
        if implied < edge.min_speed_kmph:
            return False, implied, (f"implies {implied:.1f} km/h -- too slow to be "
                                    f"the same continuous journey")
        return True, implied, f"{implied:.0f} km/h over {edge.distance_km:.1f} km"

    def plausibility(self, from_camera: str, to_camera: str, gap_s: float) -> float:
        """0-1 score for how *typical* this journey time is, once possible.

        A bell around the corridor's typical speed: a vehicle travelling at the
        usual pace scores 1.0; one crawling or racing (but still inside the
        hard limits) scores lower without being rejected.
        """
        ok, implied, _ = self.check(from_camera, to_camera, gap_s)
        if not ok:
            return 0.0
        edge = self.edge(from_camera, to_camera)
        ratio = implied / max(edge.typical_speed_kmph, 0.1)
        return float(math.exp(-((ratio - 1.0) ** 2) / (2 * 0.65 ** 2)))

    def shortest_path(self, a: str, b: str) -> tuple[list[str], float] | None:
        """Shortest route over DECLARED edges only, by road distance.

        Derived edges are excluded on purpose. A derived edge is a straight-line
        guess between two cameras, not a road -- routing over them would invent
        junctions that do not exist. Declared edges are the road network you
        actually measured, so a path through them is a real route, and any
        camera on it that did not see the vehicle is a genuine coverage gap.
        """
        a, b = a.upper(), b.upper()
        if a == b:
            return ([a], 0.0)

        adjacency: dict[str, list[tuple[str, float]]] = {}
        for edge in self._edges.values():
            if edge.derived:
                continue
            adjacency.setdefault(edge.from_id, []).append((edge.to_id, edge.distance_km))
        if a not in adjacency:
            return None

        import heapq

        dist = {a: 0.0}
        prev: dict[str, str] = {}
        queue = [(0.0, a)]
        seen: set[str] = set()
        while queue:
            d, node = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            if node == b:
                path = [b]
                while path[-1] != a:
                    path.append(prev[path[-1]])
                return (list(reversed(path)), d)
            for nxt, w in adjacency.get(node, ()):
                nd = d + w
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(queue, (nd, nxt))
        return None

    @property
    def declared_edges(self) -> list[Edge]:
        return [e for e in self._edges.values() if not e.derived]

    def summary(self) -> str:
        declared = len({tuple(sorted((e.from_id, e.to_id))) for e in self.declared_edges})
        return (f"{len(self.nodes)} cameras, {declared} declared edge(s), "
                f"auto_edges={'on' if self.auto_edges else 'off'}")


def load_city_graph(path: Path, cameras) -> CityGraph:
    """Read configs/city_graph.yaml. Missing file is fine -- we derive."""
    raw = {}
    if Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    d = raw.get("defaults", {}) or {}
    defaults = GraphDefaults(
        detour_factor=float(d.get("detour_factor", 1.35)),
        max_speed_kmph=float(d.get("max_speed_kmph", 80.0)),
        min_speed_kmph=float(d.get("min_speed_kmph", 4.0)),
        typical_speed_kmph=float(d.get("typical_speed_kmph", 28.0)),
        speed_limit_kmph=float(d.get("speed_limit_kmph", 50.0)),
        max_gap_minutes=float(d.get("max_gap_minutes", 30.0)),
    )
    graph = CityGraph(cameras, defaults, auto_edges=bool(raw.get("auto_edges", True)))

    for entry in raw.get("edges") or []:
        graph.add_edge(
            entry["from"], entry["to"], entry["distance_km"],
            typical_speed_kmph=entry.get("typical_speed_kmph"),
            max_speed_kmph=entry.get("max_speed_kmph"),
            min_speed_kmph=entry.get("min_speed_kmph"),
            speed_limit_kmph=entry.get("speed_limit_kmph"),
            bidirectional=bool(entry.get("bidirectional", True)),
        )
    log.info("city graph: %s", graph.summary())
    return graph
