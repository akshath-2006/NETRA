"""Tests for road routing and journey reconstruction.

    PYTHONPATH=backend python backend/tests/test_trajectory.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.city_graph import CityGraph, GraphDefaults      # noqa: E402
from app.identity.resolver import IdentityResolver, SightingView  # noqa: E402
from app.trajectory.engine import TrajectoryEngine            # noqa: E402

T0 = datetime(2026, 9, 3, 10, 42, 0)
_passed, _failed = 0, []


def check(name, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def cam(cid, lat, lon):
    return SimpleNamespace(id=cid, name=cid, location=SimpleNamespace(lat=lat, lon=lon))


CAMERAS = [cam("CAM01", 12.9756, 77.6068), cam("CAM02", 12.9731, 77.6195),
           cam("CAM03", 12.9604, 77.6472), cam("CAM04", 12.9500, 77.6700)]


def graph():
    """A chain: 01 -- 02 -- 03 -- 04. No direct long edges, like a real network."""
    g = CityGraph(CAMERAS, GraphDefaults())
    g.add_edge("CAM01", "CAM02", 1.6, typical_speed_kmph=26, max_speed_kmph=60)
    g.add_edge("CAM02", "CAM03", 3.1, typical_speed_kmph=32, max_speed_kmph=70)
    g.add_edge("CAM03", "CAM04", 2.2, typical_speed_kmph=30, max_speed_kmph=60)
    return g


def view(sid, camera, offset_s, plate="KA01AB1234", dwell=3.0):
    first = T0 + timedelta(seconds=offset_s)
    return SightingView(id=sid, camera_id=camera, first_seen=first,
                        last_seen=first + timedelta(seconds=dwell),
                        vehicle_class="car", plate_text=plate, plate_confidence=0.88,
                        colour_hex="3C6EBE", aspect=2.3)


def journey_from(views):
    r = IdentityResolver(graph())
    identities, verdicts = r.resolve(views)
    engine = TrajectoryEngine(graph())
    js = engine.build_all(identities, {v.id: v for v in views}, verdicts)
    return js


# ---------------------------------------------------------------------------
print("\nrouting over the declared network")

g = graph()
path, dist = g.shortest_path("CAM01", "CAM03")
check("routes through the intermediate camera", path == ["CAM01", "CAM02", "CAM03"])
check("route distance is the sum of segments", abs(dist - 4.7) < 1e-6, str(dist))

path, dist = g.shortest_path("CAM01", "CAM04")
check("routes across three segments", path == ["CAM01", "CAM02", "CAM03", "CAM04"])
check("and sums all of them", abs(dist - 6.9) < 1e-6, str(dist))

check("adjacent cameras route directly", g.shortest_path("CAM02", "CAM03")[0]
      == ["CAM02", "CAM03"])
check("a camera routes to itself at zero distance",
      g.shortest_path("CAM01", "CAM01") == (["CAM01"], 0.0))
check("routing is symmetric",
      g.shortest_path("CAM03", "CAM01")[1] == g.shortest_path("CAM01", "CAM03")[1])
check("unknown camera has no route", g.shortest_path("CAM01", "CAM99") is None)

# Derived edges are straight-line guesses, not roads -- routing must ignore them.
bare = CityGraph(CAMERAS, GraphDefaults())
bare.edge("CAM01", "CAM03")                      # force a derived edge to exist
check("derived edges are excluded from routing", bare.shortest_path("CAM01", "CAM03") is None)

# ---------------------------------------------------------------------------
print("\njourney reconstruction")

js = journey_from([view(1, "CAM01", 0), view(2, "CAM02", 240), view(3, "CAM03", 640)])
check("one vehicle gives one journey", len(js) == 1, f"{len(js)}")
j = js[0]
check("hops are consecutive pairs, not every pair", len(j.hops) == 2, f"{len(j.hops)} hops")
check("route is in travel order", j.route == ["CAM01", "CAM02", "CAM03"])
check("distance sums the legs", abs(j.total_distance_km - 4.7) < 1e-6,
      str(j.total_distance_km))
check("duration spans first departure to last arrival",
      abs(j.total_duration_s - 637) < 1.0, str(j.total_duration_s))
check("average speed is distance over time", 25 < j.average_speed_kmph < 30,
      str(j.average_speed_kmph))
check("journey confidence is the weakest hop",
      abs(j.confidence - min(h.confidence for h in j.hops)) < 1e-9)
check("summary reads as a sentence", "km" in j.summary() and "confidence" in j.summary())

first = j.hops[0]
check("hop records real elapsed time", 230 < first.gap_s < 245, str(first.gap_s))
check("hop uses the road distance", first.distance_km == 1.6)
check("hop computes implied speed", 20 < first.implied_speed_kmph < 30)
check("hop knows the corridor's usual speed", first.typical_speed_kmph == 26)

# ---------------------------------------------------------------------------
print("\ncoverage gaps")

# Seen at CAM01 and CAM03 only -- CAM02 is on the road but never saw it.
js = journey_from([view(1, "CAM01", 0), view(2, "CAM03", 640)])
check("a skipped camera still produces a journey", len(js) == 1 and len(js[0].hops) == 1)
j = js[0]
check("the unobserved camera on the route is identified", j.hops[0].via == ["CAM02"],
      str(j.hops[0].via))
check("it is reported as a coverage gap", j.coverage_gaps == ["CAM02"])
check("distance uses the routed road, not a straight line",
      abs(j.hops[0].distance_km - 4.7) < 1e-6, str(j.hops[0].distance_km))
check("the full route includes the inferred camera",
      j.full_route == ["CAM01", "CAM02", "CAM03"], str(j.full_route))
check("the observed route does not", j.route == ["CAM01", "CAM03"])
check("the hop is marked as routed", j.hops[0].routed)

js = journey_from([view(1, "CAM01", 0), view(2, "CAM02", 240), view(3, "CAM03", 640)])
check("no gaps when every camera saw it", js[0].coverage_gaps == [])

# ---------------------------------------------------------------------------
print("\ncongestion from real travel times")

fast = journey_from([view(1, "CAM01", 0), view(2, "CAM02", 222)])[0].hops[0]
check("a hop at the usual pace is not congested",
      not fast.congested and 0.9 < fast.delay_ratio < 1.1, str(fast.delay_ratio))

slow = journey_from([view(1, "CAM01", 0), view(2, "CAM02", 600)])[0].hops[0]
check("a hop at half the usual pace is flagged congested", slow.congested,
      str(slow.delay_ratio))
check("delay ratio says how much slower", slow.delay_ratio > 2.0, str(slow.delay_ratio))
check("the slowest hop is identifiable",
      journey_from([view(1, "CAM01", 0), view(2, "CAM02", 600),
                    view(3, "CAM03", 1000)])[0].slowest_hop.to_camera == "CAM02")

# ---------------------------------------------------------------------------
print("\nedge cases")

check("a single sighting is not a journey", journey_from([view(1, "CAM01", 0)]) == [])
check("no sightings gives no journeys", journey_from([]) == [])

two = journey_from([view(1, "CAM01", 0), view(2, "CAM02", 240),
                    view(3, "CAM01", 20, plate="TN09BX2255"),
                    view(4, "CAM02", 260, plate="TN09BX2255")])
check("two vehicles give two journeys", len(two) == 2, f"{len(two)}")
check("journeys keep their own plates",
      {j.plate for j in two} == {"KA01AB1234", "TN09BX2255"})

j = journey_from([view(1, "CAM01", 0), view(2, "CAM02", 240)])[0]
check("plate is carried onto the journey", j.plate == "KA01AB1234")
check("hop describes itself readably", "km/h" in j.hops[0].describe())

# ---------------------------------------------------------------------------
print()
if _failed:
    print(f"{_passed} passed, {len(_failed)} FAILED: {', '.join(_failed)}")
    raise SystemExit(1)
print(f"{_passed} passed, 0 failed")
