"""Tests for traffic metrics and alert rules.

The rules that matter most here are the ones that REFUSE to fire: congestion
from two trips, a watchlist hit below the similarity floor. A dashboard that
cries wolf is worse than one that says nothing.

    PYTHONPATH=backend python backend/tests/test_analytics.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analytics.alerts import (AlertEngine, Watchlist,  # noqa: E402
                                  WatchlistEntry)
from app.analytics.metrics import build_report, percentile  # noqa: E402

T0 = datetime(2026, 9, 3, 10, 0, 0)
_passed, _failed = 0, []


def check(name, condition, detail=""):
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def sight(camera, minute, klass="car", plate="KA01AB1234"):
    return SimpleNamespace(camera_id=camera, vehicle_class=klass, plate_text=plate,
                           first_seen=T0 + timedelta(minutes=minute))


def hop(a, b, gap_s, km=1.6, speed=None, arrived=None, plate=None):
    return SimpleNamespace(from_camera=a, to_camera=b, gap_s=gap_s, distance_km=km,
                           implied_speed_kmph=speed if speed is not None
                           else km / (gap_s / 3600.0),
                           typical_speed_kmph=26.0, delay_ratio=1.0,
                           arrived_at=arrived or T0, plate=plate)


def journey(route, gaps=()):
    return SimpleNamespace(route=list(route), coverage_gaps=list(gaps))


# ---------------------------------------------------------------------------
print("\npercentiles")

check("median of an odd list", percentile([1, 2, 3], 0.5) == 2)
check("interpolates between values", percentile([0, 10], 0.5) == 5)
check("p15 sits near the low end", percentile(list(range(101)), 0.15) == 15)
check("single value is its own percentile", percentile([7], 0.9) == 7)
check("empty input is zero, not a crash", percentile([], 0.5) == 0.0)

# ---------------------------------------------------------------------------
print("\nvolume and mix")

sightings = ([sight("CAM01", 0), sight("CAM01", 1), sight("CAM02", 2, "truck")]
             + [sight("CAM01", 11) for _ in range(6)])
report = build_report(sightings, [], [], bucket_minutes=5)

check("counts every sighting", report.total_sightings == 9)
check("buckets by the configured window", report.bucket_minutes == 5)
check("empty buckets are kept -- a gap is data too", len(report.buckets) == 3,
      f"{len(report.buckets)} buckets")
check("the quiet bucket is genuinely empty", report.buckets[1].count == 0)
check("peak bucket is the busy one", report.peak_bucket.count == 6)
check("vehicle mix is counted", report.vehicle_mix == {"car": 8, "truck": 1},
      str(report.vehicle_mix))
check("window spans first to last", report.window_start == T0)

# ---------------------------------------------------------------------------
print("\ncamera utilisation")

mixed = [sight("CAM01", 0), sight("CAM01", 1), sight("CAM01", 2, plate=None),
         sight("CAM02", 3, plate=None), sight("CAM02", 4, plate=None)]
report = build_report(mixed, [], [])
cams = {c.camera_id: c for c in report.cameras}
check("sightings attributed per camera", cams["CAM01"].sightings == 3)
check("share sums to one", abs(sum(c.share for c in report.cameras) - 1.0) < 1e-9)
check("plate rate is a quality signal, not a traffic one",
      abs(cams["CAM01"].plate_rate - 2 / 3) < 1e-9 and cams["CAM02"].plate_rate == 0.0)
check("busiest camera identified", report.busiest_camera.camera_id == "CAM01")

# ---------------------------------------------------------------------------
print("\ncorridor travel times and congestion")

# Six trips: five near 240 s, one crawling. Baseline comes from the data.
legs = [hop("CAM01", "CAM02", s) for s in (230, 235, 240, 245, 250, 700)]
report = build_report([sight("CAM01", 0)], [], legs)
corridor = report.corridors[0]
check("corridor is identified", corridor.label == "CAM01->CAM02")
check("trips are counted", corridor.trips == 6)
check("baseline is measured when there is enough data", corridor.baseline_from_data)
check("median ignores the outlier", 240 <= corridor.median_travel_s <= 250,
      str(corridor.median_travel_s))
check("congestion index is a ratio above one", corridor.congestion_index > 1.0)
check("free flow is classified as free flow",
      build_report([sight("CAM01", 0)], [],
                   [hop("CAM01", "CAM02", 240) for _ in range(6)]
                   ).corridors[0].level == "free flow")

boundary = build_report([sight("CAM01", 0)], [],
                        [hop("CAM01", "CAM02", s) for s in
                         (200, 205, 210, 600, 620, 640)]).corridors[0]
check("just under 2x free flow is heavy, not severe",
      boundary.level == "heavy" and 1.9 < boundary.congestion_index < 2.0,
      f"{boundary.level} {boundary.congestion_index}")

severe = build_report([sight("CAM01", 0)], [],
                      [hop("CAM01", "CAM02", s) for s in
                       (200, 205, 210, 700, 740, 780)]).corridors[0]
check("well past 2x free flow is severe",
      severe.level == "severe" and severe.congestion_index > 2.0,
      f"{severe.level} {severe.congestion_index}")

thin = build_report([sight("CAM01", 0)], [],
                    [hop("CAM01", "CAM02", 240), hop("CAM01", "CAM02", 600)],
                    typical_speeds={"CAM01->CAM02": 26.0}).corridors[0]
check("with too few trips the baseline comes from config, and says so",
      not thin.baseline_from_data)

# ---------------------------------------------------------------------------
print("\norigin-destination")

report = build_report([sight("CAM01", 0)], [journey(["CAM01", "CAM02", "CAM03"]),
                                            journey(["CAM01", "CAM02", "CAM03"]),
                                            journey(["CAM03", "CAM01"])], [])
check("OD counts trips by endpoints, not by hops",
      report.od_matrix["CAM01"]["CAM03"] == 2, str(report.od_matrix))
check("reverse direction is its own pair", report.od_matrix["CAM03"]["CAM01"] == 1)
check("a single-camera journey is not an OD pair",
      "CAM09" not in build_report([sight("CAM01", 0)], [journey(["CAM09"])], []).od_matrix)

# ---------------------------------------------------------------------------
print("\nalert rules")

engine = AlertEngine(congestion_index=1.5, min_trips_for_congestion=3,
                     speed_limits={"CAM01->CAM02": 40.0})

slow = build_report([sight("CAM01", 0)], [],
                    [hop("CAM01", "CAM02", s) for s in
                     (200, 205, 210, 700, 740, 780)])
alerts = engine.congestion(slow)
check("a genuinely slow corridor raises congestion", len(alerts) == 1)
check("the alert states the measured value and the threshold",
      alerts[0].value > 1.5 and alerts[0].threshold == 1.5)
check("severity escalates past twice free flow", alerts[0].severity == "critical")
check("the detail explains where the baseline came from",
      "measured" in alerts[0].detail or "configured" in alerts[0].detail)

two_trips = build_report([sight("CAM01", 0)], [],
                         [hop("CAM01", "CAM02", 200), hop("CAM01", "CAM02", 900)],
                         typical_speeds={"CAM01->CAM02": 26.0})
check("congestion is NOT called from two trips", engine.congestion(two_trips) == [])

fast = hop("CAM01", "CAM02", 60, km=1.6, speed=96.0)
alerts = engine.overspeed([fast])
check("over the legal limit raises an alert", len(alerts) == 1)
check("over-speed is flagged for review, not asserted", alerts[0].needs_review)
check("the detail admits it is a segment average, not a radar reading",
      "average" in alerts[0].detail.lower())
check("under the limit raises nothing",
      engine.overspeed([hop("CAM01", "CAM02", 240, speed=24.0)]) == [])
check("a corridor with no configured limit is not policed",
      engine.overspeed([hop("CAM02", "CAM03", 60, speed=200.0)]) == [])

# ---------------------------------------------------------------------------
print("\nwatchlist -- flagged, never asserted")

wl = Watchlist(entries=[WatchlistEntry("KA01AB1234", "stolen", "critical")],
               min_similarity=0.85)
engine = AlertEngine(watchlist=wl)

hits = engine.watchlist_hits([sight("CAM01", 0, plate="KA01AB1234")])
check("an exact plate matches", len(hits) == 1)
check("it carries the configured severity", hits[0].severity == "critical")
check("it is marked for human review", hits[0].needs_review)
check("the message says 'possible'", "possible" in hits[0].message.lower())
check("the detail says it is not confirmed", "not confirmed" in hits[0].detail.lower())

near = engine.watchlist_hits([sight("CAM01", 0, plate="KA0IAB1Z34")])
check("an OCR near-miss still matches -- exact matching would miss real hits",
      len(near) == 1, "")
check("and the alert reports the similarity that produced it",
      0.85 <= near[0].confidence < 1.0, str(near[0].confidence))
check("and names the raw read alongside the watchlist plate",
      "read as" in near[0].message, near[0].message)

check("a different plate raises nothing",
      engine.watchlist_hits([sight("CAM01", 0, plate="TN09BX2255")]) == [])
check("a vehicle with no plate raises nothing",
      engine.watchlist_hits([sight("CAM01", 0, plate=None)]) == [])
check("the same vehicle at the same camera alerts once",
      len(engine.watchlist_hits([sight("CAM01", 0), sight("CAM01", 1)])) == 1)
check("an empty watchlist is inert",
      AlertEngine().watchlist_hits([sight("CAM01", 0)]) == [])

# ---------------------------------------------------------------------------
print("\ncoverage gaps")

engine = AlertEngine(coverage_gap_count=2)
gaps = engine.coverage_gaps([journey(["CAM01", "CAM03"], ["CAM02"]),
                             journey(["CAM01", "CAM03"], ["CAM02"])])
check("a camera skipped repeatedly is reported", len(gaps) == 1
      and gaps[0].subject == "CAM02")
check("the message says how many it missed", "2 vehicles" in gaps[0].message)
check("the detail points at the camera, not the traffic",
      "detection" in gaps[0].detail or "readability" in gaps[0].detail)
check("one skip is not yet a pattern",
      engine.coverage_gaps([journey(["CAM01", "CAM03"], ["CAM02"])]) == [])
check("gaps stored as a string are parsed too",
      len(engine.coverage_gaps([SimpleNamespace(route=[], coverage_gaps="CAM02"),
                                SimpleNamespace(route=[], coverage_gaps="CAM02")])) == 1)

# ---------------------------------------------------------------------------
print("\nordering and empty state")

engine = AlertEngine(congestion_index=1.5, min_trips_for_congestion=3,
                     coverage_gap_count=2, speed_limits={"CAM01->CAM02": 40.0},
                     watchlist=wl)
alerts = engine.evaluate(slow, [journey(["CAM01", "CAM03"], ["CAM02"])] * 2,
                         [hop("CAM01", "CAM02", 60, speed=96.0)],
                         [sight("CAM01", 0, plate="KA01AB1234")])
check("all rules run together", len({a.kind for a in alerts}) == 4, str(alerts))
check("critical alerts sort first", alerts[0].severity == "critical")

empty = build_report([], [], [])
check("no data produces an empty report, not a crash", empty.total_sightings == 0)
check("and no alerts", AlertEngine().evaluate(empty, [], [], []) == [])
check("hotspots are empty when nothing is congested", empty.hotspots == [])
check("hotspots rank delay by how many vehicles it affects",
      slow.hotspots[0].label == "CAM01->CAM02")

# ---------------------------------------------------------------------------
print()
if _failed:
    print(f"{_passed} passed, {len(_failed)} FAILED: {', '.join(_failed)}")
    raise SystemExit(1)
print(f"{_passed} passed, 0 failed")
