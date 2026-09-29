"""Tests for the camera graph, appearance features and identity resolution.

The most important tests here are the ones that assert a REJECTION. Any system
can link things; the claim this project makes is that it refuses to link things
it cannot justify.

    PYTHONPATH=backend python backend/tests/test_identity.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.city_graph import CityGraph, GraphDefaults, haversine_km  # noqa: E402
from app.identity import appearance                                    # noqa: E402
from app.identity.resolver import IdentityResolver, SightingView       # noqa: E402

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
           cam("CAM03", 12.9604, 77.6472)]


def graph(**kw):
    g = CityGraph(CAMERAS, GraphDefaults(**kw))
    g.add_edge("CAM01", "CAM02", 1.6, typical_speed_kmph=26, max_speed_kmph=60)
    g.add_edge("CAM02", "CAM03", 3.1, typical_speed_kmph=32, max_speed_kmph=70)
    g.add_edge("CAM01", "CAM03", 4.4, typical_speed_kmph=30, max_speed_kmph=70)
    return g


def view(sid, camera, offset_s, plate=None, conf=0.85, klass="car",
         colour="3C6EBE", aspect=2.3, dwell=3.0):
    first = T0 + timedelta(seconds=offset_s)
    return SightingView(id=sid, camera_id=camera, first_seen=first,
                        last_seen=first + timedelta(seconds=dwell),
                        vehicle_class=klass, plate_text=plate,
                        plate_confidence=conf, colour_hex=colour, aspect=aspect)


# ---------------------------------------------------------------------------
print("\ncamera graph")

d = haversine_km(12.9756, 77.6068, 12.9731, 77.6195)
check("haversine gives a sane city distance", 1.0 < d < 2.0, f"{d:.2f} km")
check("distance is symmetric",
      abs(haversine_km(12.9756, 77.6068, 12.9604, 77.6472)
          - haversine_km(12.9604, 77.6472, 12.9756, 77.6068)) < 1e-9)

g = graph()
check("declared edge keeps its distance", g.edge("CAM01", "CAM02").distance_km == 1.6)
check("declared edges are bidirectional", g.edge("CAM02", "CAM01").distance_km == 1.6)
check("declared edges are not marked derived", not g.edge("CAM01", "CAM02").derived)

bare = CityGraph(CAMERAS, GraphDefaults())
e = bare.edge("CAM01", "CAM02")
check("missing edges are derived from coordinates", e is not None and e.derived)
check("derived distance exceeds straight line (detour factor)", e.distance_km > d)

off = CityGraph(CAMERAS, GraphDefaults(), auto_edges=False)
check("auto_edges off means no route", off.edge("CAM01", "CAM02") is None)
check("unknown camera has no route", g.edge("CAM01", "CAM99") is None)

# ---------------------------------------------------------------------------
print("\nthe physics gate")

ok, spd, why = g.check("CAM01", "CAM02", 220)
check("a normal journey is possible", ok and 20 < spd < 30, why)

ok, spd, why = g.check("CAM01", "CAM02", 5)
check("teleporting is rejected", not ok and spd > 1000, why)
check("rejection says why in plain english", "km/h" in why and "ceiling" in why)

ok, _, why = g.check("CAM01", "CAM02", 60)
check("above the corridor ceiling is rejected", not ok, why)

ok, _, why = g.check("CAM01", "CAM02", 4000)
check("a gap beyond the window is rejected", not ok and "window" in why, why)

ok, _, why = g.check("CAM01", "CAM01", 300)
check("same camera is not a cross-camera link", not ok and "same camera" in why)

ok, _, why = g.check("CAM01", "CAM02", -30)
check("arriving before departing is rejected", not ok)

ok, _, why = g.check("CAM01", "CAM02", 1750)
check("crawling for 29 minutes is rejected as one journey", not ok, why)

check("plausibility peaks at the typical speed",
      g.plausibility("CAM01", "CAM02", 222) > 0.99)
check("plausibility falls away from typical",
      g.plausibility("CAM01", "CAM02", 222) > g.plausibility("CAM01", "CAM02", 500))
check("an impossible journey has zero plausibility",
      g.plausibility("CAM01", "CAM02", 5) == 0.0)

# ---------------------------------------------------------------------------
print("\nappearance")

check("same class scores 1", appearance.class_similarity("car", "car") == 1.0)
check("car/truck is a plausible confusion",
      0 < appearance.class_similarity("car", "truck") < 1)
check("car/motorcycle is not", appearance.class_similarity("car", "motorcycle") == 0.0)
check("unknown class is neutral", appearance.class_similarity(None, "car") == 0.5)

check("identical colours score 1", appearance.colour_similarity("3C6EBE", "3C6EBE") == 1.0)
check("black vs white scores ~0", appearance.colour_similarity("000000", "FFFFFF") < 0.01)
check("near colours score high", appearance.colour_similarity("3C6EBE", "3E70C0") > 0.95)
check("missing colour is neutral", appearance.colour_similarity(None, "3C6EBE") == 0.5)
check("bgr converts to rgb hex", appearance.to_hex((190, 110, 60)) == "3C6EBE")

a = SimpleNamespace(vehicle_class="car", colour_hex="3C6EBE", aspect=2.3)
b = SimpleNamespace(vehicle_class="car", colour_hex="3E70C0", aspect=2.35)
c = SimpleNamespace(vehicle_class="motorcycle", colour_hex="C03030", aspect=1.1)
check("same vehicle scores high", appearance.similarity(a, b) > 0.9)
check("different vehicle scores low", appearance.similarity(a, c) < 0.2)

# ---------------------------------------------------------------------------
print("\nidentity resolution")

r = IdentityResolver(graph())

v = r.evaluate(view(1, "CAM01", 0, "KA01AB1234"), view(2, "CAM02", 240, "KA01AB1234"))
check("same plate, plausible time -> linked", v.accepted, v.reason)
check("link records the implied speed", 15 < v.implied_speed_kmph < 30)
check("link records the road distance", v.distance_km == 1.6)

# THE test: the plates agree, the physics does not.
v = r.evaluate(view(1, "CAM01", 0, "KA01AB1234"), view(2, "CAM03", 8, "KA01AB1234"))
check("IDENTICAL plates are rejected when the journey is impossible", not v.accepted)
check("the rejection still reports the plate agreement", v.plate_similarity == 1.0)
check("the reason names both facts",
      "plate similarity" in v.reason and "km/h" in v.reason, v.reason)
check("an impossible link scores zero", v.score == 0.0)

# An OCR near-miss must still link.
v = r.evaluate(view(1, "CAM01", 0, "MH12DE1433"), view(2, "CAM02", 240, "MH12OE1433"))
check("a one-character OCR near-miss still links", v.accepted, v.reason)
check("near-miss similarity stays high", v.plate_similarity > 0.9)

v = r.evaluate(view(1, "CAM01", 0, "KA01AB1234"), view(2, "CAM02", 240, "TN09BX2255"))
check("genuinely different plates do not link", not v.accepted)
check("and the reason says so", "differ" in v.reason, v.reason)

# Low-confidence plates are weaker evidence.
hi = r.evaluate(view(1, "CAM01", 0, "KA01AB1234", conf=0.95),
                view(2, "CAM02", 240, "KA01AB1234", conf=0.95))
lo = r.evaluate(view(3, "CAM01", 0, "KA01AB1234", conf=0.30),
                view(4, "CAM02", 240, "KA01AB1234", conf=0.30))
check("a doubted plate produces a weaker link", lo.score < hi.score,
      f"{lo.score:.2f} vs {hi.score:.2f}")

# No plate: suggestible, never confident.
v = r.evaluate(view(1, "CAM01", 0, None), view(2, "CAM02", 240, None))
check("a plateless link is capped below certainty", v.score <= r.plateless_cap)
check("and it is never accepted while require_plate is on", not v.accepted)

relaxed = IdentityResolver(graph(), require_plate=False, plateless_cap=0.70)
v2 = relaxed.evaluate(view(1, "CAM01", 0, None), view(2, "CAM02", 240, None))
check("relaxing require_plate is possible but must be deliberate",
      v2.accepted and "capped" in v2.reason, v2.reason)

v = r.evaluate(view(1, "CAM01", 0, None, klass="car"),
               view(2, "CAM02", 240, None, klass="motorcycle", colour="C03030"))
check("plateless link between unlike vehicles is refused", not v.accepted)

# The regression an end-to-end run found: identical-looking vehicles with no
# plate must NOT be merged. Appearance is corroboration, not identification.
v = r.evaluate(view(1, "CAM01", 0, None), view(2, "CAM02", 240, None))
check("identical-looking plateless vehicles are never linked", not v.accepted,
      f"score={v.score:.2f}")
check("the cap sits below the accept threshold -- otherwise it does nothing",
      r.plateless_cap < r.accept_threshold,
      f"cap={r.plateless_cap} threshold={r.accept_threshold}")
check("refusal explains that appearance is not identification",
      "not identification" in v.reason, v.reason)

lookalikes = [view(i, c, o, None) for i, (c, o) in
              enumerate([("CAM01", 0), ("CAM02", 240), ("CAM03", 640),
                         ("CAM01", 30), ("CAM02", 270), ("CAM03", 670)], start=1)]
ids, _ = r.resolve(lookalikes)
check("six identical plateless vehicles stay six identities",
      all(len(i.cameras) == 1 for i in ids), f"{len(ids)} identities")

# ---------------------------------------------------------------------------
print("\nclustering into journeys")

journey = [view(1, "CAM01", 0, "KA01AB1234"),
           view(2, "CAM02", 240, "KA01AB1234"),
           view(3, "CAM03", 640, "KA01AB1234")]
identities, verdicts = r.resolve(journey)
multi = [i for i in identities if len(i.cameras) > 1]
check("three sightings become one vehicle", len(multi) == 1, f"{len(multi)} journeys")
check("the route is in travel order", multi[0].cameras == ["CAM01", "CAM02", "CAM03"])
check("the canonical plate is carried", multi[0].plate == "KA01AB1234")
check("confidence is the weakest hop, not the average",
      abs(multi[0].confidence - min(multi[0].link_scores)) < 1e-9)

mixed = journey + [view(4, "CAM02", 300, "TN09BX2255"),
                   view(5, "CAM03", 700, "TN09BX2255")]
identities, _ = r.resolve(mixed)
multi = [i for i in identities if len(i.cameras) > 1]
check("two different vehicles stay two identities", len(multi) == 2, f"{len(multi)}")
check("neither vehicle absorbs the other",
      sorted(len(i.sighting_ids) for i in multi) == [2, 3])

# A valid plate should win over a garbage one with higher raw confidence.
messy = [view(1, "CAM01", 0, "KA01AB1234", conf=0.55),
         view(2, "CAM02", 240, "KA01AB1234", conf=0.90)]
identities, _ = r.resolve(messy)
check("canonical plate prefers the most confident valid read",
      identities[0].plate == "KA01AB1234")

identities, verdicts = r.resolve([view(1, "CAM01", 0, "KA01AB1234")])
check("a lone sighting is not turned into a journey",
      all(len(i.cameras) == 1 for i in identities))
check("no sightings resolves cleanly", r.resolve([]) == ([], []))

identities, verdicts = r.resolve([view(1, "CAM01", 0, "KA01AB1234"),
                                  view(2, "CAM03", 8, "KA01AB1234")])
rejected = [v for v in verdicts if not v.accepted]
check("impossible pairs are still recorded, for the audit trail", len(rejected) == 1)
check("recorded rejection keeps its reason", "km/h" in rejected[0].reason)

# ---------------------------------------------------------------------------
print()
if _failed:
    print(f"{_passed} passed, {len(_failed)} FAILED: {', '.join(_failed)}")
    raise SystemExit(1)
print(f"{_passed} passed, 0 failed")
