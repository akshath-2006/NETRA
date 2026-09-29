"""Tests for the 45% display rule, Unclear Possible Leads, and plate status.

Runs against a throwaway SQLite file. No video, no model, no network.

    PYTHONPATH=backend python backend/tests/test_leads.py

WHAT THESE TESTS ARE GUARDING
The display threshold is a PRESENTATION rule, not a retention rule. The whole
point is that low-confidence evidence survives and stays inspectable. A test
suite that only checked "weak things are hidden" would pass happily while the
system quietly deleted the evidence, so most of what is below checks that the
refused pairs are still there, still complete, and still reachable.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.store.db import get_sessionmaker, init_db                    # noqa: E402
from app.store.models import (Camera, IdentityLink, LeadReview,         # noqa: E402
                              Sighting)
from app.store.repository import (LEAD_STATUSES, lead_links,          # noqa: E402
                                  links_touching_sightings,
                                  plate_reads_for, set_lead_status,
                                  add_plate_read)
from app.vision.consensus import PlateReading, build_consensus        # noqa: E402

T0 = datetime(2026, 9, 20, 10, 42, 0)
_passed, _failed = 0, []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))


tmp = Path(tempfile.mkdtemp()) / "leads.db"
init_db(tmp)
S = get_sessionmaker(tmp)


def make_sighting(s, cam, track, plate, conf, offset_s, klass="car"):
    row = Sighting(
        run_id="testrun", camera_id=cam, track_id=track, vehicle_class=klass,
        class_confidence=0.9, detection_confidence=0.88,
        first_seen=T0 + timedelta(seconds=offset_s),
        last_seen=T0 + timedelta(seconds=offset_s + 8),
        duration_s=8.0, frames=40, plate_text=plate, plate_confidence=conf,
        box_x1=10, box_y1=10, box_x2=110, box_y2=90)
    s.add(row); s.flush()
    return row


def make_link(s, a, b, *, score, accepted, plate_sim=0.8, appear=0.6, topo=0.9,
              gap=240.0, dist=1.6, speed=24.0, reason=""):
    row = IdentityLink(
        from_sighting_id=a.id, to_sighting_id=b.id,
        from_camera=a.camera_id, to_camera=b.camera_id,
        gap_s=gap, distance_km=dist, implied_speed_kmph=speed,
        plate_similarity=plate_sim, appearance_similarity=appear,
        topology_score=topo, score=score, accepted=accepted, reason=reason,
        created_at=T0)
    s.add(row); s.flush()
    return row


print("\nfixtures")
with S() as s:
    # sightings.camera_id is a real foreign key, so the cameras come first.
    for i, cam in enumerate(["CAM01", "CAM02", "CAM03", "CAM04", "CAM05"]):
        s.add(Camera(id=cam, name=f"Test {cam}", lat=12.97 + i * 0.01,
                     lon=77.60 + i * 0.01, heading_deg=90, enabled=True))
    s.commit()

with S() as s:
    # A confident, accepted pair -- must stay in the normal results.
    a1 = make_sighting(s, "CAM01", 1, "KA01AB1234", 0.95, 0)
    a2 = make_sighting(s, "CAM02", 1, "KA01AB1234", 0.93, 240)
    good = make_link(s, a1, a2, score=0.91, accepted=True,
                     reason="plate similarity 1.00, travel time consistent")

    # A refused near-miss -- the lead material.
    b1 = make_sighting(s, "CAM01", 2, "KA01AB1284", 0.52, 10, klass="bus")
    b2 = make_sighting(s, "CAM03", 2, "KA01A81234", 0.48, 130)
    weak = make_link(s, b1, b2, score=0.43, accepted=False, plate_sim=0.78,
                     appear=0.41, topo=0.22, gap=120.0, dist=4.0, speed=120.0,
                     reason="implies 120 km/h over 4.0 km")

    # A second refused pair, different camera and class, for filter tests.
    c1 = make_sighting(s, "CAM04", 3, "KA05XY9999", 0.44, 20, klass="truck")
    c2 = make_sighting(s, "CAM05", 3, "KA05XY9989", 0.41, 400)
    weak2 = make_link(s, c1, c2, score=0.31, accepted=False, plate_sim=0.72,
                      appear=0.30, topo=0.10, reason="plate similarity below floor")

    add_plate_read(s, b1.id, "KA01AB1284", 0.52, frame_index=12, plate_width_px=180)
    add_plate_read(s, b1.id, "KA01AB1234", 0.49, frame_index=15, plate_width_px=176)
    s.commit()
    good_id, weak_id, weak2_id = good.id, weak.id, weak2.id
    b1_id, b2_id, a1_id, a2_id = b1.id, b2.id, a1.id, a2.id
print("  built 6 sightings, 1 accepted link, 2 refused links")

# ---------------------------------------------------------------------------
print("\nthe 45% display rule")
THRESHOLD = 0.45
with S() as s:
    rows, total = lead_links(s, limit=50)
    ids = {l.id for l, _a, _b, _r in rows}

    check("a refused pair below the threshold becomes a lead", weak_id in ids)
    check("an accepted pair NEVER appears as a lead", good_id not in ids)
    check("leads are counted for pagination", total == 2, f"total={total}")

    # The rule is about presentation. The evidence must still be in the store.
    check("the low-confidence link is still persisted",
          s.get(IdentityLink, weak_id) is not None)
    check("both its sightings are still persisted",
          s.get(Sighting, b1_id) is not None and s.get(Sighting, b2_id) is not None)

    link = s.get(IdentityLink, weak_id)
    check("the lead keeps every contributing signal",
          link.plate_similarity == 0.78 and link.appearance_similarity == 0.41
          and link.topology_score == 0.22,
          f"{link.plate_similarity}/{link.appearance_similarity}/{link.topology_score}")
    check("the lead keeps the travel-time evidence",
          link.gap_s == 120.0 and link.distance_km == 4.0
          and link.implied_speed_kmph == 120.0)
    check("the lead keeps the written reason", "120 km/h" in link.reason, link.reason)
    check("a lead is never marked accepted", link.accepted is False)
    check("its score really is below the display threshold",
          link.score < THRESHOLD, f"{link.score}")
    check("the accepted link is above it", s.get(IdentityLink, good_id).score >= THRESHOLD)

# ---------------------------------------------------------------------------
print("\nevidence reachable from a lead")
with S() as s:
    reads = plate_reads_for(s, [b1_id])
    check("OCR reads behind a lead are retrievable", len(reads.get(b1_id, [])) == 2,
          str(len(reads.get(b1_id, []))))
    check("reads come back strongest first",
          reads[b1_id][0].confidence >= reads[b1_id][1].confidence)
    touching = links_touching_sightings(s, [a1_id, a2_id])
    check("a vehicle's judged pairs are retrievable for its case file",
          any(l.id == good_id for l in touching))

# ---------------------------------------------------------------------------
print("\nfiltering and review workflow")
with S() as s:
    rows, total = lead_links(s, camera="CAM04")
    check("filter by camera", total == 1 and rows[0][0].id == weak2_id, f"total={total}")

    rows, total = lead_links(s, plate="KA05")
    check("filter by plate fragment", total == 1, f"total={total}")

    rows, total = lead_links(s, vehicle_class="bus")
    check("filter by vehicle class", total == 1 and rows[0][0].id == weak_id)

    rows, total = lead_links(s, min_score=0.40)
    check("filter by minimum score", total == 1 and rows[0][0].id == weak_id,
          f"total={total}")

    rows, total = lead_links(s, status="unreviewed")
    check("everything starts unreviewed", total == 2, f"total={total}")

with S() as s:
    set_lead_status(s, weak_id, "potential_match", "same plate family, timing odd")
with S() as s:
    rows, total = lead_links(s, status="potential_match")
    check("a status change is recorded", total == 1 and rows[0][0].id == weak_id)
    check("the note is kept", rows[0][3].note.startswith("same plate family"))
    rows, total = lead_links(s, status="unreviewed")
    check("the reviewed lead leaves the unreviewed queue", total == 1, f"total={total}")

with S() as s:
    set_lead_status(s, weak_id, "escalated")
with S() as s:
    check("status can be changed again",
          s.query(LeadReview).filter(LeadReview.link_id == weak_id).one().status == "escalated")
    check("review rows are one per link, not appended",
          s.query(LeadReview).filter(LeadReview.link_id == weak_id).count() == 1)
    check("marking a lead never alters the evidence",
          s.get(IdentityLink, weak_id).score == 0.43)

with S() as s:
    try:
        set_lead_status(s, weak_id, "definitely_the_same")
        bad_ok = False
    except ValueError:
        bad_ok = True
    check("an unknown status is refused", bad_ok)
    check("the status vocabulary is fixed", set(LEAD_STATUSES) == {
        "unreviewed", "reviewed", "potential_match", "rejected", "escalated"})

# ---------------------------------------------------------------------------
print("\nplate consensus status thresholds")
strong = [PlateReading("KA01AB1234", 0.95, i, 190, []) for i in range(8)]
c = build_consensus(strong, confirm_confidence=0.60, tentative_confidence=0.40)
check("agreeing high-confidence reads are confirmed", c.status == "confirmed",
      f"{c.status} {c.confidence}")
check("a confirmed plate is shown", c.ok)

mixed = [PlateReading("KA01AB1234", 0.42, 0, 120, []),
         PlateReading("KA01A81234", 0.41, 1, 118, []),
         PlateReading("KA01AB1264", 0.40, 2, 121, [])]
c2 = build_consensus(mixed, confirm_confidence=0.60, tentative_confidence=0.40)
check("disagreeing weak reads are not confirmed", c2.status != "confirmed",
      f"{c2.status} {c2.confidence}")

junk = [PlateReading("1234", 0.31, 0, 60, []), PlateReading("B1234", 0.30, 1, 62, [])]
c3 = build_consensus(junk, confirm_confidence=0.60, tentative_confidence=0.40)
check("unreadable consensus reports no usable plate", not c3.ok or c3.status == "unreadable",
      f"{c3.status} {c3.confidence} {c3.text!r}")

# The misalignment bug: a fixed-length confidence array must not be indexed by
# text position, or every character is weighted by the wrong slot.
misaligned = [PlateReading("1234", 0.9, 0, 100, [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
              for _ in range(3)]
c4 = build_consensus(misaligned)
check("a confidence array that does not match the text falls back safely",
      c4.text == "1234", f"{c4.text!r}")

print(f"\n{_passed} passed, {len(_failed)} FAILED")
if _failed:
    for n in _failed:
        print("   -", n)
    raise SystemExit(1)
