"""Tests for the event bus and the event store.

Runs against a throwaway SQLite file, so it needs no video, no model and no
network.

    PYTHONPATH=backend python backend/tests/test_store.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.bus import Event, LocalBus, Topics          # noqa: E402
from app.store.db import get_sessionmaker, init_db        # noqa: E402
from app.store.models import Sighting                     # noqa: E402
from app.store.repository import (add_plate_read, find_by_plate, recent_sightings,  # noqa: E402
                                  save_sighting, stats, upsert_cameras)
from app.store.subscriber import SightingWriter           # noqa: E402

T0 = datetime(2026, 9, 3, 10, 42, 0)
_passed, _failed = 0, []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def fake_track(track_id: int, camera_id="CAM01", label="car", offset=0.0, frames=40):
    """Duck-typed stand-in for vision.tracker.Track."""
    first = T0 + timedelta(seconds=offset)
    return SimpleNamespace(
        track_id=track_id, camera_id=camera_id, label=label,
        label_confidence=0.87, best_confidence=0.91,
        first_seen=first, last_seen=first + timedelta(seconds=3.2),
        duration_s=3.2, frames=frames, crossing="inbound",
        best_box=(10, 20, 110, 90),
        to_sighting=lambda: {"camera_id": camera_id, "track_id": track_id},
    )


def fake_camera(cid, lat=12.9, lon=77.6):
    return SimpleNamespace(id=cid, name=f"{cid} test", enabled=True, heading_deg=90.0,
                           location=SimpleNamespace(lat=lat, lon=lon))


# ---------------------------------------------------------------------------
print("\nevent bus")

bus = LocalBus("test")
got: list[Event] = []
bus.subscribe(Topics.SIGHTING_COMPLETED, got.append)
bus.publish(Topics.SIGHTING_COMPLETED, {"n": 1}, source="CAM01")
check("subscriber receives the event", len(got) == 1 and got[0].payload["n"] == 1)
check("source is carried", got[0].source == "CAM01")
check("publish count tracked", bus.published == 1)

bus.publish(Topics.PLATE_READ, {"n": 2})
check("other topics are not delivered to this subscriber", len(got) == 1)

second: list[Event] = []
bus.subscribe(Topics.SIGHTING_COMPLETED, second.append)
bus.publish(Topics.SIGHTING_COMPLETED, {"n": 3})
check("fan-out reaches every subscriber", len(got) == 2 and len(second) == 1)

def explode(_event):
    raise RuntimeError("subscriber blew up")

bus.subscribe(Topics.CROSSING, explode)
after: list[Event] = []
bus.subscribe(Topics.CROSSING, after.append)
bus.publish(Topics.CROSSING, {"n": 4})
check("a failing subscriber does not stop the others", len(after) == 1)

check("event serialises to json", '"topic"' in Event(Topics.CROSSING, {"a": 1}).to_json())

# ---------------------------------------------------------------------------
print("\nschema and cameras")

tmp = Path(tempfile.mkdtemp()) / "test.db"
init_db(tmp)
Session = get_sessionmaker(tmp)
check("database file created", tmp.exists())

with Session() as s:
    n = upsert_cameras(s, [fake_camera("CAM01"), fake_camera("CAM02")])
check("cameras mirrored from config", n == 2)

with Session() as s:
    upsert_cameras(s, [fake_camera("CAM01"), fake_camera("CAM02")])
    check("upsert is idempotent", stats(s)["cameras"] == 2)

with Session() as s:
    import sqlalchemy as sa
    mode = s.execute(sa.text("PRAGMA journal_mode")).scalar()
check("WAL mode is on (concurrent readers while workers write)",
      str(mode).lower() == "wal", str(mode))

# ---------------------------------------------------------------------------
print("\nsightings")

with Session() as s:
    row = save_sighting(s, "runA", fake_track(1))
check("sighting persisted", row is not None and row.id is not None)
check("fields round-trip", row.vehicle_class == "car" and row.frames == 40
      and row.crossing == "inbound" and row.box_x2 == 110)
check("virtual clock preserved", row.first_seen == T0)

with Session() as s:
    dup = save_sighting(s, "runA", fake_track(1))
check("same run + camera + track is not written twice", dup is None)

with Session() as s:
    other_run = save_sighting(s, "runB", fake_track(1))
check("a different run may reuse track id 1", other_run is not None)

with Session() as s:
    for i in range(2, 6):
        save_sighting(s, "runA", fake_track(i, camera_id="CAM02",
                                            label="truck" if i % 2 else "bus",
                                            offset=i * 10))
    total = stats(s)["sightings"]
check("multiple sightings accumulate", total == 6, f"total={total}")

with Session() as s:
    st = stats(s)
check("grouped by camera", st["by_camera"] == {"CAM01": 2, "CAM02": 4}, str(st["by_camera"]))
check("grouped by class", sum(st["by_class"].values()) == 6, str(st["by_class"]))
check("time window computed", st["first_seen"] == T0 and st["last_seen"] > T0)
check("plate columns start empty", st["with_plate"] == 0)

with Session() as s:
    rows = recent_sightings(s, limit=3)
check("recent sightings are newest first and limited",
      len(rows) == 3 and rows[0].first_seen >= rows[-1].first_seen)

with Session() as s:
    rows = recent_sightings(s, limit=50, camera_id="CAM02")
check("filter by camera", len(rows) == 4 and all(r.camera_id == "CAM02" for r in rows))

# ---------------------------------------------------------------------------
print("\nplate reads (the M5 contract)")

with Session() as s:
    sight = recent_sightings(s, limit=1)[0]
    for i, (text, conf) in enumerate([("KA01AB1234", 0.81), ("KA01AB1Z34", 0.42),
                                      ("KA01AB1234", 0.88)]):
        add_plate_read(s, sight.id, text, conf, frame_index=i, plate_width_px=96)
    sight.plate_text = "KA01AB1234"
    sight.plate_confidence = 0.88
    s.add(sight)
    s.commit()

with Session() as s:
    check("plate reads stored", stats(s)["plate_reads"] == 3)
    check("consensus plate denormalised onto the sighting", stats(s)["with_plate"] == 1)
    found = find_by_plate(s, "ka01ab1234")
check("plate lookup is case-insensitive on input", len(found) == 1)
check("plate lookup returns time-ordered rows", found[0].plate_text == "KA01AB1234")

with Session() as s:
    check("unknown plate returns nothing", find_by_plate(s, "XX00XX0000") == [])

# ---------------------------------------------------------------------------
print("\nbus -> store wiring")

tmp2 = Path(tempfile.mkdtemp()) / "wired.db"
init_db(tmp2)
S2 = get_sessionmaker(tmp2)
with S2() as s:
    upsert_cameras(s, [fake_camera("CAM01")])

bus2 = LocalBus("wired")
writer = SightingWriter(S2, "runC").register(bus2)
for i in range(1, 4):
    bus2.publish(Topics.SIGHTING_COMPLETED, {"track": fake_track(i)}, source="CAM01")
check("worker publishes, store subscribes, rows land", writer.written == 3)

bus2.publish(Topics.SIGHTING_COMPLETED, {"track": fake_track(1)}, source="CAM01")
check("replayed event is skipped, not duplicated",
      writer.written == 3 and writer.skipped == 1)

bus2.publish(Topics.SIGHTING_COMPLETED, {"nothing": True}, source="CAM01")
check("malformed payload is ignored, not fatal", writer.written == 3)

with S2() as s:
    check("store holds exactly what was published", stats(s)["sightings"] == 3)

# ---------------------------------------------------------------------------
print()
if _failed:
    print(f"{_passed} passed, {len(_failed)} FAILED: {', '.join(_failed)}")
    raise SystemExit(1)
print(f"{_passed} passed, 0 failed")


# ---------------------------------------------------------------------------
print("\nresolution is re-runnable (M7/M8 regression)")

from app.store.repository import save_journeys, save_resolution   # noqa: E402

tmp3 = Path(tempfile.mkdtemp()) / "resolve.db"
init_db(tmp3)
S3 = get_sessionmaker(tmp3)
with S3() as s:
    upsert_cameras(s, [fake_camera("CAM01"), fake_camera("CAM02")])
    for i in range(1, 5):
        save_sighting(s, "runR", fake_track(i, camera_id="CAM01" if i < 3 else "CAM02",
                                            offset=i * 60))


class _Ident:
    def __init__(self, key, ids, cams):
        self.key, self.sighting_ids, self.cameras = key, ids, cams
        self.plate, self.plate_confidence, self.confidence = "KA01AB1234", 0.8, 0.8
        self.first_seen, self.last_seen = T0, T0 + timedelta(minutes=4)


class _Verdict:
    def __init__(self, a, b):
        self.from_id, self.to_id, self.accepted, self.score = a, b, True, 0.8
        self.from_camera, self.to_camera = "CAM01", "CAM02"
        self.gap_s = self.distance_km = self.implied_speed_kmph = 1.0
        self.plate_similarity = self.appearance_similarity = self.topology_score = 1.0
        self.reason = "ok"


class _Hop:
    from_camera, to_camera = "CAM01", "CAM02"
    departed_at, arrived_at = T0, T0 + timedelta(minutes=4)
    gap_s, distance_km, implied_speed_kmph = 240.0, 1.6, 24.0
    typical_speed_kmph, delay_ratio, confidence = 26.0, 1.08, 0.8
    via, routed = [], True


class _Journey:
    identity_key, plate, plate_confidence, confidence = "V1", "KA01AB1234", 0.8, 0.8
    sighting_ids, route, full_route, coverage_gaps = [1, 3], ["CAM01", "CAM02"], ["CAM01", "CAM02"], []
    hops = [_Hop()]
    total_distance_km, total_duration_s, average_speed_kmph = 1.6, 240.0, 24.0
    started_at, ended_at = T0, T0 + timedelta(minutes=4)


def run_resolution(session):
    ident = _Ident("V1", [1, 3], ["CAM01", "CAM02"])
    summary = save_resolution(session, [ident], [_Verdict(1, 3)])
    from app.store.repository import list_identities
    keys = {r.key: r.id for r in list_identities(session, limit=100)}
    return summary, save_journeys(session, [_Journey()], keys)


with S3() as s:
    first_summary, first_journeys = run_resolution(s)
check("first resolution writes identities and journeys",
      first_summary["identities"] == 1 and first_journeys == 1)

try:
    with S3() as s:
        second_summary, second_journeys = run_resolution(s)
    rerun_ok = second_summary["identities"] == 1 and second_journeys == 1
    rerun_err = ""
except Exception as exc:
    rerun_ok, rerun_err = False, str(exc)[:90]
check("running resolve a SECOND time does not blow up on foreign keys",
      rerun_ok, rerun_err)

with S3() as s:
    check("re-running does not duplicate identities", stats(s)["sightings"] == 4)
    from app.store.models import Journey as JRow
    check("journeys are replaced, not appended", s.query(JRow).count() == 1)
