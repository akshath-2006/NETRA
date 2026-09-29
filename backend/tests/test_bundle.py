"""Tests for moving a finished run between machines.

    PYTHONPATH=backend python backend/tests/test_bundle.py

No video, no models, no network. Two throwaway databases stand in for the
remote box and the laptop.

WHAT THESE ARE REALLY GUARDING. Every table except ``cameras`` has an
autoincrement primary key, so a row's id on the remote machine is meaningless
on the laptop. Import therefore renumbers everything and rewrites every
foreign key as it goes. Get that wrong and nothing crashes -- you get a
database that opens fine and quietly shows the wrong vehicle at the wrong
camera. So most of what follows checks meaning rather than counts: that a link
still joins the same two sightings, that a journey still owns its own hops,
that an OCR read is still attached to the vehicle it was read from.
"""

from __future__ import annotations

import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.store.bundle import (BundleError, export_bundle,                # noqa: E402
                              import_bundle, read_manifest)
from app.store.db import get_sessionmaker, init_db                       # noqa: E402
from app.store.models import (Alert, Camera, IdentityLink, Journey,      # noqa: E402
                              JourneyHop, PlateRead, Sighting,
                              VehicleIdentity)

T0 = datetime(2026, 9, 24, 9, 0, 0)
_passed, _failed = 0, []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))


def build_db(root: Path, run_id: str, *, plate: str, cams=("CAM01", "CAM02")) -> Path:
    """One complete run: cameras, sightings, reads, an identity, a link, a journey."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    db = root / "data" / "netra.db"
    init_db(db)
    S = get_sessionmaker(db)
    with S() as s:
        for i, cam in enumerate(cams):
            s.add(Camera(id=cam, name=f"Test {cam}", lat=51.61 + i * 0.01,
                         lon=-0.13 + i * 0.01, heading_deg=90, enabled=True))
        s.commit()

    with S() as s:
        ident = VehicleIdentity(key=f"k-{run_id}", plate=plate, plate_confidence=0.9,
                                confidence=0.81, sighting_count=2, camera_count=2,
                                cameras=">".join(cams), first_seen=T0,
                                last_seen=T0 + timedelta(seconds=240))
        s.add(ident); s.flush()

        rows = []
        for i, cam in enumerate(cams):
            row = Sighting(run_id=run_id, camera_id=cam, track_id=i + 1,
                           vehicle_class="car", class_confidence=0.9,
                           detection_confidence=0.88,
                           first_seen=T0 + timedelta(seconds=240 * i),
                           last_seen=T0 + timedelta(seconds=240 * i + 8),
                           duration_s=8.0, frames=40, plate_text=plate,
                           plate_confidence=0.9, identity_id=ident.id,
                           box_x1=10, box_y1=10, box_x2=110, box_y2=90)
            s.add(row); s.flush()
            rows.append(row)
            for k in range(3):
                s.add(PlateRead(sighting_id=row.id, text=plate,
                                confidence=0.8 + k / 100, frame_index=10 + k,
                                at=T0, plate_width_px=180 + k))

        link = IdentityLink(identity_id=ident.id, from_sighting_id=rows[0].id,
                            to_sighting_id=rows[1].id, from_camera=cams[0],
                            to_camera=cams[1], gap_s=240.0, distance_km=1.6,
                            implied_speed_kmph=24.0, plate_similarity=1.0,
                            appearance_similarity=0.7, topology_score=0.9,
                            score=0.91, accepted=True, reason="plate identical",
                            created_at=T0)
        s.add(link); s.flush()

        j = Journey(identity_id=ident.id, identity_key=ident.key, plate=plate,
                    plate_confidence=0.9, confidence=0.81,
                    route=">".join(cams), full_route=">".join(cams),
                    hop_count=1, total_distance_km=1.6, total_duration_s=240.0,
                    average_speed_kmph=24.0, started_at=T0,
                    ended_at=T0 + timedelta(seconds=248))
        s.add(j); s.flush()
        s.add(JourneyHop(journey_id=j.id, sequence=0, from_camera=cams[0],
                         to_camera=cams[1], departed_at=T0,
                         arrived_at=T0 + timedelta(seconds=240), gap_s=240.0,
                         distance_km=1.6, implied_speed_kmph=24.0,
                         typical_speed_kmph=30.0, delay_ratio=1.25, confidence=0.8))
        s.add(Alert(kind="congestion", severity="warning", subject=f"{cams[0]}>{cams[1]}",
                    message=f"slow on {cams[0]}", detail="", at=T0, value=1.8,
                    threshold=1.5, confidence=1.0))
        s.commit()

    snaps = root / "data" / "evidence" / "sightings"
    snaps.mkdir(parents=True, exist_ok=True)
    for i, cam in enumerate(cams):
        (snaps / f"{run_id}_{cam}_{i + 1}.jpg").write_bytes(b"\xff\xd8\xff" + bytes(64))
    return db


def meaning(db: Path) -> dict:
    """Fingerprint the RELATIONSHIPS, not the ids -- ids are expected to change."""
    c = sqlite3.connect(db)
    out = {
        "integrity": c.execute("PRAGMA integrity_check").fetchone()[0],
        "fk_violations": len(c.execute("PRAGMA foreign_key_check").fetchall()),
        "links": sorted(c.execute(
            "SELECT a.run_id, a.camera_id, a.track_id, b.camera_id, b.track_id, "
            "       l.score, l.accepted, l.gap_s, l.distance_km "
            "FROM identity_links l JOIN sightings a ON a.id=l.from_sighting_id "
            "JOIN sightings b ON b.id=l.to_sighting_id").fetchall()),
        "journeys": sorted(c.execute(
            "SELECT j.identity_key, j.plate, j.route, j.hop_count, COUNT(h.id) "
            "FROM journeys j LEFT JOIN journey_hops h ON h.journey_id=j.id "
            "GROUP BY j.id").fetchall()),
        "identities": sorted(c.execute(
            "SELECT v.key, v.plate, v.confidence, COUNT(s.id) "
            "FROM vehicle_identities v LEFT JOIN sightings s ON s.identity_id=v.id "
            "GROUP BY v.id").fetchall()),
        "reads": sorted(c.execute(
            "SELECT s.camera_id, s.track_id, p.text, p.confidence, p.frame_index "
            "FROM plate_reads p JOIN sightings s ON s.id=p.sighting_id").fetchall()),
        "alerts": sorted(c.execute(
            "SELECT kind, subject, message FROM alerts").fetchall()),
    }
    c.close()
    return out


tmp = Path(tempfile.mkdtemp())
remote = tmp / "remote"
laptop = tmp / "laptop"

print("\nfixtures")
remote_db = build_db(remote, "remoterun01", plate="KA01AB1234")
build_db(laptop, "laptoprun9", plate="MH12DE1433", cams=("CAM03", "CAM04"))
print("  a remote run and an unrelated local run, on separate databases")

# ---------------------------------------------------------------------------
print("\nexport")
manifest = export_bundle(database=remote_db, data_dir=remote / "data",
                         configs_dir=None, out=tmp / "run.tar.gz",
                         note="unit test")
bundle = Path(manifest["bundle"])
check("the archive is written", bundle.exists())
check("the manifest names the run", manifest["runs"] == ["remoterun01"],
      str(manifest["runs"]))
check("the manifest counts the rows", manifest["row_counts"]["sightings"] == 2
      and manifest["row_counts"]["plate_reads"] == 6, str(manifest["row_counts"]))
check("the manifest counts the evidence", manifest["evidence_files"] == 2,
      str(manifest["evidence_files"]))
check("the manifest records where it ran",
      "platform" in manifest["produced_by"] and "device" in manifest["produced_by"])
check("the manifest is readable back out of the archive",
      read_manifest(bundle)["note"] == "unit test")

# Export must survive a database that is mid-WAL, which is the normal state.
with sqlite3.connect(remote_db) as live:
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("INSERT INTO alerts (kind, severity, subject, message, detail, "
                 "value, threshold, confidence, needs_review, created_at) "
                 "VALUES ('watchlist','info','x','late write','',0,0,1,0,?)",
                 (T0.isoformat(),))
    live.commit()
wal_bundle = Path(export_bundle(database=remote_db, data_dir=remote / "data",
                                configs_dir=None, out=tmp / "wal.tar.gz")["bundle"])
with tarfile.open(wal_bundle) as t:
    t.extract("netra.db", tmp / "walcheck")
walc = sqlite3.connect(tmp / "walcheck" / "netra.db")
check("a write still in the WAL is captured, not lost",
      walc.execute("SELECT COUNT(*) FROM alerts WHERE message='late write'"
                   ).fetchone()[0] == 1)
walc.close()

# ---------------------------------------------------------------------------
print("\nimport into a machine that already has different results")
before = meaning(laptop / "data" / "netra.db")
report = import_bundle(bundle=bundle, database=laptop / "data" / "netra.db",
                       data_dir=laptop / "data", mode="merge", dry_run=True)
check("a dry run reports what it would do", report.runs_imported == ["remoterun01"])
check("a dry run writes nothing",
      meaning(laptop / "data" / "netra.db") == before)

report = import_bundle(bundle=bundle, database=laptop / "data" / "netra.db",
                       data_dir=laptop / "data", mode="merge")
after = meaning(laptop / "data" / "netra.db")
src = meaning(remote_db)

check("the remote run is imported", report.runs_imported == ["remoterun01"])
check("the database stays internally consistent", after["integrity"] == "ok")
check("no foreign key is left dangling", after["fk_violations"] == 0,
      str(after["fk_violations"]))
check("the local run is untouched",
      all(row in after["links"] for row in before["links"]))
check("every imported link still joins the SAME two sightings",
      all(row in after["links"] for row in src["links"]),
      f"{src['links']} not within {after['links']}")
check("every imported journey keeps its own hops",
      all(row in after["journeys"] for row in src["journeys"]))
check("the imported identity keeps its sighting count",
      all(row in after["identities"] for row in src["identities"]))
check("every OCR read stays attached to the vehicle it came from",
      all(row in after["reads"] for row in src["reads"]))
check("nothing from the local run is lost",
      len(after["reads"]) == len(before["reads"]) + len(src["reads"]),
      f"{len(after['reads'])} vs {len(before['reads'])}+{len(src['reads'])}")
check("evidence images come across", report.evidence_copied == 2,
      str(report.evidence_copied))
check("the snapshot lands where the API looks for it",
      (laptop / "data" / "evidence" / "sightings"
       / "remoterun01_CAM01_1.jpg").exists())

# ---------------------------------------------------------------------------
print("\nimporting the same bundle twice")
snapshot = meaning(laptop / "data" / "netra.db")
again = import_bundle(bundle=bundle, database=laptop / "data" / "netra.db",
                      data_dir=laptop / "data", mode="merge")
check("the second import imports nothing", again.runs_imported == [])
check("it says which run it skipped", again.runs_skipped == ["remoterun01"])
check("no row is duplicated", meaning(laptop / "data" / "netra.db") == snapshot)

# ---------------------------------------------------------------------------
print("\nrefusals")
try:
    import_bundle(bundle=tmp / "nosuch.tar.gz", database=laptop / "data" / "netra.db",
                  data_dir=laptop / "data")
    ok = False
except BundleError:
    ok = True
check("a missing bundle is refused", ok)

notabundle = tmp / "notabundle.tar.gz"
with tarfile.open(notabundle, "w:gz") as t:
    p = tmp / "readme.txt"
    p.write_text("hello")
    t.add(p, arcname="readme.txt")
try:
    read_manifest(notabundle)
    ok = False
except BundleError:
    ok = True
check("an archive without a manifest is refused", ok)

# A tar member escaping the extraction directory is the classic archive attack,
# and a bundle is something a user downloads from a machine they may not own.
evil = tmp / "evil.tar.gz"
with tarfile.open(evil, "w:gz") as t:
    m = tmp / "m.json"
    m.write_text('{"format": 1, "runs": []}')
    t.add(m, arcname="manifest.json")
    t.add(m, arcname="../escaped.json")
try:
    import_bundle(bundle=evil, database=laptop / "data" / "netra.db",
                  data_dir=laptop / "data")
    ok = False
except BundleError:
    ok = True
check("a member with a path escaping the bundle is refused", ok)
check("nothing escaped", not (tmp.parent / "escaped.json").exists())

# ---------------------------------------------------------------------------
print("\nreplace mode keeps a way back")
repl_root = tmp / "replaceme"
repl_db = build_db(repl_root, "victim01", plate="KL07XY0001", cams=("CAM05", "CAM06"))
rep = import_bundle(bundle=bundle, database=repl_db, data_dir=repl_root / "data",
                    mode="replace")
backups = list(repl_db.parent.glob("netra.*.backup.db"))
check("replace makes a dated backup first", len(backups) == 1, str(backups))
check("the backup still holds the old run",
      sqlite3.connect(backups[0]).execute(
          "SELECT COUNT(*) FROM sightings WHERE run_id='victim01'").fetchone()[0] == 2)
check("the database now holds the bundle's run",
      sqlite3.connect(repl_db).execute(
          "SELECT COUNT(*) FROM sightings WHERE run_id='remoterun01'").fetchone()[0] == 2)
check("replace leaves no stale WAL beside the new database",
      not Path(str(repl_db) + "-wal").exists())
check("replace reports the runs it brought in", rep.runs_imported == ["remoterun01"])

print(f"\n{_passed} passed, {len(_failed)} FAILED")
if _failed:
    for n in _failed:
        print("   -", n)
    raise SystemExit(1)
