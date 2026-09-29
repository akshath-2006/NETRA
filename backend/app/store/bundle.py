"""Moving a finished run between machines.

THE PROBLEM THIS SOLVES. NETRA already separates the expensive half from the
visible half: ``run``, ``resolve`` and ``analytics`` write to ``data/netra.db``
and ``data/evidence/``; ``serve`` and the dashboard only ever read them. So
processing can happen anywhere -- a GPU box, a borrowed workstation, a
notebook in someone's cloud -- without touching the architecture at all. What
was missing was the last, unglamorous step: getting the results back.

Copying ``netra.db`` by hand is not that step, for three reasons:

* **WAL.** The database runs in write-ahead-log mode. The ``.db`` file alone
  is stale; the recent writes live in ``-wal``. Copy the one file and you
  silently lose the tail of the run -- the worst kind of data loss, because
  everything still opens and looks fine.
* **Evidence.** Snapshots are files on disk, not blobs in the database. A
  database without them opens to a dashboard full of missing images.
* **It overwrites.** You usually want the remote run ADDED to what you already
  have, not swapped for it.

So: ``export`` writes one self-describing archive, ``import`` merges it in.
Import is idempotent -- a run already present is skipped, not duplicated --
because the realistic failure is someone importing the same bundle twice at
1 a.m. and wondering why every vehicle now appears twice.

Nothing here is NETRA-specific cleverness. It is the boring part that makes
remote execution actually usable.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: Import order, and which columns point at which table. Table-driven so that
#: adding a column to a model needs no change here at all -- only a NEW TABLE,
#: or a new foreign key, does.
TABLE_ORDER: tuple[tuple[str, dict[str, str]], ...] = (
    ("cameras", {}),
    ("vehicle_identities", {}),
    ("sightings", {"identity_id": "vehicle_identities"}),
    ("plate_reads", {"sighting_id": "sightings"}),
    ("identity_links", {"identity_id": "vehicle_identities",
                        "from_sighting_id": "sightings",
                        "to_sighting_id": "sightings"}),
    ("journeys", {"identity_id": "vehicle_identities"}),
    ("journey_hops", {"journey_id": "journeys"}),
    ("alerts", {}),
    ("lead_reviews", {"link_id": "identity_links"}),
)

MANIFEST = "manifest.json"
DB_MEMBER = "netra.db"
EVIDENCE_MEMBER = "evidence"
CONFIG_MEMBER = "configs"

BUNDLE_FORMAT = 1


class BundleError(Exception):
    """Anything that makes a bundle unusable."""


@dataclass
class ImportReport:
    runs_imported: list[str] = field(default_factory=list)
    runs_skipped: list[str] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    skipped_rows: dict[str, int] = field(default_factory=dict)
    evidence_copied: int = 0
    evidence_skipped: int = 0
    camera_conflicts: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def snapshot_database(src: Path, dst: Path) -> None:
    """A consistent single-file copy of a live WAL database.

    ``VACUUM INTO`` does the whole job: it checkpoints, it copies, and it
    leaves no -wal or -shm beside the result. The ``backup`` fallback exists
    for SQLite older than 3.27, which some system Pythons still ship.
    """
    src_con = _connect(src, read_only=True)
    try:
        try:
            src_con.execute("VACUUM INTO ?", (str(dst),))
        except sqlite3.OperationalError:
            dst_con = sqlite3.connect(str(dst))
            try:
                src_con.backup(dst_con)
            finally:
                dst_con.close()
    finally:
        src_con.close()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_bundle(*, database: Path, data_dir: Path, configs_dir: Path | None,
                  out: Path, include_evidence: bool = True,
                  note: str = "") -> dict:
    """Write one archive containing everything a dashboard needs to show a run.

    The whole database goes, never a subset of runs. Identities, links and
    journeys are computed ACROSS runs, so a per-run export would ship links
    whose other end is missing -- and a journey assembled from a link with one
    end missing is a fabricated journey. Import merges by run and skips
    duplicates, so exporting everything costs nothing but bytes.
    """
    if not database.exists():
        raise BundleError(f"no database at {database} -- nothing to export")

    out = out.expanduser()
    if out.is_dir():
        out = out / f"netra-{datetime.now():%Y%m%d-%H%M%S}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="netra-export-"))
    try:
        db_copy = tmp / DB_MEMBER
        snapshot_database(database, db_copy)

        con = _connect(db_copy, read_only=True)
        try:
            present = _tables(con)
            counts = {}
            for table, _ in TABLE_ORDER:
                if table in present:
                    counts[table] = con.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            run_ids = [r[0] for r in con.execute(
                "SELECT DISTINCT run_id FROM sightings ORDER BY run_id")] \
                if "sightings" in present else []
            span = con.execute(
                "SELECT MIN(first_seen), MAX(last_seen) FROM sightings").fetchone() \
                if "sightings" in present else (None, None)
        finally:
            con.close()

        evidence_src = data_dir / "evidence"
        evidence_files = 0
        if include_evidence and evidence_src.is_dir():
            evidence_files = sum(1 for p in evidence_src.rglob("*") if p.is_file())

        manifest = {
            "format": BUNDLE_FORMAT,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "runs": run_ids,
            "row_counts": counts,
            "evidence_files": evidence_files,
            "observation_span": {"first": span[0], "last": span[1]} if span else {},
            "produced_by": _host_note(),
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        with tarfile.open(out, "w:gz") as tar:
            tar.add(tmp / MANIFEST, arcname=MANIFEST)
            tar.add(db_copy, arcname=DB_MEMBER)
            if include_evidence and evidence_src.is_dir():
                tar.add(evidence_src, arcname=EVIDENCE_MEMBER)
            if configs_dir and configs_dir.is_dir():
                for cfgfile in sorted(configs_dir.glob("*.yaml")):
                    tar.add(cfgfile, arcname=f"{CONFIG_MEMBER}/{cfgfile.name}")

        manifest["bundle"] = str(out)
        manifest["bundle_bytes"] = out.stat().st_size
        return manifest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _host_note() -> dict:
    """Where this ran, so a bundle can say what produced it."""
    import platform
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = f"cuda:{torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["device"] = "mps"
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract without letting an archive write outside ``dest``.

    A tar member may be named ``../../etc/something`` or be a symlink pointing
    anywhere. We are extracting an archive a user may have downloaded from a
    machine they do not control, so each member is checked before it lands.
    """
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest) + "/") and target != dest:
            raise BundleError(f"refusing unsafe path in bundle: {member.name!r}")
        if member.issym() or member.islnk():
            raise BundleError(f"refusing link member in bundle: {member.name!r}")
    tar.extractall(dest)


def read_manifest(bundle: Path) -> dict:
    with tarfile.open(bundle, "r:*") as tar:
        try:
            f = tar.extractfile(MANIFEST)
        except KeyError:
            f = None
        if f is None:
            raise BundleError(f"{bundle.name} has no {MANIFEST} -- not a NETRA bundle")
        return json.loads(f.read().decode("utf-8"))


def import_bundle(*, bundle: Path, database: Path, data_dir: Path,
                  mode: str = "merge", dry_run: bool = False) -> ImportReport:
    """Bring a remote run into this machine's database and evidence store."""
    bundle = bundle.expanduser()
    if not bundle.exists():
        raise BundleError(f"no bundle at {bundle}")
    if mode not in ("merge", "replace"):
        raise BundleError(f"unknown mode {mode!r} -- use merge or replace")

    manifest = read_manifest(bundle)
    if int(manifest.get("format", 0)) > BUNDLE_FORMAT:
        raise BundleError(
            f"bundle format {manifest.get('format')} is newer than this build "
            f"understands ({BUNDLE_FORMAT}) -- update NETRA on this machine")

    tmp = Path(tempfile.mkdtemp(prefix="netra-import-"))
    try:
        with tarfile.open(bundle, "r:*") as tar:
            _safe_extract(tar, tmp)

        src_db = tmp / DB_MEMBER
        if not src_db.exists():
            raise BundleError(f"{bundle.name} contains no {DB_MEMBER}")

        report = ImportReport(dry_run=dry_run)

        if mode == "replace":
            return _import_replace(src_db, tmp, database, data_dir, manifest,
                                   report, dry_run)
        return _import_merge(src_db, tmp, database, data_dir, report, dry_run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _import_replace(src_db: Path, extracted: Path, database: Path, data_dir: Path,
                    manifest: dict, report: ImportReport, dry_run: bool) -> ImportReport:
    """Swap the local database for the bundle's, keeping a dated backup.

    The backup is not optional politeness. Replace is the mode someone reaches
    for at 2 a.m., and it is the one that can lose a day's work.
    """
    src_con = _connect(src_db, read_only=True)
    try:
        runs = [r[0] for r in src_con.execute("SELECT DISTINCT run_id FROM sightings")]
        for table, _ in TABLE_ORDER:
            if table in _tables(src_con):
                report.rows[table] = src_con.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        src_con.close()
    report.runs_imported = runs

    if not dry_run:
        database.parent.mkdir(parents=True, exist_ok=True)
        if database.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = database.with_name(f"{database.stem}.{stamp}.backup{database.suffix}")
            # NOT shutil.copy2. The database is in WAL mode, so the .db file on
            # its own can be stale -- or, on a database that has never
            # checkpointed, effectively empty. Copying it would produce a
            # backup that opens cleanly and contains nothing, which is the
            # worst possible outcome for the one safety net in this function.
            # snapshot_database() checkpoints as it copies.
            snapshot_database(database, backup)
            for side in ("-wal", "-shm"):                 # leave no stale WAL behind
                p = Path(str(database) + side)
                if p.exists():
                    p.unlink()
        shutil.copy2(src_db, database)
        report.evidence_copied, report.evidence_skipped = _copy_evidence(
            extracted / EVIDENCE_MEMBER, data_dir / "evidence", overwrite=True)
    return report


def _import_merge(src_db: Path, extracted: Path, database: Path, data_dir: Path,
                  report: ImportReport, dry_run: bool) -> ImportReport:
    from app.store.db import init_db

    init_db(database)                       # create the schema if this is a fresh machine

    src = _connect(src_db, read_only=True)
    dst = _connect(database)
    try:
        dst.execute("PRAGMA foreign_keys = ON")
        src_tables, dst_tables = _tables(src), _tables(dst)

        have_runs = {r[0] for r in dst.execute("SELECT DISTINCT run_id FROM sightings")}
        bundle_runs = [r[0] for r in src.execute(
            "SELECT DISTINCT run_id FROM sightings ORDER BY run_id")]
        report.runs_imported = [r for r in bundle_runs if r not in have_runs]
        report.runs_skipped = [r for r in bundle_runs if r in have_runs]

        if not report.runs_imported:
            return report                    # already here; importing again would duplicate

        wanted_runs = set(report.runs_imported)
        # old id -> new id, per table. Everything downstream is rewritten through it.
        idmap: dict[str, dict[int, int]] = {t: {} for t, _ in TABLE_ORDER}

        for table, fks in TABLE_ORDER:
            if table not in src_tables or table not in dst_tables:
                continue
            common = [c for c in _columns(src, table) if c in set(_columns(dst, table))]
            if not common:
                continue
            inserted = skipped = 0

            for row in src.execute(f"SELECT * FROM {table}"):
                data = {c: row[c] for c in common}

                keep, data, skip_reason = _prepare_row(
                    table, data, fks, idmap, wanted_runs, src, dst)
                if not keep:
                    skipped += 1
                    continue

                if table == "cameras":
                    existing = dst.execute("SELECT lat, lon FROM cameras WHERE id=?",
                                           (data["id"],)).fetchone()
                    if existing:
                        if (abs((existing["lat"] or 0) - (data.get("lat") or 0)) > 1e-6
                                or abs((existing["lon"] or 0) - (data.get("lon") or 0)) > 1e-6):
                            report.camera_conflicts.append(str(data["id"]))
                        skipped += 1
                        continue

                old_id = data.get("id")
                if table != "cameras":
                    data.pop("id", None)     # let the local database assign a fresh one

                cols = list(data)
                sql = (f"INSERT INTO {table} ({', '.join(cols)}) "
                       f"VALUES ({', '.join('?' for _ in cols)})")
                cur = dst.execute(sql, [data[c] for c in cols])
                if table != "cameras" and isinstance(old_id, int):
                    idmap[table][old_id] = cur.lastrowid
                inserted += 1

            if inserted:
                report.rows[table] = inserted
            if skipped:
                report.skipped_rows[table] = skipped

        if dry_run:
            dst.rollback()
        else:
            dst.commit()
    finally:
        src.close()
        dst.close()

    if not dry_run:
        report.evidence_copied, report.evidence_skipped = _copy_evidence(
            extracted / EVIDENCE_MEMBER, data_dir / "evidence", overwrite=False)
    return report


def _prepare_row(table: str, data: dict, fks: dict[str, str],
                 idmap: dict[str, dict[int, int]], wanted_runs: set[str],
                 src: sqlite3.Connection, dst: sqlite3.Connection):
    """Decide whether a row comes across, and rewrite its foreign keys.

    A row is dropped rather than repointed whenever the thing it references
    did not come across. That is the rule that keeps a half-imported bundle
    honest: no link is allowed to survive with one end missing, because a
    journey assembled from a dangling link is a fabricated journey.
    """
    # Sightings are the anchor: everything else follows from which runs came in.
    if table == "sightings":
        if data.get("run_id") not in wanted_runs:
            return False, data, "run already present"

    # Identities, links and journeys only make sense with their sightings.
    if table == "vehicle_identities":
        old = data.get("id")
        referenced = src.execute(
            "SELECT 1 FROM sightings WHERE identity_id=? AND run_id IN "
            f"({','.join('?' * len(wanted_runs))}) LIMIT 1",
            [old, *sorted(wanted_runs)]).fetchone()
        if not referenced:
            return False, data, "no imported sighting uses this identity"

    for column, target in fks.items():
        old = data.get(column)
        if old is None:
            continue
        new = idmap.get(target, {}).get(old)
        if new is None:
            return False, data, f"{column} points at a row that was not imported"
        data[column] = new

    # Alerts carry no foreign key, so the only protection against importing the
    # same bundle's alerts twice is their own content.
    if table == "alerts":
        dup = dst.execute(
            "SELECT 1 FROM alerts WHERE kind=? AND subject=? AND message=? "
            "AND IFNULL(at,'')=IFNULL(?,'') LIMIT 1",
            (data.get("kind"), data.get("subject"), data.get("message"),
             data.get("at"))).fetchone()
        if dup:
            return False, data, "identical alert already present"

    return True, data, ""


def _copy_evidence(src_dir: Path, dst_dir: Path, *, overwrite: bool) -> tuple[int, int]:
    """Copy snapshot JPEGs across.

    Filenames are ``<run_id>_<camera>_<track>.jpg`` -- derived from the
    sighting, not stored in it -- so they survive primary-key remapping
    untouched. That is why the merge above never has to rename an image.
    """
    if not src_dir.is_dir():
        return 0, 0
    copied = skipped = 0
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        target = dst_dir / p.relative_to(src_dir)
        if target.exists() and not overwrite:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied += 1
    return copied, skipped
