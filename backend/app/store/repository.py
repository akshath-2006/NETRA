"""Query and write helpers.

The rest of the application talks to this module, not to SQLAlchemy. Keeping
the queries in one place means the API layer in Milestone 6 and the analytics
in Milestone 9 share exactly one definition of "what counts as a sighting".
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from sqlalchemy import func, or_, select
from sqlalchemy import update as sa_update
from sqlalchemy.orm import Session, aliased

from app.core.logging import get_logger
from app.store.models import (Camera, IdentityLink, Journey, JourneyHop,
                              LeadReview, PlateRead, Sighting, VehicleIdentity)

log = get_logger("repo")


# --- cameras ---------------------------------------------------------------

def upsert_cameras(session: Session, cameras: Iterable) -> int:
    """Mirror cameras.yaml into the database. Idempotent."""
    count = 0
    for cam in cameras:
        row = session.get(Camera, cam.id)
        if row is None:
            row = Camera(id=cam.id)
            session.add(row)
        row.name = cam.name
        row.lat = cam.location.lat
        row.lon = cam.location.lon
        row.heading_deg = cam.heading_deg
        row.enabled = cam.enabled
        row.updated_at = datetime.now()
        count += 1
    session.commit()
    return count


def list_cameras(session: Session) -> Sequence[Camera]:
    return session.scalars(select(Camera).order_by(Camera.id)).all()


# --- sightings -------------------------------------------------------------

def save_sighting(session: Session, run_id: str, track) -> Sighting | None:
    """Persist one completed track. Returns None if it already exists.

    The uniqueness check is what makes re-processing a clip safe: run it twice
    with the same run_id and the vehicle count does not double.
    """
    existing = session.scalar(
        select(Sighting).where(
            Sighting.run_id == run_id,
            Sighting.camera_id == track.camera_id,
            Sighting.track_id == track.track_id,
        )
    )
    if existing is not None:
        return None

    box = track.best_box or (0, 0, 0, 0)
    plate = getattr(track, "plate", None)
    row = Sighting(
        run_id=run_id,
        camera_id=track.camera_id,
        track_id=track.track_id,
        vehicle_class=track.label,
        class_confidence=round(track.label_confidence, 4),
        detection_confidence=round(track.best_confidence, 4),
        first_seen=track.first_seen,
        last_seen=track.last_seen,
        duration_s=round(track.duration_s, 3),
        frames=track.frames,
        crossing=track.crossing,
        box_x1=int(box[0]), box_y1=int(box[1]),
        box_x2=int(box[2]), box_y2=int(box[3]),
        plate_text=(plate.text or None) if plate is not None and plate.ok else None,
        plate_confidence=plate.confidence if plate is not None and plate.ok else None,
        colour_hex=getattr(track, "colour_hex", None),
        aspect=getattr(track, "aspect", None),
    )
    session.add(row)
    session.flush()          # need row.id before attaching the reads

    # Keep every individual OCR attempt. These are the evidence behind the
    # consensus -- being able to show a judge "we read this 34 times, here is
    # the vote" is worth far more than a bare string.
    for reading in getattr(track, "plate_readings", ()) or ():
        session.add(PlateRead(
            sighting_id=row.id,
            text=reading.text,
            confidence=float(reading.confidence),
            frame_index=int(getattr(reading, "frame_index", 0)),
            at=track.first_seen,
            plate_width_px=int(getattr(reading, "plate_width_px", 0)),
        ))

    session.commit()
    return row


def add_plate_read(session: Session, sighting_id: int, text: str, confidence: float,
                   frame_index: int = 0, at: datetime | None = None,
                   plate_width_px: int = 0) -> PlateRead:
    """One OCR attempt. Milestone 5 calls this many times per sighting."""
    row = PlateRead(sighting_id=sighting_id, text=text, confidence=confidence,
                    frame_index=frame_index, at=at or datetime.now(),
                    plate_width_px=plate_width_px)
    session.add(row)
    session.commit()
    return row


def recent_sightings(session: Session, limit: int = 50,
                     camera_id: str | None = None) -> Sequence[Sighting]:
    stmt = select(Sighting).order_by(Sighting.first_seen.desc()).limit(limit)
    if camera_id:
        stmt = stmt.where(Sighting.camera_id == camera_id)
    return session.scalars(stmt).all()


def find_by_plate(session: Session, plate: str) -> Sequence[Sighting]:
    """Exact-match plate lookup, ordered in time -- the raw material for the
    journey reconstruction in Milestone 8."""
    return session.scalars(
        select(Sighting)
        .where(Sighting.plate_text == plate.upper())
        .order_by(Sighting.first_seen)
    ).all()


def stats(session: Session) -> dict:
    """Everything the `db stats` command and the M6 KPI row need."""
    total = session.scalar(select(func.count(Sighting.id))) or 0
    by_camera = session.execute(
        select(Sighting.camera_id, func.count(Sighting.id))
        .group_by(Sighting.camera_id).order_by(Sighting.camera_id)
    ).all()
    by_class = session.execute(
        select(Sighting.vehicle_class, func.count(Sighting.id))
        .group_by(Sighting.vehicle_class).order_by(func.count(Sighting.id).desc())
    ).all()
    by_crossing = session.execute(
        select(Sighting.crossing, func.count(Sighting.id))
        .where(Sighting.crossing.is_not(None)).group_by(Sighting.crossing)
    ).all()
    window = session.execute(
        select(func.min(Sighting.first_seen), func.max(Sighting.last_seen))
    ).one()
    return {
        "sightings": total,
        "cameras": session.scalar(select(func.count(Camera.id))) or 0,
        "plate_reads": session.scalar(select(func.count(PlateRead.id))) or 0,
        "with_plate": session.scalar(
            select(func.count(Sighting.id)).where(Sighting.plate_text.is_not(None))) or 0,
        "by_camera": dict(by_camera),
        "by_class": dict(by_class),
        "by_crossing": dict(by_crossing),
        "first_seen": window[0],
        "last_seen": window[1],
    }


# --- identity (M7) ---------------------------------------------------------

def sighting_views(session: Session, camera_id: str | None = None):
    """Every sighting, in the shape the resolver wants.

    Deliberately a plain dataclass rather than the ORM row, so the resolver has
    no database dependency and stays unit-testable.
    """
    from app.identity.resolver import SightingView

    stmt = select(Sighting).order_by(Sighting.first_seen)
    if camera_id:
        stmt = stmt.where(Sighting.camera_id == camera_id)
    return [
        SightingView(
            id=r.id, camera_id=r.camera_id,
            first_seen=r.first_seen, last_seen=r.last_seen,
            vehicle_class=r.vehicle_class,
            plate_text=r.plate_text, plate_confidence=r.plate_confidence,
            colour_hex=r.colour_hex, aspect=r.aspect,
        )
        for r in session.scalars(stmt).all()
    ]


def save_resolution(session: Session, identities, verdicts) -> dict:
    """Replace the previous resolution with this one.

    Resolution is a pure function of the sightings, so it is recomputed whole
    rather than patched. That keeps it impossible for identities and links to
    drift out of agreement.
    """
    from app.store.models import IdentityLink, VehicleIdentity
    from app.store.models import Journey as JourneyRow
    from app.store.models import JourneyHop

    # Order matters. Journeys reference identities, so they have to go first --
    # otherwise the second run of `resolve` dies on a foreign key constraint,
    # which is exactly the sort of thing that only shows up when you re-run
    # something in front of a judge.
    session.query(JourneyHop).delete()
    session.query(JourneyRow).delete()
    session.execute(sa_update(Sighting).values(identity_id=None))
    session.query(IdentityLink).delete()
    session.query(VehicleIdentity).delete()
    session.flush()

    by_sighting: dict[int, int] = {}
    kept = 0
    for ident in identities:
        # A single unlinked sighting is not a journey; do not manufacture one.
        if len(ident.sighting_ids) < 2 and not ident.plate:
            continue
        row = VehicleIdentity(
            key=ident.key, plate=ident.plate,
            plate_confidence=round(ident.plate_confidence or 0.0, 4),
            confidence=round(ident.confidence or 0.0, 4),
            sighting_count=len(ident.sighting_ids),
            camera_count=len(ident.cameras),
            cameras=">".join(ident.cameras)[:256],
            first_seen=ident.first_seen, last_seen=ident.last_seen,
        )
        session.add(row)
        session.flush()
        kept += 1
        for sid in ident.sighting_ids:
            by_sighting[sid] = row.id
        session.execute(sa_update(Sighting)
                        .where(Sighting.id.in_(ident.sighting_ids))
                        .values(identity_id=row.id))

    for v in verdicts:
        session.add(IdentityLink(
            identity_id=by_sighting.get(v.from_id) if v.accepted else None,
            from_sighting_id=v.from_id, to_sighting_id=v.to_id,
            from_camera=v.from_camera, to_camera=v.to_camera,
            gap_s=round(v.gap_s, 2), distance_km=round(v.distance_km, 3),
            implied_speed_kmph=round(v.implied_speed_kmph, 2),
            plate_similarity=round(v.plate_similarity, 4),
            appearance_similarity=round(v.appearance_similarity, 4),
            topology_score=round(v.topology_score, 4),
            score=round(v.score, 4), accepted=v.accepted, reason=v.reason[:256],
        ))

    session.commit()
    return {"identities": kept,
            "accepted": sum(1 for v in verdicts if v.accepted),
            "rejected": sum(1 for v in verdicts if not v.accepted)}


def list_identities(session: Session, limit: int = 50, min_cameras: int = 1):
    from app.store.models import VehicleIdentity

    return session.scalars(
        select(VehicleIdentity)
        .where(VehicleIdentity.camera_count >= min_cameras)
        .order_by(VehicleIdentity.camera_count.desc(),
                  VehicleIdentity.first_seen.desc())
        .limit(limit)).all()


def rejected_links(session: Session, limit: int = 20, min_plate_similarity: float = 0.7):
    """Near-misses the system refused. The credibility exhibit."""
    from app.store.models import IdentityLink

    return session.scalars(
        select(IdentityLink)
        .where(IdentityLink.accepted.is_(False),
               IdentityLink.plate_similarity >= min_plate_similarity)
        .order_by(IdentityLink.plate_similarity.desc()).limit(limit)).all()


def identity_links(session: Session, identity_id: int):
    from app.store.models import IdentityLink

    return session.scalars(
        select(IdentityLink)
        .where(IdentityLink.identity_id == identity_id,
               IdentityLink.accepted.is_(True))
        .order_by(IdentityLink.gap_s)).all()


# --- trajectories (M8) -----------------------------------------------------

def save_journeys(session: Session, journeys, identity_ids: dict) -> int:
    """Replace stored journeys with a freshly built set."""
    from app.store.models import Journey as JourneyRow
    from app.store.models import JourneyHop

    session.query(JourneyHop).delete()
    session.query(JourneyRow).delete()
    session.flush()

    for j in journeys:
        row = JourneyRow(
            identity_id=identity_ids.get(j.identity_key),
            identity_key=j.identity_key,
            plate=j.plate, plate_confidence=j.plate_confidence,
            confidence=round(j.confidence, 4),
            route=">".join(j.route)[:256],
            full_route=">".join(j.full_route)[:256],
            coverage_gaps=",".join(j.coverage_gaps)[:128],
            hop_count=len(j.hops),
            total_distance_km=j.total_distance_km,
            total_duration_s=j.total_duration_s,
            average_speed_kmph=j.average_speed_kmph,
            started_at=j.started_at, ended_at=j.ended_at,
        )
        session.add(row)
        session.flush()
        for n, hop in enumerate(j.hops):
            session.add(JourneyHop(
                journey_id=row.id, sequence=n,
                from_camera=hop.from_camera, to_camera=hop.to_camera,
                via=",".join(hop.via)[:128],
                departed_at=hop.departed_at, arrived_at=hop.arrived_at,
                gap_s=hop.gap_s, distance_km=hop.distance_km,
                implied_speed_kmph=hop.implied_speed_kmph,
                typical_speed_kmph=hop.typical_speed_kmph,
                delay_ratio=hop.delay_ratio, confidence=hop.confidence,
            ))
    session.commit()
    return len(journeys)


def list_journeys(session: Session, limit: int = 50, min_hops: int = 1):
    from app.store.models import Journey as JourneyRow

    return session.scalars(
        select(JourneyRow).where(JourneyRow.hop_count >= min_hops)
        .order_by(JourneyRow.hop_count.desc(), JourneyRow.started_at.desc())
        .limit(limit)).all()


def journey_for_plate(session: Session, plate: str):
    from app.store.models import Journey as JourneyRow

    return session.scalars(
        select(JourneyRow).where(JourneyRow.plate == plate.upper())
        .order_by(JourneyRow.started_at.desc())).all()


def congested_hops(session: Session, limit: int = 20, min_delay_ratio: float = 1.5):
    """Legs taken markedly slower than the corridor's usual pace.

    The seed of the congestion analytics in Milestone 9 -- a delay observed
    from real travel times rather than assumed from vehicle counts.
    """
    from app.store.models import JourneyHop

    return session.scalars(
        select(JourneyHop).where(JourneyHop.delay_ratio >= min_delay_ratio)
        .order_by(JourneyHop.delay_ratio.desc()).limit(limit)).all()


# --- analytics and alerts (M9) ---------------------------------------------

def all_sightings(session: Session):
    return session.scalars(select(Sighting).order_by(Sighting.first_seen)).all()


def hops_with_plate(session: Session):
    """Journey hops carrying their journey's plate, for the over-speed rule."""
    from types import SimpleNamespace

    from app.store.models import Journey as JourneyRow
    from app.store.models import JourneyHop

    rows = session.execute(
        select(JourneyHop, JourneyRow.plate)
        .join(JourneyRow, JourneyHop.journey_id == JourneyRow.id)
        .order_by(JourneyHop.arrived_at)).all()
    return [SimpleNamespace(
        from_camera=h.from_camera, to_camera=h.to_camera,
        gap_s=h.gap_s, distance_km=h.distance_km,
        implied_speed_kmph=h.implied_speed_kmph,
        typical_speed_kmph=h.typical_speed_kmph,
        delay_ratio=h.delay_ratio, arrived_at=h.arrived_at, plate=plate,
    ) for h, plate in rows]


def save_alerts(session: Session, alerts) -> int:
    """Replace the stored alerts with a freshly evaluated set."""
    from app.store.models import Alert as AlertRow

    session.query(AlertRow).delete()
    for a in alerts:
        session.add(AlertRow(
            kind=a.kind, severity=a.severity, subject=a.subject,
            message=a.message[:256], detail=a.detail[:512], at=a.at,
            value=round(a.value, 4), threshold=round(a.threshold, 4),
            plate=a.plate, confidence=round(a.confidence, 4),
            needs_review=a.needs_review,
        ))
    session.commit()
    return len(alerts)


def list_alerts(session: Session, limit: int = 50, kind: str | None = None):
    from app.store.models import Alert as AlertRow

    order = {"critical": 0, "warning": 1, "info": 2}
    stmt = select(AlertRow)
    if kind:
        stmt = stmt.where(AlertRow.kind == kind)
    rows = session.scalars(stmt.limit(limit * 4)).all()
    rows.sort(key=lambda r: (order.get(r.severity, 9), r.kind))
    return rows[:limit]


# ---------------------------------------------------------------------------
# Unclear Possible Leads
#
# A "lead" is a pair the resolver scored but did NOT assert -- a near miss. The
# evidence is already on the IdentityLink row; this is the query layer that
# turns it into something an investigator can work through.
#
# Note what a lead is not: it is not a low-confidence identity. Because a link
# must clear accept_threshold to join an identity at all, and identity
# confidence is the MINIMUM of its accepted links, no multi-camera identity can
# exist below that threshold. The sub-threshold evidence lives here instead.
# ---------------------------------------------------------------------------

LEAD_STATUSES = ("unreviewed", "reviewed", "potential_match", "rejected", "escalated")


def lead_links(session: Session, limit: int = 100, offset: int = 0,
               camera: str | None = None, plate: str | None = None,
               min_score: float = 0.0, max_score: float = 1.0,
               status: str | None = None, vehicle_class: str | None = None,
               since=None, until=None):
    """Refused links, newest first, with their review state and both sightings."""
    A = aliased(Sighting)
    B = aliased(Sighting)
    q = (session.query(IdentityLink, A, B, LeadReview)
         .join(A, IdentityLink.from_sighting_id == A.id)
         .join(B, IdentityLink.to_sighting_id == B.id)
         .outerjoin(LeadReview, LeadReview.link_id == IdentityLink.id)
         .filter(IdentityLink.accepted == False)  # noqa: E712
         .filter(IdentityLink.score >= min_score)
         .filter(IdentityLink.score <= max_score))
    if camera:
        q = q.filter(or_(IdentityLink.from_camera == camera,
                         IdentityLink.to_camera == camera))
    if plate:
        like = f"%{plate.upper()}%"
        q = q.filter(or_(A.plate_text.like(like), B.plate_text.like(like)))
    if vehicle_class:
        q = q.filter(or_(A.vehicle_class == vehicle_class,
                         B.vehicle_class == vehicle_class))
    if since is not None:
        q = q.filter(A.first_seen >= since)
    if until is not None:
        q = q.filter(A.first_seen <= until)
    if status:
        if status == "unreviewed":
            q = q.filter(or_(LeadReview.status == None,  # noqa: E711
                             LeadReview.status == "unreviewed"))
        else:
            q = q.filter(LeadReview.status == status)
    total = q.count()
    rows = (q.order_by(IdentityLink.score.desc(), IdentityLink.id.desc())
             .offset(offset).limit(limit).all())
    return rows, total


def set_lead_status(session: Session, link_id: int, status: str, note: str = ""):
    """Record a human disposition. Upsert -- one review per link."""
    if status not in LEAD_STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {LEAD_STATUSES}")
    row = session.query(LeadReview).filter(LeadReview.link_id == link_id).one_or_none()
    if row is None:
        row = LeadReview(link_id=link_id)
        session.add(row)
    row.status = status
    row.note = (note or "")[:512]
    row.updated_at = datetime.now()
    session.commit()
    return row


def links_touching_sightings(session: Session, sighting_ids: list[int]):
    """Every judged pair -- accepted or refused -- involving these sightings."""
    if not sighting_ids:
        return []
    return (session.query(IdentityLink)
            .filter(or_(IdentityLink.from_sighting_id.in_(sighting_ids),
                        IdentityLink.to_sighting_id.in_(sighting_ids)))
            .order_by(IdentityLink.accepted.desc(), IdentityLink.score.desc())
            .all())


def plate_reads_for(session: Session, sighting_ids: list[int], per_sighting: int = 6):
    """Strongest OCR readings per sighting -- the evidence behind the consensus."""
    if not sighting_ids:
        return {}
    rows = (session.query(PlateRead)
            .filter(PlateRead.sighting_id.in_(sighting_ids))
            .order_by(PlateRead.confidence.desc()).all())
    out: dict[int, list] = {}
    for r in rows:
        bucket = out.setdefault(r.sighting_id, [])
        if len(bucket) < per_sighting:
            bucket.append(r)
    return out
