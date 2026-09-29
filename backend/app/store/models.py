"""Database schema.

SQLite for the prototype, Postgres in production -- and these models are
identical either way. That is the whole reason to use an ORM here rather than
raw SQL: the migration story is one connection string, not a rewrite.

WHY SQLITE IS ENOUGH
SQLite allows one writer at a time, which sounds fatal for N camera workers.
It is not, because of a decision made back in Milestone 0: consensus happens in
the worker, so a camera writes **one row per completed track**, not one per
frame. Three cameras produce single-digit writes per second. WAL mode plus a
busy timeout absorbs that without breaking a sweat.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Index, Integer,
                        String, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Camera(Base):
    """Mirror of cameras.yaml, so the API can serve the network without
    reading config, and so historical data keeps meaning if config changes."""

    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    heading_deg: Mapped[float] = mapped_column(Float, default=0.0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    sightings: Mapped[list["Sighting"]] = relationship(back_populates="camera")

    def __repr__(self) -> str:
        return f"<Camera {self.id} {self.name!r}>"


class Sighting(Base):
    """One vehicle, seen once, by one camera.

    This is the atom of the whole system. A track that entered the frame,
    was followed, and left. Everything downstream -- identity resolution,
    trajectories, analytics -- is built by relating sightings to each other.
    """

    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # run_id makes re-running a camera idempotent-safe: track ids restart from
    # 1 on every run, so without it a second run would collide with the first.
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)

    vehicle_class: Mapped[str] = mapped_column(String(24), index=True)
    class_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Virtual wall-clock times from VideoSource -- the shared city clock.
    first_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    frames: Mapped[int] = mapped_column(Integer, default=0)

    crossing: Mapped[str | None] = mapped_column(String(24), nullable=True)

    box_x1: Mapped[int] = mapped_column(Integer, default=0)
    box_y1: Mapped[int] = mapped_column(Integer, default=0)
    box_x2: Mapped[int] = mapped_column(Integer, default=0)
    box_y2: Mapped[int] = mapped_column(Integer, default=0)

    # Filled by Milestone 5. Denormalised onto the sighting so that the
    # cross-camera queries in M7 stay a single indexed lookup.
    plate_text: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Appearance features for Milestone 7. Cheap on purpose: dominant colour of
    # the vehicle crop and the box aspect ratio carry most of the
    # discriminating power a learned re-ID embedding would, for none of the cost.
    colour_hex: Mapped[str | None] = mapped_column(String(8), nullable=True)
    aspect: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Set by the resolver: which vehicle this sighting was attributed to.
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicle_identities.id"), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    camera: Mapped["Camera"] = relationship(back_populates="sightings")
    plate_reads: Mapped[list["PlateRead"]] = relationship(
        back_populates="sighting", cascade="all, delete-orphan")

    __table_args__ = (
        # The same track from the same run is one sighting. Re-running a clip
        # cannot silently double the vehicle count.
        UniqueConstraint("run_id", "camera_id", "track_id", name="uq_sighting_track"),
        Index("ix_sightings_camera_time", "camera_id", "first_seen"),
        Index("ix_sightings_plate_time", "plate_text", "first_seen"),
    )

    def __repr__(self) -> str:
        return (f"<Sighting {self.camera_id}#{self.track_id} {self.vehicle_class} "
                f"{self.first_seen:%H:%M:%S} plate={self.plate_text}>")


class PlateRead(Base):
    """One OCR attempt on one frame. Populated in Milestone 5.

    Kept separate from Sighting deliberately: the individual reads are the
    evidence behind the consensus, and being able to show them ("we read this
    plate 34 times, here is the vote") is worth far more to a judge than a
    single string.
    """

    __tablename__ = "plate_reads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"), index=True)

    text: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    plate_width_px: Mapped[int] = mapped_column(Integer, default=0)

    sighting: Mapped["Sighting"] = relationship(back_populates="plate_reads")


class VehicleIdentity(Base):
    """A cluster of sightings believed to be one physical vehicle.

    Note what this is not: a claim of certainty. ``confidence`` is the weakest
    link in the chain, and a journey with one shaky hop is reported as shaky.
    """

    __tablename__ = "vehicle_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(32), index=True)
    plate: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    plate_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    sighting_count: Mapped[int] = mapped_column(Integer, default=0)
    camera_count: Mapped[int] = mapped_column(Integer, default=0)
    cameras: Mapped[str] = mapped_column(String(256), default="")   # "CAM01>CAM02>CAM03"

    first_seen: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    def __repr__(self) -> str:
        return f"<Identity {self.key} {self.plate} {self.cameras} conf={self.confidence:.2f}>"


class IdentityLink(Base):
    """One judged pair of sightings -- accepted or rejected, with the reason.

    Rejected links are stored deliberately. Showing a judge a pair the system
    *refused* -- "plate similarity 0.94, but that implies 214 km/h" -- proves
    there is reasoning under the dashboard rather than a lookup table.
    """

    __tablename__ = "identity_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicle_identities.id"), nullable=True, index=True)

    from_sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"), index=True)
    to_sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"), index=True)
    from_camera: Mapped[str] = mapped_column(String(32))
    to_camera: Mapped[str] = mapped_column(String(32))

    gap_s: Mapped[float] = mapped_column(Float, default=0.0)
    distance_km: Mapped[float] = mapped_column(Float, default=0.0)
    implied_speed_kmph: Mapped[float] = mapped_column(Float, default=0.0)

    plate_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    appearance_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    topology_score: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)

    accepted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reason: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Journey(Base):
    """One vehicle's reconstructed trip. Milestone 8.

    Stored rather than recomputed on every request because the dashboard, the
    analytics and the alert rules all read it, and they must all see the same
    numbers.
    """

    __tablename__ = "journeys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicle_identities.id"), nullable=True, index=True)
    identity_key: Mapped[str] = mapped_column(String(32), index=True)

    plate: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    plate_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, index=True)

    route: Mapped[str] = mapped_column(String(256), default="")       # observed
    full_route: Mapped[str] = mapped_column(String(256), default="")  # incl. inferred
    coverage_gaps: Mapped[str] = mapped_column(String(128), default="")

    hop_count: Mapped[int] = mapped_column(Integer, default=0)
    total_distance_km: Mapped[float] = mapped_column(Float, default=0.0)
    total_duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    average_speed_kmph: Mapped[float] = mapped_column(Float, default=0.0)

    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    ended_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    hops: Mapped[list["JourneyHop"]] = relationship(
        back_populates="journey", cascade="all, delete-orphan",
        order_by="JourneyHop.sequence")

    def __repr__(self) -> str:
        return f"<Journey {self.plate or self.identity_key} {self.route}>"


class JourneyHop(Base):
    """One leg between two consecutive observed cameras."""

    __tablename__ = "journey_hops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    journey_id: Mapped[int] = mapped_column(ForeignKey("journeys.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)

    from_camera: Mapped[str] = mapped_column(String(32), index=True)
    to_camera: Mapped[str] = mapped_column(String(32), index=True)
    via: Mapped[str] = mapped_column(String(128), default="")   # unobserved cameras

    departed_at: Mapped[datetime] = mapped_column(DateTime)
    arrived_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    gap_s: Mapped[float] = mapped_column(Float, default=0.0)
    distance_km: Mapped[float] = mapped_column(Float, default=0.0)
    implied_speed_kmph: Mapped[float] = mapped_column(Float, default=0.0)
    typical_speed_kmph: Mapped[float] = mapped_column(Float, default=0.0)
    delay_ratio: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    journey: Mapped["Journey"] = relationship(back_populates="hops")


class Alert(Base):
    """A raised alert. Milestone 9.

    Stored with the value that triggered it and the threshold it crossed, so a
    controller can see why rather than being asked to trust a red dot.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    subject: Mapped[str] = mapped_column(String(64), index=True)

    message: Mapped[str] = mapped_column(String(256))
    detail: Mapped[str] = mapped_column(String(512), default="")

    at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.0)

    plate: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class LeadReview(Base):
    """An investigator's disposition of one low-confidence lead.

    A SEPARATE TABLE ON PURPOSE. The link row is machine output -- what the
    resolver computed and why. This is human output -- what a person decided
    about it. Keeping them apart means re-running `resolve` never destroys an
    investigator's work, and an investigator's note never masquerades as
    evidence the system produced.

    New table rather than new columns, so it is created by create_all() on an
    existing database without a migration.
    """

    __tablename__ = "lead_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    link_id: Mapped[int] = mapped_column(ForeignKey("identity_links.id"),
                                         index=True, unique=True)

    # unreviewed | reviewed | potential_match | rejected | escalated
    status: Mapped[str] = mapped_column(String(24), default="unreviewed", index=True)
    note: Mapped[str] = mapped_column(String(512), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
