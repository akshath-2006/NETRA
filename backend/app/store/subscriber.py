"""Bus subscriber that writes sightings to the store.

This is the piece that shows the bus is real rather than decorative. The camera
worker publishes a sighting and does not know or care that a database exists.
In production this subscriber runs in a separate consumer process reading from
Kafka; the function body is unchanged.
"""

from __future__ import annotations

from app.core.bus import Event, EventBus, Topics
from app.core.logging import get_logger
from app.store.repository import save_sighting

log = get_logger("store.sub")


class SightingWriter:
    """Subscribes to sighting.completed and persists what arrives."""

    def __init__(self, sessionmaker, run_id: str) -> None:
        self._sessionmaker = sessionmaker
        self.run_id = run_id
        self.written = 0
        self.skipped = 0

    def register(self, bus: EventBus) -> "SightingWriter":
        bus.subscribe(Topics.SIGHTING_COMPLETED, self.on_sighting)
        return self

    def on_sighting(self, event: Event) -> None:
        track = event.payload.get("track")
        if track is None:
            return
        with self._sessionmaker() as session:
            row = save_sighting(session, self.run_id, track)
        if row is None:
            self.skipped += 1
        else:
            self.written += 1
            log.debug("saved sighting %s#%s (%s)", track.camera_id, track.track_id, track.label)
