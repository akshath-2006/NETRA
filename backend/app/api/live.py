"""Live surfaces: MJPEG camera preview and the WebSocket event feed.

Both deliberately read from what already exists rather than inventing a second
channel between the workers and the API:

* **MJPEG** re-serves the JPEG each worker writes to ``data/live/<CAM>.jpg``.
* **WebSocket** tails the database for rows newer than the last one it sent.

Tailing the store rather than wiring an in-memory queue between processes means
a browser that connects late still sees a live feed, a worker restart does not
drop the socket, and there is no IPC to fail in front of judges. It costs one
cheap indexed query every half second.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.core.logging import get_logger
from app.store.models import Sighting

log = get_logger("api.live")

BOUNDARY = "netraframe"
_placeholder_cache: dict[str, bytes] = {}


def placeholder(camera_id: str, message: str = "NO SIGNAL") -> bytes:
    """A dark tile, so a stopped camera looks deliberate rather than broken."""
    key = f"{camera_id}:{message}"
    if key in _placeholder_cache:
        return _placeholder_cache[key]

    img = np.full((360, 640, 3), (28, 22, 18), dtype=np.uint8)
    cv2.putText(img, camera_id, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (245, 240, 235), 1, cv2.LINE_AA)
    cv2.putText(img, message, (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 130, 140), 1, cv2.LINE_AA)
    cv2.line(img, (0, 92), (640, 92), (60, 170, 235), 2)
    ok, buf = cv2.imencode(".jpg", img)
    _placeholder_cache[key] = buf.tobytes() if ok else b""
    return _placeholder_cache[key]


async def mjpeg_stream(live_dir: Path, camera_id: str, fps: float = 10.0):
    """Yield a multipart MJPEG stream from the worker's latest frame."""
    path = Path(live_dir) / f"{camera_id}.jpg"
    delay = 1.0 / max(1.0, fps)
    while True:
        try:
            data = path.read_bytes() if path.exists() else placeholder(camera_id)
        except (OSError, ValueError):
            data = placeholder(camera_id)
        if not data:
            data = placeholder(camera_id)
        yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
               f"Content-Length: {len(data)}\r\n\r\n").encode() + data + b"\r\n"
        await asyncio.sleep(delay)


def _row_to_event(row: Sighting) -> dict:
    return {
        "id": row.id,
        "camera_id": row.camera_id,
        "track_id": row.track_id,
        "vehicle_class": row.vehicle_class,
        "plate": row.plate_text,
        "plate_confidence": row.plate_confidence,
        "crossing": row.crossing,
        "first_seen": row.first_seen.isoformat(timespec="milliseconds"),
        "duration_s": row.duration_s,
    }


async def event_feed(websocket: WebSocket, sessionmaker, poll_s: float = 0.5) -> None:
    """Push new sightings to the browser as the workers produce them."""
    await websocket.accept()

    # Start from the most recent few rows so a page load is never blank.
    with sessionmaker() as session:
        recent = session.scalars(
            select(Sighting).order_by(Sighting.id.desc()).limit(15)).all()
    last_id = recent[0].id if recent else 0
    await websocket.send_json({"type": "backlog",
                               "events": [_row_to_event(r) for r in reversed(recent)]})

    try:
        while True:
            await asyncio.sleep(poll_s)
            with sessionmaker() as session:
                rows = session.scalars(
                    select(Sighting).where(Sighting.id > last_id)
                    .order_by(Sighting.id).limit(50)).all()
            if rows:
                last_id = rows[-1].id
                await websocket.send_json({"type": "sightings",
                                           "events": [_row_to_event(r) for r in rows]})
            else:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        log.debug("event feed client disconnected")
    except Exception:
        log.exception("event feed failed")
