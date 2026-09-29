"""FastAPI application -- the intelligence core's front door.

Everything here is a thin read layer. The API never computes: it asks the
repository, which is the single definition of "what counts as a sighting" that
the analytics in Milestone 9 will share. Keeping it thin is what stops the
dashboard and the pipeline drifting apart.

Interactive docs are generated for free at /docs -- worth showing a judge.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.api.live import BOUNDARY, event_feed, mjpeg_stream
from app.api.tiles import cache_stats, get_tile
from app.core.config import AppConfig, load_config
from app.core.logging import get_logger, setup_logging
from app.store.db import get_sessionmaker, init_db
from app.store.models import Sighting as SightingRow
from app.store.repository import (LEAD_STATUSES, all_sightings, congested_hops,
                                  find_by_plate, hops_with_plate, identity_links,
                                  journey_for_plate, lead_links,
                                  links_touching_sightings, list_alerts,
                                  list_cameras, list_identities, list_journeys,
                                  plate_reads_for, recent_sightings, rejected_links,
                                  set_lead_status, stats, upsert_cameras)

log = get_logger("api")


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()
    setup_logging(cfg.runtime.log_level)
    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)
    with Session() as session:
        upsert_cameras(session, cfg.cameras)

    app = FastAPI(title="NETRA", version="0.11.0",
                  description="City-wide multi-camera ANPR and trajectory analytics")
    # The dashboard runs on Vite's dev server during development, so it is a
    # different origin. Local-only prototype; tighten before any deployment.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    app.state.cfg = cfg
    app.state.sessionmaker = Session
    app.state.started_at = datetime.now()

    # -- meta ---------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        """Liveness plus everything the demo needs to be ready."""
        tiles = cache_stats(cfg.paths.data_dir / "tiles")
        with Session() as session:
            s = stats(session)
        return {
            "status": "ok", "version": app.version,
            "started_at": app.state.started_at.isoformat(timespec="seconds"),
            "database": str(cfg.paths.database),
            "cameras_configured": len(cfg.cameras),
            "sightings": s["sightings"],
            "seeded": s["sightings"] > 0,
            "tiles_cached": tiles["tiles"],
            "offline_ready": tiles["tiles"] > 0,
        }

    @app.get("/api/tiles/{z}/{x}/{y}.png")
    def map_tile(z: int, x: int, y: int) -> Response:
        """Cached map tiles.

        Served from disk when we have them, fetched and cached on a miss, and
        replaced by a plain dark tile when neither works -- so venue wifi
        failing degrades the map instead of breaking it.
        """
        if not (0 <= z <= 20) or x < 0 or y < 0 or x >= 2 ** z or y >= 2 ** z:
            raise HTTPException(404, "tile out of range")
        data, source = get_tile(cfg.paths.data_dir / "tiles", z, x, y)
        return Response(content=data, media_type="image/png", headers={
            "Cache-Control": "public, max-age=604800",
            "X-Tile-Source": source,
        })

    # -- network ------------------------------------------------------------

    @app.get("/api/cameras")
    def cameras() -> list[dict]:
        live_dir = cfg.paths.data_dir / "live"
        with Session() as session:
            rows = list_cameras(session)
        out = []
        for row in rows:
            frame = live_dir / f"{row.id}.jpg"
            out.append({
                "id": row.id, "name": row.name,
                "lat": row.lat, "lon": row.lon,
                "heading_deg": row.heading_deg, "enabled": row.enabled,
                # "live" means a worker wrote a frame recently, not merely that
                # the camera is configured.
                "live": frame.exists() and
                        (datetime.now().timestamp() - frame.stat().st_mtime) < 5.0,
                "stream_url": f"/api/live/{row.id}",
            })
        return out

    # -- data ---------------------------------------------------------------

    @app.get("/api/stats")
    def kpis() -> dict:
        with Session() as session:
            s = stats(session)
        from app.store.models import VehicleIdentity
        from sqlalchemy import func as sqlfunc, select as sqlselect
        with Session() as session:
            journeys = session.scalar(
                sqlselect(sqlfunc.count(VehicleIdentity.id))
                .where(VehicleIdentity.camera_count > 1)) or 0
        return {
            "journeys": journeys,
            "cameras": s["cameras"],
            "cameras_live": sum(1 for c in cameras() if c["live"]),
            "sightings": s["sightings"],
            "plates": s["with_plate"],
            "plate_reads": s["plate_reads"],
            "by_camera": s["by_camera"],
            "by_class": s["by_class"],
            "by_direction": s["by_crossing"],
            "first_seen": s["first_seen"].isoformat() if s["first_seen"] else None,
            "last_seen": s["last_seen"].isoformat() if s["last_seen"] else None,
        }

    def _plate_status(plate: str | None, conf: float | None) -> str:
        """How much the operator should trust this plate.

        The pipeline already refuses to store anything it judged unreadable, so
        a stored plate is at least tentative. We re-derive the confirmed /
        tentative split here from the same thresholds the worker used, which
        keeps the screen and the pipeline telling the same story without
        needing a schema change.
        """
        if not plate:
            return "unreadable"
        a = (cfg.raw.get("system", {}) or {}).get("anpr", {}) or {}
        return "confirmed" if (conf or 0.0) >= float(a.get("confirm_confidence", 0.60)) \
            else "tentative"

    @app.get("/api/sightings")
    def sightings(limit: int = Query(50, ge=1, le=500),
                  camera: str | None = None) -> list[dict]:
        with Session() as session:
            rows = recent_sightings(session, limit=limit, camera_id=camera)
            return [{
                "id": r.id, "camera_id": r.camera_id, "track_id": r.track_id,
                "vehicle_class": r.vehicle_class,
                "class_confidence": r.class_confidence,
                "plate": r.plate_text, "plate_confidence": r.plate_confidence,
                "plate_status": _plate_status(r.plate_text, r.plate_confidence),
                "crossing": r.crossing, "duration_s": r.duration_s,
                "frames": r.frames,
                "first_seen": r.first_seen.isoformat(timespec="milliseconds"),
                "last_seen": r.last_seen.isoformat(timespec="milliseconds"),
            } for r in rows]

    @app.get("/api/plates/{plate}")
    def plate_lookup(plate: str) -> dict:
        """Every sighting of one plate, in time order.

        This is the raw material for Milestone 8's journey reconstruction --
        the endpoint stays, the response gains hops and confidences.
        """
        with Session() as session:
            rows = find_by_plate(session, plate)
        if not rows:
            raise HTTPException(404, f"no sightings for plate {plate.upper()}")
        return {
            "plate": plate.upper(),
            "sightings": len(rows),
            "cameras": sorted({r.camera_id for r in rows}),
            "first_seen": rows[0].first_seen.isoformat(timespec="milliseconds"),
            "last_seen": rows[-1].last_seen.isoformat(timespec="milliseconds"),
            "hops": [{
                "camera_id": r.camera_id,
                "at": r.first_seen.isoformat(timespec="milliseconds"),
                "vehicle_class": r.vehicle_class,
                "plate_confidence": r.plate_confidence,
            } for r in rows],
        }

    # -- identity (M7) ------------------------------------------------------

    @app.get("/api/identities")
    def identities(limit: int = Query(50, ge=1, le=500), min_cameras: int = 1) -> list[dict]:
        """Vehicles resolved across cameras, most-travelled first."""
        with Session() as session:
            rows = list_identities(session, limit=limit, min_cameras=min_cameras)
            return [{
                "key": r.key, "plate": r.plate,
                "plate_confidence": r.plate_confidence,
                "confidence": r.confidence,
                "cameras": r.cameras.split(">") if r.cameras else [],
                "camera_count": r.camera_count,
                "sighting_count": r.sighting_count,
                "first_seen": r.first_seen.isoformat(timespec="milliseconds"),
                "last_seen": r.last_seen.isoformat(timespec="milliseconds"),
                "hops": [{
                    "from": l.from_camera, "to": l.to_camera,
                    "gap_s": l.gap_s, "distance_km": l.distance_km,
                    "implied_speed_kmph": l.implied_speed_kmph,
                    "score": l.score, "reason": l.reason,
                } for l in identity_links(session, r.id)],
            } for r in rows]

    @app.get("/api/rejected-links")
    def rejections(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
        """Links the system refused despite similar plates.

        This endpoint exists for the demo. Showing what was *not* asserted, and
        why, is the clearest evidence that the association is reasoned rather
        than pattern-matched.
        """
        with Session() as session:
            rows = rejected_links(session, limit=limit)
            return [{
                "from_camera": r.from_camera, "to_camera": r.to_camera,
                "plate_similarity": r.plate_similarity,
                "gap_s": r.gap_s, "distance_km": r.distance_km,
                "implied_speed_kmph": r.implied_speed_kmph,
                "reason": r.reason,
            } for r in rows]

    # -- trajectories (M8) --------------------------------------------------

    def _journey_dict(row) -> dict:
        return {
            "identity_key": row.identity_key,
            "plate": row.plate,
            "plate_confidence": row.plate_confidence,
            "confidence": row.confidence,
            "route": row.route.split(">") if row.route else [],
            "full_route": row.full_route.split(">") if row.full_route else [],
            "coverage_gaps": [g for g in row.coverage_gaps.split(",") if g],
            "total_distance_km": row.total_distance_km,
            "total_duration_s": row.total_duration_s,
            "average_speed_kmph": row.average_speed_kmph,
            "started_at": row.started_at.isoformat(timespec="milliseconds"),
            "ended_at": row.ended_at.isoformat(timespec="milliseconds"),
            "hops": [{
                "sequence": h.sequence,
                "from": h.from_camera, "to": h.to_camera,
                "via": [v for v in h.via.split(",") if v],
                "departed_at": h.departed_at.isoformat(timespec="milliseconds"),
                "arrived_at": h.arrived_at.isoformat(timespec="milliseconds"),
                "gap_s": h.gap_s, "distance_km": h.distance_km,
                "implied_speed_kmph": h.implied_speed_kmph,
                "typical_speed_kmph": h.typical_speed_kmph,
                "delay_ratio": h.delay_ratio,
                "congested": h.delay_ratio >= 1.5,
                "confidence": h.confidence,
            } for h in row.hops],
        }

    @app.get("/api/journeys")
    def journeys(limit: int = Query(50, ge=1, le=500), min_hops: int = 1) -> list[dict]:
        with Session() as session:
            return [_journey_dict(r) for r in
                    list_journeys(session, limit=limit, min_hops=min_hops)]

    @app.get("/api/journeys/{plate}")
    def journey_by_plate(plate: str) -> list[dict]:
        with Session() as session:
            rows = journey_for_plate(session, plate)
        if not rows:
            raise HTTPException(404, f"no reconstructed journey for {plate.upper()}")
        return [_journey_dict(r) for r in rows]

    @app.get("/api/congestion")
    def congestion(limit: int = Query(20, ge=1, le=200),
                   min_delay_ratio: float = 1.5) -> list[dict]:
        """Legs taken markedly slower than the corridor's usual pace.

        Congestion measured from real travel times, not inferred from counts.
        """
        with Session() as session:
            rows = congested_hops(session, limit=limit, min_delay_ratio=min_delay_ratio)
            return [{
                "from": h.from_camera, "to": h.to_camera,
                "at": h.arrived_at.isoformat(timespec="milliseconds"),
                "distance_km": h.distance_km,
                "implied_speed_kmph": h.implied_speed_kmph,
                "typical_speed_kmph": h.typical_speed_kmph,
                "delay_ratio": h.delay_ratio,
            } for h in rows]

    # -- analytics and alerts (M9) ------------------------------------------

    @app.get("/api/analytics")
    def analytics(bucket_minutes: int = Query(5, ge=1, le=60)) -> dict:
        """Everything the charts need, computed from observed data only."""
        from app.analytics.metrics import build_report
        from app.core.city_graph import load_city_graph

        graph = load_city_graph(cfg.paths.root / "configs" / "city_graph.yaml",
                                cfg.cameras)
        typical = {}
        for a in cfg.cameras:
            for b in cfg.cameras:
                if a.id == b.id:
                    continue
                edge = graph.edge(a.id, b.id)
                if edge and not edge.derived:
                    typical[f"{a.id}->{b.id}"] = edge.typical_speed_kmph

        with Session() as session:
            report = build_report(all_sightings(session),
                                  list_journeys(session, limit=10000, min_hops=1),
                                  hops_with_plate(session),
                                  bucket_minutes=bucket_minutes,
                                  typical_speeds=typical)

        return {
            "window": {
                "start": report.window_start.isoformat() if report.window_start else None,
                "end": report.window_end.isoformat() if report.window_end else None,
                "bucket_minutes": report.bucket_minutes,
            },
            "totals": {
                "sightings": report.total_sightings,
                "journeys": report.total_journeys,
                "plates_read": report.plates_read,
                "plate_rate": round(report.plate_rate, 4),
            },
            "vehicle_mix": report.vehicle_mix,
            "volume": [{"start": b.start.isoformat(timespec="minutes"),
                        "count": b.count, "by_class": b.by_class}
                       for b in report.buckets],
            "cameras": [{"camera_id": c.camera_id, "sightings": c.sightings,
                         "share": round(c.share, 4), "plates_read": c.plates_read,
                         "plate_rate": round(c.plate_rate, 4),
                         "by_class": c.by_class} for c in report.cameras],
            "corridors": [{"from": c.from_camera, "to": c.to_camera,
                           "label": c.label, "trips": c.trips,
                           "distance_km": c.distance_km,
                           "median_travel_s": c.median_travel_s,
                           "free_flow_travel_s": c.free_flow_travel_s,
                           "median_speed_kmph": c.median_speed_kmph,
                           "congestion_index": c.congestion_index,
                           "level": c.level,
                           "baseline_from_data": c.baseline_from_data}
                          for c in report.corridors],
            "hotspots": [c.label for c in report.hotspots],
            "od_matrix": report.od_matrix,
            "peak": ({"start": report.peak_bucket.start.isoformat(timespec="minutes"),
                      "count": report.peak_bucket.count}
                     if report.peak_bucket else None),
        }

    @app.get("/api/alerts")
    def alerts(limit: int = Query(50, ge=1, le=500), kind: str | None = None) -> list[dict]:
        with Session() as session:
            rows = list_alerts(session, limit=limit, kind=kind)
            return [{
                "kind": r.kind, "severity": r.severity, "subject": r.subject,
                "message": r.message, "detail": r.detail,
                "at": r.at.isoformat(timespec="milliseconds") if r.at else None,
                "value": r.value, "threshold": r.threshold,
                "plate": r.plate, "confidence": r.confidence,
                "needs_review": r.needs_review,
            } for r in rows]


    # -- evidence, cases and leads ------------------------------------------
    #
    # Everything below reads evidence the pipeline already stored. Nothing here
    # computes a new judgement about a vehicle -- the API's job is to assemble
    # what the resolver concluded and show its working, not to re-decide it.

    def _snapshot_path(row):
        """Deterministic name, so no schema column is needed to find the image."""
        return (cfg.paths.data_dir / "evidence" / "sightings"
                / f"{row.run_id}_{row.camera_id}_{row.track_id}.jpg")

    def _sighting_evidence(row, reads=None) -> dict:
        snap = _snapshot_path(row)
        return {
            "id": row.id, "camera_id": row.camera_id, "track_id": row.track_id,
            "vehicle_class": row.vehicle_class,
            "class_confidence": row.class_confidence,
            "detection_confidence": row.detection_confidence,
            "plate": row.plate_text,
            "plate_confidence": row.plate_confidence,
            "plate_status": _plate_status(row.plate_text, row.plate_confidence),
            "first_seen": row.first_seen.isoformat(timespec="milliseconds"),
            "last_seen": row.last_seen.isoformat(timespec="milliseconds"),
            "duration_s": row.duration_s, "frames": row.frames,
            "crossing": row.crossing,
            "colour_hex": getattr(row, "colour_hex", None),
            "aspect": getattr(row, "aspect", None),
            "box": [row.box_x1, row.box_y1, row.box_x2, row.box_y2],
            # Honest about missing evidence: the UI must be able to say "no
            # snapshot retained for this run" rather than render a broken image.
            "snapshot": f"/api/sightings/{row.id}/snapshot" if snap.exists() else None,
            "plate_reads": [
                {"text": r.text, "confidence": round(r.confidence, 3),
                 "frame_index": r.frame_index, "plate_width_px": r.plate_width_px}
                for r in (reads or [])
            ],
        }

    def _link_evidence(link) -> dict:
        return {
            "id": link.id,
            "from_camera": link.from_camera, "to_camera": link.to_camera,
            "from_sighting_id": link.from_sighting_id,
            "to_sighting_id": link.to_sighting_id,
            "gap_s": round(link.gap_s, 1),
            "distance_km": round(link.distance_km, 2),
            "implied_speed_kmph": round(link.implied_speed_kmph, 1),
            "plate_similarity": round(link.plate_similarity, 3),
            "appearance_similarity": round(link.appearance_similarity, 3),
            "topology_score": round(link.topology_score, 3),
            "score": round(link.score, 3),
            "accepted": bool(link.accepted),
            "reason": link.reason,
            "at": link.created_at.isoformat(timespec="milliseconds") if link.created_at else None,
        }

    @app.get("/api/sightings/{sighting_id}/snapshot")
    def sighting_snapshot(sighting_id: int) -> FileResponse:
        """The one retained frame for a sighting. 404 when none was kept."""
        with Session() as session:
            row = session.get(SightingRow, sighting_id)
            if row is None:
                raise HTTPException(404, f"no sighting {sighting_id}")
            path = _snapshot_path(row)
        if not path.exists():
            raise HTTPException(
                404, "no snapshot retained for this sighting "
                     "(evidence.keep_snapshots was off, or the run predates it)")
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/case/{plate}")
    def vehicle_case(plate: str) -> dict:
        """Everything NETRA holds about one vehicle, assembled for review.

        Deliberately assembled from stored evidence only. Where a piece is
        absent -- no snapshot, no journey, no accepted link -- the field is
        null and the UI says so, rather than the API inventing a value.
        """
        want = plate.upper()
        with Session() as session:
            sightings = list(find_by_plate(session, want))
            if not sightings:
                raise HTTPException(404, f"no sightings for plate {want}")
            sightings.sort(key=lambda r: r.first_seen)
            ids = [r.id for r in sightings]

            reads = plate_reads_for(session, ids)
            links = links_touching_sightings(session, ids)
            accepted = [l for l in links if l.accepted]
            refused = [l for l in links if not l.accepted]

            identity = next((i for i in list_identities(session, limit=500)
                             if (i.plate or "").upper() == want), None)
            journeys = list(journey_for_plate(session, want))
            journeys.sort(key=lambda j: j.started_at)
            j = journeys[0] if journeys else None

            best_plate_conf = max((s.plate_confidence or 0.0) for s in sightings)
            cameras = list(dict.fromkeys(s.camera_id for s in sightings))

            # The user-facing identity confidence -- NOT detection confidence.
            # Weakest accepted link, or the plate itself when a vehicle was
            # only ever seen by one camera and there is nothing to link.
            identity_conf = (identity.confidence if identity is not None
                             else (min((l.score for l in accepted), default=None)
                                   or best_plate_conf))
            threshold = float((cfg.raw.get("system", {}) or {})
                              .get("identity", {}).get("display_threshold", 0.45))

            return {
                "plate": want,
                "plate_confidence": best_plate_conf,
                "plate_status": _plate_status(want, best_plate_conf),
                "vehicle_class": sightings[0].vehicle_class,
                "identity_key": identity.key if identity is not None else None,
                "identity_confidence": identity_conf,
                "confidence_basis": ("weakest accepted cross-camera link"
                                     if accepted else
                                     "plate consensus only -- no cross-camera link"),
                "status": ("confirmed" if identity_conf >= threshold else "possible_lead"),
                "display_threshold": threshold,
                "first_seen": sightings[0].first_seen.isoformat(timespec="milliseconds"),
                "last_seen": sightings[-1].last_seen.isoformat(timespec="milliseconds"),
                "cameras": cameras,
                "sighting_count": len(sightings),
                "accepted_links": len(accepted),
                "refused_links": len(refused),
                "colour_hex": next((s.colour_hex for s in sightings if s.colour_hex), None),
                "journey": None if j is None else {
                    "route": j.route.split(">") if j.route else [],
                    "full_route": j.full_route.split(">") if j.full_route else [],
                    "coverage_gaps": [g for g in j.coverage_gaps.split(",") if g],
                    "total_distance_km": j.total_distance_km,
                    "total_duration_s": j.total_duration_s,
                    "average_speed_kmph": j.average_speed_kmph,
                    "confidence": j.confidence,
                },
                "sightings": [_sighting_evidence(s, reads.get(s.id, [])) for s in sightings],
                "links": [_link_evidence(l) for l in links],
            }

    @app.get("/api/leads")
    def leads(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
              camera: str | None = None, plate: str | None = None,
              vehicle_class: str | None = None, status: str | None = None,
              min_score: float = Query(0.0, ge=0.0, le=1.0),
              max_score: float = Query(1.0, ge=0.0, le=1.0)) -> dict:
        """Unclear Possible Leads -- pairs the resolver scored but refused.

        These are NOT identities and must never be shown as confirmed vehicles.
        They are near misses kept deliberately, because the pair a system
        refused is often exactly what an investigator wants to look at.
        """
        if status and status not in LEAD_STATUSES:
            raise HTTPException(422, f"status must be one of {LEAD_STATUSES}")
        with Session() as session:
            rows, total = lead_links(
                session, limit=limit, offset=offset, camera=camera, plate=plate,
                vehicle_class=vehicle_class, status=status,
                min_score=min_score, max_score=max_score)
            out = []
            for link, a, b, review in rows:
                ev = _link_evidence(link)
                ev.update({
                    "status": review.status if review is not None else "unreviewed",
                    "note": review.note if review is not None else "",
                    "why_uncertain": _why_uncertain(link),
                    "from": _sighting_evidence(a),
                    "to": _sighting_evidence(b),
                })
                out.append(ev)
            return {"total": total, "limit": limit, "offset": offset,
                    "statuses": list(LEAD_STATUSES), "leads": out}

    def _why_uncertain(link) -> list[str]:
        """Plain-language account of which signal fell short, from the numbers."""
        why = []
        idc = (cfg.raw.get("system", {}) or {}).get("identity", {}) or {}
        if link.plate_similarity < float(idc.get("min_plate_similarity", 0.72)):
            why.append(f"plate similarity {link.plate_similarity:.2f} below the "
                       f"{float(idc.get('min_plate_similarity', 0.72)):.2f} floor")
        if link.topology_score < 0.5:
            why.append(f"travel time implies {link.implied_speed_kmph:.0f} km/h over "
                       f"{link.distance_km:.1f} km — physically implausible")
        if link.appearance_similarity < 0.5:
            why.append(f"appearance similarity only {link.appearance_similarity:.2f}")
        if link.score < float(idc.get("accept_threshold", 0.55)):
            why.append(f"combined score {link.score:.2f} below the "
                       f"{float(idc.get('accept_threshold', 0.55)):.2f} accept threshold")
        return why or [link.reason or "scored below the acceptance threshold"]

    @app.post("/api/leads/{link_id}/status")
    def lead_status(link_id: int, payload: dict = Body(...)) -> dict:
        """Record an investigator's disposition of a lead."""
        status = str(payload.get("status", "")).strip()
        note = str(payload.get("note", ""))
        if status not in LEAD_STATUSES:
            raise HTTPException(422, f"status must be one of {LEAD_STATUSES}")
        with Session() as session:
            try:
                row = set_lead_status(session, link_id, status, note)
            except ValueError as e:
                raise HTTPException(422, str(e))
            return {"link_id": link_id, "status": row.status, "note": row.note,
                    "updated_at": row.updated_at.isoformat(timespec="seconds")}

    # -- live ---------------------------------------------------------------

    @app.get("/api/live/{camera_id}")
    def live(camera_id: str) -> StreamingResponse:
        known = {c.id.upper() for c in cfg.cameras}
        if camera_id.upper() not in known:
            raise HTTPException(404, f"unknown camera {camera_id}")
        return StreamingResponse(
            mjpeg_stream(cfg.paths.data_dir / "live", camera_id.upper()),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        await event_feed(websocket, Session)

    log.info("api ready: %d cameras, database %s", len(cfg.cameras), cfg.paths.database)
    return app


app = create_app()
