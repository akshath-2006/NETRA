"""The camera worker -- one camera, end to end.

This is the function Milestone 6 will run in its own OS process, once per
camera. It is written as a plain callable now so the CLI can drive a single
camera today and the supervisor can drive N tomorrow without the loop moving.

Note what it does *not* do: it never touches the database. It publishes to the
bus and moves on. That is the boundary that lets the same worker run on an edge
box later, with a Kafka bus in place of the local one.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core.bus import EventBus, Topics
from app.core.config import AppConfig, CameraConfig
from app.core.logging import get_logger
from app.core.timing import FpsMeter


@dataclass
class WorkerStats:
    camera_id: str
    frames: int = 0
    sightings: int = 0
    crossings: int = 0
    plate_reads: int = 0
    plates: int = 0
    # ANPR accounting, so the cost and the gating are visible rather than felt.
    anpr_calls: int = 0        # crops actually sent to the engine
    anpr_skipped: int = 0      # crops skipped because the track had enough
    plate_too_small: int = 0   # reads discarded below min_plate_width_px
    snapshots: int = 0         # evidence images written, one per sighting
    elapsed_s: float = 0.0
    counts: dict = field(default_factory=dict)
    by_class: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0


def new_run_id() -> str:
    """One id per process run. Track ids restart at 1 every run, so this is
    what stops a second run colliding with the first in the database."""
    return uuid.uuid4().hex[:12]


def run_camera(
    cfg: AppConfig,
    camera: CameraConfig,
    bus: EventBus,
    *,
    mode: str | None = None,
    max_frames: int | None = None,
    seconds: float | None = None,
    show: bool = False,
    save: bool = False,
    live_dir=None,
    progress_every: int = 25,
    run_id: str | None = None,
) -> WorkerStats:
    """Read one camera, detect, track, count, and publish what it finds."""
    import cv2

    from app.vision.anpr import NullEngine, build_anpr
    from app.vision.consensus import PlateReading, build_consensus
    from app.vision.counting import LineCounter
    from app.vision.detector import build_detector
    from app.vision.overlay import draw_counting_line, draw_hud, draw_tracks
    from app.vision.tracker import build_track_manager
    from app.vision.video_source import VideoSource

    log = get_logger(f"worker.{camera.id}")

    sysd = cfg.raw.get("system", {}) or {}
    perf = sysd.get("performance", {}) or {}
    run_mode = mode or cfg.runtime.mode

    # One process per camera means N processes each defaulting to "use every
    # core", which on a 2-4 core laptop makes them fight and all run slower.
    # Capping threads per worker is the single biggest win on a small machine.
    threads = int(perf.get("torch_threads_per_worker", 1) or 0)
    if threads > 0:
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:  # torch missing or refusing -- not fatal
            log.debug("could not set torch threads", exc_info=True)

    detector = build_detector(cfg)
    detector.load()
    manager = build_track_manager(cfg, camera.id)

    anpr_cfg = sysd.get("anpr", {}) or {}
    anpr = build_anpr(cfg)
    anpr.load()
    anpr_on = not isinstance(anpr, NullEngine)
    every_n = int(anpr_cfg.get("every_n_frames", 3))
    min_vehicle_px = int(anpr_cfg.get("min_vehicle_width_px", 90))
    min_plate_px = int(anpr_cfg.get("min_plate_width_px", 175))
    crop_pad = float(anpr_cfg.get("crop_pad_frac", 0.15))
    max_reads = int(anpr_cfg.get("max_reads_per_track", 10))
    good_conf = float(anpr_cfg.get("good_read_confidence", 0.55))
    reopen_gain = float(anpr_cfg.get("reopen_width_gain", 1.25))
    confirm_conf = float(anpr_cfg.get("confirm_confidence", 0.60))
    tentative_conf = float(anpr_cfg.get("tentative_confidence", 0.40))

    # Nobody is watching a fast-mode seeding run, so the preview and the
    # overlay that feeds it are pure cost. Measured at ~3% combined.
    fast = run_mode == "fast"
    want_preview = (not fast) or bool(perf.get("live_preview_in_fast_mode", False))
    want_overlay = (not fast) or bool(perf.get("overlay_in_fast_mode", False))

    source = VideoSource(camera, mode=run_mode, max_frames=max_frames)

    # Live preview: the worker writes its latest annotated frame as a JPEG and
    # the API process serves it as MJPEG. Crude, but it crosses a process
    # boundary with no IPC to go wrong mid-demo -- and in a real deployment
    # this is where an RTSP or WebRTC relay would sit instead.
    # Evidence snapshots. One JPEG per completed sighting, named deterministically
    # from (run_id, camera, track) so the API can find it from a database row
    # without a schema migration. Measured at ~10-15 KB per sighting.
    ev_cfg = sysd.get("evidence", {}) or {}
    keep_snaps = bool(ev_cfg.get("keep_snapshots", True)) and run_id is not None
    snap_quality = int(ev_cfg.get("snapshot_jpeg_quality", 80))
    snap_max_w = int(ev_cfg.get("snapshot_max_width_px", 480))
    snap_dir = None
    if keep_snaps:
        snap_dir = Path(cfg.paths.data_dir) / "evidence" / "sightings"
        snap_dir.mkdir(parents=True, exist_ok=True)

    live_path = None
    if live_dir is not None and want_preview:
        live_dir = Path(live_dir)
        live_dir.mkdir(parents=True, exist_ok=True)
        live_path = live_dir / f"{camera.id}.jpg"

    stats = WorkerStats(camera_id=camera.id)
    meter = FpsMeter()
    counter = None
    writer = None
    window = f"NETRA - {camera.id}"
    started = time.perf_counter()

    def read_plates(detections, frame) -> None:
        """OCR the vehicle crops, not the whole frame.

        Cropping first is both faster and more accurate: the plate detector
        looks at a region that definitely holds one vehicle, instead of a whole
        junction. Small vehicles are skipped outright -- a small plate wastes
        the call and poisons the vote with noise.

        Three things happen here that did not before:

        * the vehicle box is **padded** before cropping, because YOLO's box is
          often tight enough to clip a plate sitting at the vehicle's edge;
        * a track with enough good reads is **skipped**, unless a materially
          wider plate turns up that could correct an earlier truncated read;
        * reads whose plate is below ``min_plate_width_px`` are **discarded**
          rather than stored. They are not merely weak -- below that width the
          OCR returns confidently truncated strings, and because there are more
          of them than good reads they win the length vote and bury the truth.
        """
        h, w = frame.image.shape[:2]
        for det in detections:
            tid = getattr(det, "track_id", None)
            if tid is None or det.width < min_vehicle_px:
                continue
            track = manager.active.get(int(tid))
            if track is None:
                continue

            # Budget: enough good evidence already, and nothing better on offer.
            if max_reads and track.anpr_good_reads >= max_reads:
                if det.width < track.best_plate_width * reopen_gain:
                    stats.anpr_skipped += 1
                    continue

            x1, y1, x2, y2 = det.box
            px = int((x2 - x1) * crop_pad)
            py = int((y2 - y1) * crop_pad)
            crop = frame.image[max(0, y1 - py):min(h, y2 + py),
                               max(0, x1 - px):min(w, x2 + px)]
            if crop.size == 0:
                continue

            stats.anpr_calls += 1
            for hit in anpr.read(crop):
                if not hit.text:
                    continue
                if hit.width_px < min_plate_px:
                    stats.plate_too_small += 1
                    continue
                track.plate_readings.append(PlateReading(
                    text=hit.text,
                    confidence=hit.confidence,
                    frame_index=frame.index,
                    plate_width_px=hit.width_px,
                    char_confidence=hit.char_confidence,
                ))
                stats.plate_reads += 1
                if hit.width_px > track.best_plate_width:
                    track.best_plate_width = hit.width_px
                if hit.confidence >= good_conf:
                    track.anpr_good_reads += 1

    def save_snapshot(track) -> None:
        """Write the one representative frame for a finished sighting.

        Only the best-confidence crop, only once, only when the track closes.
        Not every frame -- that would be gigabytes and would tell an
        investigator nothing the best frame does not.
        """
        if snap_dir is None or track.best_crop is None:
            return
        img = track.best_crop
        h, w = img.shape[:2]
        if w > snap_max_w:                      # cap the long edge; plates stay legible
            scale = snap_max_w / float(w)
            img = cv2.resize(img, (snap_max_w, max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        name = f"{run_id}_{camera.id}_{track.track_id}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), snap_quality])
        if ok:
            (snap_dir / name).write_bytes(buf.tobytes())
            track.snapshot = name
            stats.snapshots += 1
        track.best_crop = None                  # release the memory either way

    def publish_tracks(tracks) -> None:
        for track in tracks:
            save_snapshot(track)
            # One answer, voted out of every read taken while the vehicle was
            # in view. This is the "we never trust a single frame" step.
            if track.plate_readings:
                track.plate = build_consensus(
                    track.plate_readings,
                    min_confidence=float(anpr_cfg.get("min_read_confidence", 0.30)),
                    min_reads=int(anpr_cfg.get("min_reads", 2)),
                    target_reads=int(anpr_cfg.get("target_reads", 8)),
                    confirm_confidence=confirm_conf,
                    tentative_confidence=tentative_conf,
                )
                if track.plate.ok:
                    stats.plates += 1
                    bus.publish(Topics.PLATE_READ, {
                        "camera_id": camera.id,
                        "track_id": track.track_id,
                        "plate": track.plate.text,
                        "confidence": track.plate.confidence,
                        "explain": track.plate.explain(),
                        "at": track.first_seen,
                    }, source=camera.id)
                    log.info("track %d -> plate %s (%.2f)  %s",
                             track.track_id, track.plate.text,
                             track.plate.confidence, track.plate.explain())

            bus.publish(Topics.SIGHTING_COMPLETED,
                        {"track": track, "sighting": track.to_sighting()},
                        source=camera.id)
            stats.sightings += 1

    try:
        with source:
            for frame in source.frames():
                detections = detector.track(frame.image)
                active = manager.update(detections, frame)
                if anpr_on and every_n > 0 and frame.index % every_n == 0:
                    read_plates(detections, frame)
                publish_tracks(manager.drain_completed())
                meter.tick()
                stats.frames += 1

                if counter is None and camera.counting_line is not None:
                    h, w = frame.image.shape[:2]
                    counter = LineCounter(camera.counting_line, w, h, camera.id)
                if counter is not None:
                    for crossing in counter.update(manager.stable_active):
                        stats.crossings += 1
                        bus.publish(Topics.CROSSING, {
                            "camera_id": camera.id,
                            "track_id": crossing.track_id,
                            "vehicle_class": crossing.label,
                            "direction": crossing.direction,
                            "at": crossing.at,
                        }, source=camera.id)

                # Drawing only pays for itself if something will look at it.
                if want_overlay and (show or save or live_path is not None):
                    draw_tracks(frame.image, detections, active)
                    if counter is not None:
                        draw_counting_line(frame.image, counter)
                    draw_hud(
                        frame.image,
                        title=f"{camera.id}  {camera.name}",
                        subtitle=f"{stats.sightings} sightings  |  {stats.plates} plates  |  "
                                 f"{stats.plate_reads} ocr reads",
                        stats=(("FRAME", str(frame.index)),
                               ("CLOCK", frame.timestamp.strftime("%H:%M:%S")),
                               ("ACTIVE", str(len(manager.stable_active))),
                               ("PLATES", str(stats.plates)),
                               ("FPS", f"{meter.fps:.1f}")),
                        hint="q or esc to quit",
                    )

                if live_path is not None:
                    ok, buf = cv2.imencode(".jpg", frame.image,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    if ok:
                        # write-then-rename so a reader never sees half a frame
                        tmp = live_path.with_suffix(".tmp")
                        tmp.write_bytes(buf.tobytes())
                        tmp.replace(live_path)

                if save:
                    if writer is None:
                        out = cfg.paths.data_dir / "out"
                        out.mkdir(parents=True, exist_ok=True)
                        h, w = frame.image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out / f"{camera.id.lower()}_process.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), camera.target_fps, (w, h))
                    writer.write(frame.image)

                if show:
                    cv2.imshow(window, frame.image)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
                elif progress_every and stats.frames % progress_every == 0:
                    log.info("frame %-5d active %-3d sightings %-4d fps %.1f",
                             frame.index, len(manager.stable_active),
                             stats.sightings, meter.fps)

                if seconds and (time.perf_counter() - started) >= seconds:
                    break
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        # Vehicles still in frame when the stream ends are real sightings too.
        publish_tracks(manager.flush())
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    stats.elapsed_s = time.perf_counter() - started
    if counter is not None:
        stats.counts = dict(counter.counts)
        stats.by_class = dict(counter.by_class)
    return stats
