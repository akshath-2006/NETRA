"""NETRA command line.

    python -m app.cli cameras
    python -m app.cli make-sample
    python -m app.cli preview --camera CAM01

Heavy imports (OpenCV, later YOLO) happen inside the subcommands, so that
listing cameras stays instant and a broken vision dependency never stops you
inspecting configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dataclasses import replace

from app.core.config import AppConfig, CameraConfig, ConfigError, load_config
from app.core.logging import get_logger, setup_logging
from app.core.timing import FpsMeter

log = get_logger("cli")



def _camera_from(args: argparse.Namespace, cfg: AppConfig) -> CameraConfig:
    """Resolve the camera, applying a --source override if one was given.

    cameras.yaml stays the source of truth. --source exists so you can point a
    command at any clip on disk while testing, without editing config.
    """
    camera = cfg.get_camera(args.camera)
    if getattr(args, "source", None):
        path = Path(args.source).expanduser().resolve()
        if not path.exists():
            raise ConfigError(f"--source file not found: {path}")
        camera = replace(camera, source=str(path), loop=False)
        log.info("source overridden -> %s", path.name)
    return camera


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def cmd_cameras(args: argparse.Namespace, cfg: AppConfig) -> int:
    if not cfg.cameras:
        print("no cameras configured -- see configs/cameras.yaml")
        return 1

    headers = ("ID", "NAME", "SOURCE", "FPS", "OFFSET", "STATUS")
    rows = []
    for cam in cfg.cameras:
        source = cam.source if cam.is_stream else cam.source.split("/")[-1]
        exists = cam.is_stream or Path(cam.source).exists()
        status = "enabled" if cam.enabled else "disabled"
        if not exists:
            status = "MISSING FILE"
        rows.append((cam.id, cam.name, source, f"{cam.target_fps:g}",
                     f"+{cam.clock_offset_s:g}s", status))

    widths = [max(len(str(r[i])) for r in (headers, *rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))

    missing = [r for r in rows if r[-1] == "MISSING FILE"]
    if missing:
        print(f"\n{len(missing)} source file(s) missing -- run 'make sample' "
              f"to generate test footage.")
    return 0


# ---------------------------------------------------------------------------
# make-sample
# ---------------------------------------------------------------------------

def cmd_make_sample(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.vision.sample_video import generate_sample_set

    print("Generating SYNTHETIC test footage. This is scaffolding so the")
    print("pipeline can be built before real footage exists -- it is never")
    print("used in the demo and no accuracy number comes from it.\n")

    paths = generate_sample_set(cfg.paths.video_dir, seconds=args.seconds)
    print()
    for p in paths:
        print(f"  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    print("\nnext:  make preview CAM=CAM01")
    return 0


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def cmd_preview(args: argparse.Namespace, cfg: AppConfig) -> int:
    import cv2
    from app.vision.overlay import draw_hud
    from app.vision.video_source import VideoSource, VideoSourceError

    try:
        camera = _camera_from(args, cfg)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mode = args.mode or cfg.runtime.mode
    source = VideoSource(camera, mode=mode, max_frames=args.max_frames)
    meter = FpsMeter()
    window = f"NETRA - {camera.id}"
    started = time.perf_counter()
    shown = 0

    try:
        with source:
            for frame in source.frames():
                meter.tick()
                shown += 1

                if not args.no_window:
                    draw_hud(
                        frame.image,
                        title=f"{camera.id}  {camera.name}",
                        subtitle=f"source {frame.source_index}  |  lap {frame.lap}  |  mode {mode}",
                        stats=(
                            ("FRAME", f"{frame.index}"),
                            ("CLOCK", frame.timestamp.strftime("%H:%M:%S.") +
                                      f"{frame.timestamp.microsecond // 1000:03d}"),
                            ("FPS", f"{meter.fps:.1f}"),
                        ),
                        hint="q or esc to quit",
                    )
                    cv2.imshow(window, frame.image)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        log.info("quit requested")
                        break
                elif shown % 12 == 0:
                    log.info("frame %-5d clock %s  fps %.1f",
                             frame.index, frame.timestamp.strftime("%H:%M:%S.%f")[:-3],
                             meter.fps)

                if args.seconds and (time.perf_counter() - started) >= args.seconds:
                    log.info("reached --seconds %.1f", args.seconds)
                    break
    except VideoSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        log.info("interrupted")
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"\n{shown} frames in {elapsed:.1f}s  ->  {shown / elapsed:.1f} fps average"
          if elapsed > 0 else f"\n{shown} frames")
    return 0


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace, cfg: AppConfig) -> int:
    import cv2
    from app.vision.detector import build_detector
    from app.vision.overlay import draw_detections, draw_hud, summarise
    from app.vision.video_source import VideoSource, VideoSourceError

    try:
        camera = _camera_from(args, cfg)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    detector = build_detector(cfg)
    if args.model:
        detector.model_name = args.model
    if args.conf is not None:
        detector.confidence = args.conf
    detector.load()

    mode = args.mode or cfg.runtime.mode
    source = VideoSource(camera, mode=mode, max_frames=args.max_frames)
    meter = FpsMeter()
    window = f"NETRA detect - {camera.id}"
    writer = None
    started = time.perf_counter()
    shown = 0
    total_detections = 0
    ms_total = 0.0

    try:
        with source:
            for frame in source.frames():
                detections = detector.detect(frame.image)
                meter.tick()
                shown += 1
                total_detections += len(detections)
                ms_total += detector.last_inference_ms

                draw_detections(frame.image, detections)
                draw_hud(
                    frame.image,
                    title=f"{camera.id}  {camera.name}",
                    subtitle=f"{detector.model_name} on {detector.device}  |  "
                             f"{summarise(detections)}",
                    stats=(
                        ("FRAME", f"{frame.index}"),
                        ("CLOCK", frame.timestamp.strftime("%H:%M:%S.") +
                                  f"{frame.timestamp.microsecond // 1000:03d}"),
                        ("OBJECTS", f"{len(detections)}"),
                        ("INFER", f"{detector.last_inference_ms:.0f}ms"),
                        ("FPS", f"{meter.fps:.1f}"),
                    ),
                    hint="q or esc to quit",
                )

                if args.save:
                    if writer is None:
                        out = cfg.paths.data_dir / "out"
                        out.mkdir(parents=True, exist_ok=True)
                        path = out / f"{camera.id.lower()}_detect.mp4"
                        h, w = frame.image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                            camera.target_fps, (w, h))
                        log.info("writing annotated video to %s", path)
                    writer.write(frame.image)

                if not args.no_window:
                    cv2.imshow(window, frame.image)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        log.info("quit requested")
                        break
                elif shown % 12 == 0:
                    log.info("frame %-5d objects %-3d infer %5.0fms  fps %.1f",
                             frame.index, len(detections),
                             detector.last_inference_ms, meter.fps)

                if args.seconds and (time.perf_counter() - started) >= args.seconds:
                    break
    except VideoSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        log.info("interrupted")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    if shown:
        print(f"\n{shown} frames  |  {total_detections} detections "
              f"({total_detections / shown:.1f} per frame)  |  "
              f"{ms_total / shown:.0f} ms average inference  |  "
              f"{shown / (time.perf_counter() - started):.1f} fps end to end")
    return 0


# ---------------------------------------------------------------------------
# track
# ---------------------------------------------------------------------------

def cmd_track(args: argparse.Namespace, cfg: AppConfig) -> int:
    import cv2
    from app.vision.counting import LineCounter
    from app.vision.detector import build_detector
    from app.vision.overlay import draw_counting_line, draw_hud, draw_tracks
    from app.vision.tracker import build_track_manager
    from app.vision.video_source import VideoSource, VideoSourceError

    try:
        camera = _camera_from(args, cfg)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    detector = build_detector(cfg)
    if args.conf is not None:
        detector.confidence = args.conf
    detector.load()

    manager = build_track_manager(cfg, camera.id)
    mode = args.mode or cfg.runtime.mode
    source = VideoSource(camera, mode=mode, max_frames=args.max_frames)
    meter = FpsMeter()
    window = f"NETRA track - {camera.id}"
    counter = None
    writer = None
    started = time.perf_counter()
    shown = 0
    finished: list = []

    try:
        with source:
            for frame in source.frames():
                detections = detector.track(frame.image)
                active = manager.update(detections, frame)
                finished.extend(manager.drain_completed())
                meter.tick()
                shown += 1

                if counter is None and camera.counting_line is not None:
                    h, w = frame.image.shape[:2]
                    counter = LineCounter(camera.counting_line, w, h, camera.id)
                if counter is not None:
                    counter.update(manager.stable_active)

                draw_tracks(frame.image, detections, active)
                if counter is not None:
                    draw_counting_line(frame.image, counter)

                stats = [
                    ("FRAME", f"{frame.index}"),
                    ("CLOCK", frame.timestamp.strftime("%H:%M:%S.") +
                              f"{frame.timestamp.microsecond // 1000:03d}"),
                    ("ACTIVE", f"{len(manager.stable_active)}"),
                    ("TRACKS", f"{manager.total_created}"),
                ]
                if counter is not None:
                    for name, value in counter.counts.items():
                        stats.append((name.upper()[:8], str(value)))
                stats.append(("FPS", f"{meter.fps:.1f}"))

                draw_hud(
                    frame.image,
                    title=f"{camera.id}  {camera.name}",
                    subtitle=f"{detector.tracker.replace('.yaml','')} on {detector.device}"
                             f"  |  {manager.total_created} tracks seen",
                    stats=tuple(stats),
                    hint="q or esc to quit",
                )

                if args.save:
                    if writer is None:
                        out = cfg.paths.data_dir / "out"
                        out.mkdir(parents=True, exist_ok=True)
                        path = out / f"{camera.id.lower()}_track.mp4"
                        h, w = frame.image.shape[:2]
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 camera.target_fps, (w, h))
                        log.info("writing annotated video to %s", path)
                    writer.write(frame.image)

                if not args.no_window:
                    cv2.imshow(window, frame.image)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
                elif shown % 25 == 0:
                    log.info("frame %-5d active %-3d total %-3d  fps %.1f",
                             frame.index, len(manager.stable_active),
                             manager.total_created, meter.fps)

                if args.seconds and (time.perf_counter() - started) >= args.seconds:
                    break
    except VideoSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        log.info("interrupted")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    finished.extend(manager.flush())
    _print_track_report(finished, counter, shown, time.perf_counter() - started)
    return 0


def _print_track_report(tracks, counter, frames: int, elapsed: float) -> None:
    """The evidence that tracking worked: one row per completed track."""
    print()
    if not tracks:
        print("no completed tracks. if the feed had vehicles, check that "
              "detection is finding them first (app.cli detect).")
        return

    print(f"{'ID':>5}  {'CLASS':<11} {'CONF':>5}  {'FRAMES':>6}  {'SECONDS':>7}  CROSSED")
    print("-" * 60)
    for t in sorted(tracks, key=lambda x: x.first_frame):
        print(f"{t.track_id:>5}  {t.label:<11} {t.label_confidence:>5.2f}  "
              f"{t.frames:>6}  {t.duration_s:>7.2f}  {t.crossing or '-'}")

    lengths = [t.frames for t in tracks]
    print("-" * 60)
    print(f"{len(tracks)} tracks  |  median length {sorted(lengths)[len(lengths)//2]} frames  "
          f"|  {frames} frames in {elapsed:.1f}s ({frames/elapsed:.1f} fps)")
    if counter is not None:
        print(f"line count: {counter.summary()}  (total {counter.total})")
        if counter.by_class:
            print("by class:   " + "  ".join(f"{k} {v}" for k, v in sorted(counter.by_class.items())))


# ---------------------------------------------------------------------------
# process  -- the real pipeline: detect, track, count, publish, persist
# ---------------------------------------------------------------------------

def cmd_process(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.core.bus import LocalBus, Topics
    from app.pipeline.camera_worker import new_run_id, run_camera
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import upsert_cameras
    from app.store.subscriber import SightingWriter

    try:
        camera = _camera_from(args, cfg)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)
    with Session() as session:
        upsert_cameras(session, cfg.cameras)

    run_id = args.run_id or new_run_id()
    bus = LocalBus(camera.id)
    store = SightingWriter(Session, run_id).register(bus)

    # A second subscriber, to make the point that the bus is a real fan-out and
    # not a disguised function call. In M6 this becomes the WebSocket feed.
    seen: list = []
    bus.subscribe(Topics.CROSSING, lambda e: seen.append(e))

    log.info("run_id %s", run_id)
    stats = run_camera(cfg, camera, bus,
                       mode=args.mode, max_frames=args.max_frames,
                       seconds=args.seconds, show=not args.no_window, save=args.save,
                       run_id=run_id)

    print(f"\ncamera        {stats.camera_id}  ({camera.name})")
    print(f"run_id        {run_id}")
    print(f"frames        {stats.frames} in {stats.elapsed_s:.1f}s ({stats.fps:.1f} fps)")
    print(f"events        {bus.published} published on the bus")
    print(f"sightings     {store.written} written, {store.skipped} duplicates skipped")
    print(f"crossings     {stats.crossings}  {stats.counts or ''}")
    print(f"ocr reads     {stats.plate_reads} taken, {stats.plates} plates agreed")
    if stats.by_class:
        print("by class      " + "  ".join(f"{k} {v}" for k, v in sorted(stats.by_class.items())))
    print(f"\ndatabase      {cfg.paths.database}")
    print("inspect with  python -m app.cli db stats")
    return 0


# ---------------------------------------------------------------------------
# db
# ---------------------------------------------------------------------------

def cmd_db(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import (find_by_plate, list_cameras,
                                      recent_sightings, stats, upsert_cameras)

    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)

    if args.db_command == "init":
        with Session() as session:
            n = upsert_cameras(session, cfg.cameras)
        print(f"schema ready at {cfg.paths.database}")
        print(f"{n} cameras mirrored from cameras.yaml")
        return 0

    if args.db_command == "stats":
        with Session() as session:
            s = stats(session)
        print(f"database      {cfg.paths.database}")
        print(f"cameras       {s['cameras']}")
        print(f"sightings     {s['sightings']}  ({s['with_plate']} with a plate)")
        print(f"plate reads   {s['plate_reads']}")
        if s["first_seen"]:
            print(f"time window   {s['first_seen']:%Y-%m-%d %H:%M:%S}  ->  "
                  f"{s['last_seen']:%H:%M:%S}")
        for title, data in (("by camera", s["by_camera"]), ("by class", s["by_class"]),
                            ("by direction", s["by_crossing"])):
            if data:
                print(f"{title:<13} " + "  ".join(f"{k} {v}" for k, v in data.items()))
        return 0

    if args.db_command == "cameras":
        with Session() as session:
            for cam in list_cameras(session):
                flag = "on " if cam.enabled else "off"
                print(f"{cam.id}  [{flag}]  {cam.name:<38} {cam.lat:.5f}, {cam.lon:.5f}")
        return 0

    if args.db_command == "sightings":
        with Session() as session:
            rows = recent_sightings(session, limit=args.limit, camera_id=args.camera)
        if not rows:
            print("no sightings yet -- run: python -m app.cli process --camera CAM01")
            return 0
        print(f"{'CAMERA':<8} {'TRK':>4}  {'CLASS':<11} {'FIRST SEEN':<13} "
              f"{'SECS':>5}  {'PLATE':<11} CROSSED")
        print("-" * 74)
        for r in rows:
            ts = r.first_seen.strftime("%H:%M:%S.") + f"{r.first_seen.microsecond // 1000:03d}"
            print(f"{r.camera_id:<8} {r.track_id:>4}  {r.vehicle_class:<11} {ts:<13} "
                  f"{r.duration_s:>5.1f}  {(r.plate_text or '-'):<11} {r.crossing or '-'}")
        return 0

    if args.db_command == "identities":
        from app.store.repository import identity_links, list_identities
        with Session() as session:
            rows = list_identities(session, limit=args.limit, min_cameras=args.min_cameras)
            if not rows:
                print("no identities yet -- run: python -m app.cli resolve")
                return 0
            print(f"{'KEY':<8} {'PLATE':<12} {'CONF':>5} {'CAMS':>5} {'SEEN':>5}  ROUTE")
            print("-" * 70)
            for r in rows:
                print(f"{r.key:<8} {(r.plate or '-'):<12} {r.confidence:>5.2f} "
                      f"{r.camera_count:>5} {r.sighting_count:>5}  {r.cameras}")
                if args.verbose:
                    for link in identity_links(session, r.id):
                        print(f"         {link.from_camera} -> {link.to_camera}  "
                              f"{link.gap_s/60:>5.1f} min  {link.distance_km:.1f} km  "
                              f"{link.implied_speed_kmph:.0f} km/h  score {link.score:.2f}")
        return 0

    if args.db_command == "rejects":
        from app.store.repository import rejected_links
        with Session() as session:
            rows = rejected_links(session, limit=args.limit)
        if not rows:
            print("no recorded near-misses. run 'resolve' first.")
            return 0
        print("links refused despite similar plates:\n")
        for r in rows:
            print(f"  {r.from_camera} -> {r.to_camera}   plate similarity "
                  f"{r.plate_similarity:.2f}")
            print(f"    {r.reason}")
        return 0

    if args.db_command == "plate":
        with Session() as session:
            rows = find_by_plate(session, args.value)
        if not rows:
            print(f"no sightings for plate {args.value.upper()}")
            return 0
        print(f"plate {args.value.upper()} seen {len(rows)} time(s):")
        for r in rows:
            print(f"  {r.camera_id}  {r.first_seen:%H:%M:%S}  {r.vehicle_class}  "
                  f"conf {r.plate_confidence or 0:.2f}")
        return 0

    print(f"unknown db command {args.db_command!r}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# run  -- all cameras at once, one process each
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.pipeline.camera_worker import new_run_id
    from app.pipeline.supervisor import run_supervisor
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import upsert_cameras

    cameras = cfg.enabled_cameras
    if args.cameras:
        wanted = {c.strip().upper() for c in args.cameras.split(",")}
        cameras = [c for c in cameras if c.id.upper() in wanted]
    if not cameras:
        print("no enabled cameras to run -- check configs/cameras.yaml", file=sys.stderr)
        return 2

    init_db(cfg.paths.database)
    with get_sessionmaker(cfg.paths.database)() as session:
        upsert_cameras(session, cfg.cameras)

    run_id = args.run_id or new_run_id()
    print(f"starting {len(cameras)} camera worker(s): "
          f"{', '.join(c.id for c in cameras)}")
    print(f"run_id {run_id}   (ctrl-c to stop)\n")

    run_supervisor(cfg, cameras, run_id, workers=getattr(args, "workers", None),
                   mode=args.mode, seconds=args.seconds,
                   max_frames=args.max_frames,
                   log_level=args.log_level or cfg.runtime.log_level)
    print("\nall workers stopped. inspect with: python -m app.cli db stats")
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace, cfg: AppConfig) -> int:
    import uvicorn

    print(f"api      http://{args.host}:{args.port}")
    print(f"docs     http://{args.host}:{args.port}/docs")
    print(f"database {cfg.paths.database}\n")
    uvicorn.run("app.api.main:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")
    return 0


# ---------------------------------------------------------------------------
# resolve  -- cross-camera identity resolution
# ---------------------------------------------------------------------------

def build_resolver(cfg):
    from app.core.city_graph import load_city_graph
    from app.identity.resolver import IdentityResolver

    graph = load_city_graph(cfg.paths.root / "configs" / "city_graph.yaml", cfg.cameras)
    i = (cfg.raw.get("system", {}) or {}).get("identity", {}) or {}
    return IdentityResolver(
        graph,
        weight_plate=i.get("weight_plate", 0.60),
        weight_appearance=i.get("weight_appearance", 0.20),
        weight_topology=i.get("weight_topology", 0.20),
        accept_threshold=i.get("accept_threshold", 0.55),
        plateless_cap=i.get("plateless_cap", 0.50),
        require_plate=i.get("require_plate", True),
        min_plate_similarity=i.get("min_plate_similarity", 0.72),
    ), graph


def cmd_resolve(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import (list_identities, save_journeys,
                                      save_resolution, sighting_views)
    from app.trajectory.engine import TrajectoryEngine

    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)
    resolver, graph = build_resolver(cfg)

    with Session() as session:
        views = sighting_views(session)
    if not views:
        print("no sightings to resolve -- run: python -m app.cli run")
        return 0

    identities, verdicts = resolver.resolve(views)

    # Identities are a set of sightings; journeys are the ordered trips through
    # them. Built here so both always agree.
    engine = TrajectoryEngine(graph)
    journeys = engine.build_all(identities, {v.id: v for v in views}, verdicts)

    with Session() as session:
        summary = save_resolution(session, identities, verdicts)
        keys = {r.key: r.id for r in list_identities(session, limit=10000)}
        saved = save_journeys(session, journeys, keys)

    multi_camera = [i for i in identities if len(i.cameras) > 1]
    print(f"\ngraph         {graph.summary()}")
    print(f"sightings     {len(views)}")
    print(f"identities    {summary['identities']}  "
          f"({len(multi_camera)} seen at more than one camera)")
    print(f"links         {summary['accepted']} accepted, "
          f"{summary['rejected']} rejected near-misses recorded")
    print(f"journeys      {saved} reconstructed")

    gaps = [j for j in journeys if j.coverage_gaps]
    if gaps:
        print(f"\ncoverage gaps -- cameras on the route that never saw the vehicle:")
        for j in gaps[:args.limit]:
            print(f"  {j.plate or j.identity_key:<12} {' -> '.join(j.route)}  "
                  f"passed {', '.join(j.coverage_gaps)} unobserved")

    if multi_camera:
        print(f"\n{'PLATE':<12} {'CONF':>5}  {'CAMS':>4}  ROUTE")
        print("-" * 62)
        for ident in multi_camera[:args.limit]:
            print(f"{(ident.plate or '-'):<12} {ident.confidence:>5.2f}  "
                  f"{len(ident.cameras):>4}  {' -> '.join(ident.cameras)}")

    rejects = [v for v in verdicts if not v.accepted and v.plate_similarity >= 0.7]
    if rejects:
        print(f"\nrejected despite similar plates -- the reasoning, visible:")
        for v in sorted(rejects, key=lambda x: -x.plate_similarity)[:args.limit]:
            print(f"  {v.from_camera} -> {v.to_camera}  "
                  f"{v.plate_pair[0]} / {v.plate_pair[1]}  {v.reason}")

    print("\ninspect with  python -m app.cli db identities")
    return 0


# ---------------------------------------------------------------------------
# journey
# ---------------------------------------------------------------------------

def cmd_journey(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import journey_for_plate, list_journeys

    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)

    with Session() as session:
        rows = (journey_for_plate(session, args.plate) if args.plate
                else list_journeys(session, limit=args.limit, min_hops=1))
        if not rows:
            target = f"plate {args.plate.upper()}" if args.plate else "any vehicle"
            print(f"no journeys for {target}. run: python -m app.cli resolve")
            return 0

        for row in rows:
            mins, secs = divmod(int(row.total_duration_s), 60)
            print(f"\n{row.plate or row.identity_key}"
                  f"   confidence {row.confidence:.2f}")
            print(f"{'-' * 66}")
            for hop in row.hops:
                via = f"  via {hop.via} (unobserved)" if hop.via else ""
                flag = "  CONGESTED" if hop.delay_ratio >= 1.5 else ""
                print(f"  {hop.departed_at:%H:%M:%S}  {hop.from_camera}"
                      f" -> {hop.to_camera}{via}")
                print(f"            {hop.gap_s/60:>5.1f} min   {hop.distance_km:>5.1f} km   "
                      f"{hop.implied_speed_kmph:>3.0f} km/h "
                      f"(usual {hop.typical_speed_kmph:.0f})   "
                      f"score {hop.confidence:.2f}{flag}")
            print(f"{'-' * 66}")
            print(f"  route     {row.full_route.replace('>', ' -> ')}")
            print(f"  total     {row.total_distance_km:.1f} km   "
                  f"{mins} min {secs:02d} s   {row.average_speed_kmph:.0f} km/h average")
            if row.coverage_gaps:
                print(f"  gaps      {row.coverage_gaps} on the route but never saw it")
    return 0


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------

def cmd_analytics(args: argparse.Namespace, cfg: AppConfig) -> int:
    from app.analytics.alerts import AlertEngine, load_watchlist
    from app.analytics.metrics import build_report
    from app.core.city_graph import load_city_graph
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import (all_sightings, hops_with_plate,
                                      list_journeys, save_alerts)

    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)
    graph = load_city_graph(cfg.paths.root / "configs" / "city_graph.yaml", cfg.cameras)
    system = cfg.raw.get("system", {}) or {}
    acfg = system.get("analytics", {}) or {}
    alcfg = system.get("alerts", {}) or {}

    with Session() as session:
        sightings = all_sightings(session)
        journeys = list_journeys(session, limit=10000, min_hops=1)
        hops = hops_with_plate(session)

    if not sightings:
        print("nothing to analyse -- run: python -m app.cli run")
        return 0

    typical, limits = {}, {}
    for a in cfg.cameras:
        for b in cfg.cameras:
            if a.id == b.id:
                continue
            edge = graph.edge(a.id, b.id)
            if edge and not edge.derived:
                typical[f"{a.id}->{b.id}"] = edge.typical_speed_kmph
                limits[f"{a.id}->{b.id}"] = edge.speed_limit_kmph

    report = build_report(sightings, journeys, hops,
                          bucket_minutes=int(acfg.get("bucket_minutes", 5)),
                          typical_speeds=typical)

    engine = AlertEngine(
        congestion_index=float(alcfg.get("congestion_index", 1.5)),
        min_trips_for_congestion=int(alcfg.get("min_trips_for_congestion", 3)),
        coverage_gap_count=int(alcfg.get("coverage_gap_count", 2)),
        speed_limits=limits,
        watchlist=load_watchlist(cfg.paths.root / "configs" / "watchlist.yaml"),
    )
    alerts = engine.evaluate(report, journeys, hops, sightings)
    with Session() as session:
        save_alerts(session, alerts)

    span = ""
    if report.window_start:
        span = (f"{report.window_start:%H:%M:%S} -> {report.window_end:%H:%M:%S}")
    print(f"\nwindow        {span}")
    print(f"sightings     {report.total_sightings}   "
          f"plates {report.plates_read} ({report.plate_rate:.0%})")
    print(f"journeys      {report.total_journeys}")
    print(f"vehicle mix   " + "  ".join(f"{k} {v}" for k, v in report.vehicle_mix.items()))

    print(f"\nCAMERA UTILISATION")
    print(f"{'CAMERA':<9} {'SEEN':>5} {'SHARE':>6} {'PLATES':>7} {'RATE':>6}  ACTIVE")
    for c in report.cameras:
        print(f"{c.camera_id:<9} {c.sightings:>5} {c.share:>5.0%} {c.plates_read:>7} "
              f"{c.plate_rate:>5.0%}  {c.first_seen:%H:%M:%S} - {c.last_seen:%H:%M:%S}")

    if report.corridors:
        print(f"\nCORRIDOR TRAVEL TIMES")
        print(f"{'CORRIDOR':<16} {'TRIPS':>6} {'KM':>5} {'MEDIAN':>8} {'FREEFLOW':>9} "
              f"{'INDEX':>6}  LEVEL")
        for c in report.corridors:
            flag = "" if c.baseline_from_data else "  (baseline from config)"
            print(f"{c.label:<16} {c.trips:>6} {c.distance_km:>5.1f} "
                  f"{c.median_travel_s/60:>7.1f}m {c.free_flow_travel_s/60:>8.1f}m "
                  f"{c.congestion_index:>6.2f}  {c.level}{flag}")

    if report.od_matrix:
        print(f"\nORIGIN -> DESTINATION")
        for origin, dests in sorted(report.od_matrix.items()):
            for dest, n in sorted(dests.items()):
                print(f"  {origin} -> {dest:<9} {n} trip(s)")

    peak = report.peak_bucket
    if peak and peak.count:
        print(f"\npeak {report.bucket_minutes}-min period   "
              f"{peak.start:%H:%M} - {peak.end:%H:%M}   {peak.count} vehicles")

    print(f"\nALERTS  ({len(alerts)})")
    if not alerts:
        print("  none -- nothing crossed a threshold")
    for a in alerts[:args.limit]:
        mark = "!" if a.severity == "critical" else ("*" if a.severity == "warning" else " ")
        review = "   [needs human review]" if a.needs_review else ""
        print(f"  {mark} [{a.severity:<8}] {a.message}{review}")
        if a.detail and args.verbose:
            print(f"      {a.detail}")
    if alerts and not args.verbose:
        print("\n  (--verbose for the reasoning behind each alert)")
    return 0


# ---------------------------------------------------------------------------
# doctor  -- the preflight check
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Everything that could embarrass you on demo day, checked in ten seconds.

    Run it the night before and again ten minutes before you present. Every
    failure here is one you would otherwise discover in front of judges.
    """
    import shutil
    import socket
    from pathlib import Path

    checks: list[tuple[str, str, str]] = []      # (state, name, detail)

    def ok(name, detail=""):    checks.append(("PASS", name, detail))
    def warn(name, detail=""):  checks.append(("WARN", name, detail))
    def bad(name, detail=""):   checks.append(("FAIL", name, detail))

    # -- python packages ---------------------------------------------------
    for module, label, needed_from in (
        ("cv2", "opencv", "M1"), ("yaml", "pyyaml", "M1"),
        ("ultralytics", "ultralytics", "M2"), ("lap", "lap (bytetrack solver)", "M3"),
        ("sqlalchemy", "sqlalchemy", "M4"), ("fast_alpr", "fast-alpr", "M5"),
        ("fastapi", "fastapi", "M6"), ("uvicorn", "uvicorn", "M6"),
    ):
        try:
            __import__(module)
            ok(f"import {label}")
        except ImportError:
            bad(f"import {label}", f"needed from {needed_from}: "
                                   f"pip install -r backend/requirements.txt")

    # -- device ------------------------------------------------------------
    try:
        from app.vision.detector import resolve_device
        device = resolve_device(cfg.runtime.device)
        (ok if device != "cpu" else warn)(
            f"inference device: {device}",
            "" if device != "cpu" else "cpu works but is slow; mps on Apple Silicon")
    except Exception as exc:
        warn("inference device", str(exc)[:80])

    # -- config and footage ------------------------------------------------
    enabled = cfg.enabled_cameras
    (ok if enabled else bad)(f"{len(enabled)} camera(s) enabled", "" if enabled
                             else "nothing to run -- check configs/cameras.yaml")
    missing = [c.id for c in enabled
               if not c.is_stream and not Path(c.source).exists()]
    if missing:
        bad(f"camera footage", f"missing for {', '.join(missing)} -- run 'make sample'")
    elif enabled:
        ok("camera footage present")

    # -- the graph ---------------------------------------------------------
    try:
        from app.core.city_graph import load_city_graph
        graph = load_city_graph(cfg.paths.root / "configs" / "city_graph.yaml", cfg.cameras)
        declared = len({tuple(sorted((e.from_id, e.to_id))) for e in graph.declared_edges})
        (ok if declared else warn)(
            f"city graph: {declared} declared edge(s)",
            "" if declared else "no measured roads -- distances will be straight-line guesses")

        # The trap from M7: offsets that imply impossible speeds reject every link.
        impossible = []
        for a in enabled:
            for b in enabled:
                if a.id >= b.id:
                    continue
                gap = abs(b.clock_offset_s - a.clock_offset_s)
                if gap <= 0:
                    continue
                possible, speed, why = graph.check(a.id, b.id, gap)
                if not possible and "ceiling" in why:
                    impossible.append(f"{a.id}->{b.id} implies {speed:.0f} km/h")
        if impossible:
            warn("clock offsets vs road distances",
                 "; ".join(impossible[:3]) + " -- links will be rejected as impossible")
        else:
            ok("clock offsets are physically consistent")
    except Exception as exc:
        warn("city graph", str(exc)[:80])

    # -- model weights -----------------------------------------------------
    weights = list(cfg.paths.model_dir.glob("*.pt")) if cfg.paths.model_dir.exists() else []
    (ok if weights else warn)(
        f"detector weights cached ({len(weights)})",
        "" if weights else "will download on first run (~5 MB) -- do it before you demo")

    # -- database ----------------------------------------------------------
    try:
        from app.store.db import get_sessionmaker, init_db
        from app.store.repository import stats as db_stats
        init_db(cfg.paths.database)
        with get_sessionmaker(cfg.paths.database)() as session:
            st = db_stats(session)
        ok("database writable", str(cfg.paths.database))
        (ok if st["sightings"] else warn)(
            f"seeded with {st['sightings']} sighting(s)",
            "" if st["sightings"] else "dashboard will open empty -- run 'make seed'")
    except Exception as exc:
        bad("database", str(exc)[:80])

    # -- offline map -------------------------------------------------------
    from app.api.tiles import cache_stats
    tiles = cache_stats(cfg.paths.data_dir / "tiles")
    (ok if tiles["tiles"] else warn)(
        f"map tiles cached ({tiles['tiles']}, {tiles['bytes'] / 1e6:.1f} MB)",
        "" if tiles["tiles"] else "map needs internet -- run 'make cache-tiles'")

    # -- ports and disk ----------------------------------------------------
    for port, label in ((8000, "api"), (5173, "dashboard")):
        sock = socket.socket()
        sock.settimeout(0.3)
        busy = sock.connect_ex(("127.0.0.1", port)) == 0
        sock.close()
        (warn if busy else ok)(f"port {port} ({label})",
                               "already in use -- stop the old process first" if busy else "free")

    free_gb = shutil.disk_usage(cfg.paths.root).free / 1e9
    (ok if free_gb > 2 else warn)(f"disk free {free_gb:.1f} GB",
                                  "" if free_gb > 2 else "tight for video output")

    # -- report ------------------------------------------------------------
    width = max(len(name) for _, name, _ in checks)
    print()
    for state, name, detail in checks:
        mark = {"PASS": " ok ", "WARN": "warn", "FAIL": "FAIL"}[state]
        print(f"  [{mark}]  {name:<{width}}  {detail}")

    fails = sum(1 for s, _, _ in checks if s == "FAIL")
    warns = sum(1 for s, _, _ in checks if s == "WARN")
    print(f"\n  {len(checks) - fails - warns} passed, {warns} warning(s), {fails} failure(s)")
    if fails:
        print("  fix the failures before demoing.")
    elif warns:
        print("  no blockers. the warnings are things that make the demo weaker, not broken.")
    else:
        print("  ready.")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# seed  -- process, resolve and analyse in one go
# ---------------------------------------------------------------------------

def cmd_seed(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Fill the database so the dashboard opens populated.

    The demo pattern from Milestone 0: seed first so analytics, journeys and
    alerts are rich the instant a judge looks at the screen, THEN run the live
    pipeline on top for the "watch it happen" moment. An empty dashboard shows
    nothing, however good the code behind it is.
    """
    from app.pipeline.camera_worker import new_run_id
    from app.pipeline.supervisor import run_supervisor
    from app.store.db import get_sessionmaker, init_db
    from app.store.repository import upsert_cameras

    cameras = cfg.enabled_cameras
    if not cameras:
        print("no enabled cameras", file=sys.stderr)
        return 2

    init_db(cfg.paths.database)
    with get_sessionmaker(cfg.paths.database)() as session:
        upsert_cameras(session, cfg.cameras)

    print(f"seeding from {len(cameras)} camera(s) at full speed "
          f"({args.max_frames} frames each)...\n")
    run_supervisor(cfg, cameras, args.run_id or new_run_id(), mode="fast",
                   workers=getattr(args, "workers", None),
                   max_frames=args.max_frames,
                   log_level=args.log_level or "WARNING")

    print("\nresolving identities and journeys...")
    resolve_args = argparse.Namespace(limit=5, log_level="WARNING")
    cmd_resolve(resolve_args, cfg)

    print("\ncomputing analytics and alerts...")
    analytics_args = argparse.Namespace(limit=5, verbose=False, log_level="WARNING")
    cmd_analytics(analytics_args, cfg)

    print("\nseeded. now: make api  +  make dash  (and 'make run' for live tiles)")
    return 0


# ---------------------------------------------------------------------------
# preflight -- is this footage worth processing at all?
# ---------------------------------------------------------------------------

def cmd_preflight(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Judge every configured source before an hour is spent on it.

    Written after two of ten TfL clips produced nothing and every obvious
    explanation -- codec, resolution, frame rate, corruption, path -- turned
    out to be wrong. They were valid 1-second "camera offline" cards. Nothing
    in the pipeline said so; the run simply produced fewer sightings than
    expected and nobody could tell which camera had failed or why.

    This makes that failure visible in seconds instead of an hour, and names
    the remedy rather than just the symptom.
    """
    # Only OpenCV and the config loader are imported at this level, on purpose.
    # Judging a video file must not require the database layer: preflight is
    # the FIRST thing you run on a new machine, often before a full install,
    # and "is this footage usable?" has nothing to do with SQLAlchemy.
    from app.vision.preflight import preflight_sources, summarise

    anpr_cfg = (cfg.raw.get("system", {}) or {}).get("anpr", {}) or {}
    min_plate = int(anpr_cfg.get("min_plate_width_px", 0) or 0)

    entries: list[tuple[str, str]] = []
    if args.dir:
        root = Path(args.dir).expanduser()
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
        exts = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".mpg", ".mpeg"}
        found = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
        entries = [(p.name, str(p)) for p in found]
        if not entries:
            print(f"no video files under {root}")
            return 0
    else:
        cams = cfg.cameras if args.all else cfg.enabled_cameras
        if args.cameras:
            wanted = {c.strip().upper() for c in args.cameras.split(",")}
            cams = [c for c in cams if c.id.upper() in wanted]
        entries = [(c.id, c.source) for c in cams]
        if not entries:
            print("no cameras to check -- see configs/cameras.yaml", file=sys.stderr)
            return 2

    probes = preflight_sources(entries, deep=args.deep, min_plate_width_px=min_plate,
                               head_frames=args.head_frames)

    # "Successfully processed" is a database question, not a file question, so
    # it is answered from the store rather than guessed from the footage.
    processed: dict[str, tuple[int, int]] = {}
    if not args.dir and cfg.paths.database.exists():
        try:
            from sqlalchemy import func, select

            from app.store.db import get_sessionmaker
            from app.store.models import Sighting
            with get_sessionmaker(cfg.paths.database)() as session:
                latest = session.execute(
                    select(Sighting.run_id).order_by(Sighting.id.desc()).limit(1)
                ).scalar()
                if latest:
                    rows = session.execute(
                        select(Sighting.camera_id, func.count(Sighting.id),
                               func.count(Sighting.plate_text))
                        .where(Sighting.run_id == latest)
                        .group_by(Sighting.camera_id)).all()
                    processed = {r[0]: (int(r[1]), int(r[2])) for r in rows}
        except Exception as exc:                       # never let reporting fail the check
            log.debug("could not read processed counts: %s", exc)

    label = {"ready": "READY", "placeholder": "PLACEHOLDER", "corrupt": "CORRUPT",
             "unsupported": "UNSUPPORTED", "missing": "MISSING", "stream": "STREAM"}

    w = max(6, min(46, max(len(p.camera_id) for p in probes)))
    print(f"\n{'source':<{w}} {'verdict':<12} {'size':>10} {'fps':>6} "
          f"{'frames':>7} {'secs':>7} {'codec':>8}  last run")
    print("-" * (w + 66))
    for p in probes:
        res = f"{p.width}x{p.height}" if p.width else "-"
        seen = processed.get(p.camera_id)
        ran = f"{seen[0]} sightings, {seen[1]} plates" if seen else (
            "not in last run" if processed else "-")
        print(f"{p.camera_id[:w]:<{w}} {label.get(p.verdict, p.verdict):<12} {res:>10} "
              f"{p.fps:>6.1f} {p.frames_to_process:>7} {p.duration_s:>7.1f} "
              f"{(p.codec or p.fourcc or '-'):>8}  {ran}")

    counts = summarise(probes)
    print("\n" + "  ".join(f"{k}: {v}" for k, v in counts.items() if v))

    # Detail only where it matters -- a wall of green lines teaches nothing.
    for p in probes:
        if p.verdict != "ready" or p.warnings:
            print(f"\n  {p.camera_id}  [{p.verdict}]  {p.path}")
            if p.reason:
                print(f"      {p.reason}")
            for w in p.warnings:
                print(f"      warning: {w}")

    if args.json:
        out = Path(args.json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": counts,
                                   "videos": [p.as_dict() for p in probes]},
                                  indent=2), encoding="utf-8")
        print(f"\nwritten to {out}")

    bad = [p for p in probes if p.verdict not in ("ready", "stream")]
    if bad:
        print(f"\n{len(bad)} source(s) will produce nothing: "
              f"{', '.join(p.camera_id for p in bad)}")
        if args.strict:
            return 1
    return 0


# ---------------------------------------------------------------------------
# export / import -- moving a finished run between machines
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Pack this machine's results into one archive.

    Run this on whichever machine did the processing. The archive is the only
    thing that has to travel.
    """
    from app.store.bundle import BundleError, export_bundle

    try:
        manifest = export_bundle(
            database=cfg.paths.database,
            data_dir=cfg.paths.data_dir,
            configs_dir=cfg.paths.root / "configs",
            out=Path(args.out).expanduser(),
            include_evidence=not args.no_evidence,
            note=args.note or "")
    except BundleError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2

    size_mb = manifest["bundle_bytes"] / 1e6
    print(f"\nwrote {manifest['bundle']}  ({size_mb:.1f} MB)")
    print(f"  runs      {len(manifest['runs'])}: {', '.join(manifest['runs']) or '(none)'}")
    for table, n in manifest["row_counts"].items():
        if n:
            print(f"  {table:<20} {n}")
    print(f"  evidence images      {manifest['evidence_files']}")
    print(f"  produced on          {manifest['produced_by'].get('platform', '?')} "
          f"[{manifest['produced_by'].get('device', '?')}]")
    print("\non the machine with the dashboard:\n"
          f"  python -m app.cli import {Path(manifest['bundle']).name}")
    return 0


def cmd_import(args: argparse.Namespace, cfg: AppConfig) -> int:
    """Merge an archive produced elsewhere into this machine's database."""
    from app.store.bundle import BundleError, import_bundle, read_manifest

    bundle = Path(args.bundle).expanduser()
    try:
        manifest = read_manifest(bundle)
    except (BundleError, OSError, ValueError) as exc:
        print(f"cannot read bundle: {exc}", file=sys.stderr)
        return 2

    print(f"\n{bundle.name}")
    print(f"  created   {manifest.get('created_at', '?')}")
    print(f"  from      {manifest.get('produced_by', {}).get('platform', '?')} "
          f"[{manifest.get('produced_by', {}).get('device', '?')}]")
    if manifest.get("note"):
        print(f"  note      {manifest['note']}")
    print(f"  runs      {', '.join(manifest.get('runs', [])) or '(none)'}")

    try:
        report = import_bundle(bundle=bundle, database=cfg.paths.database,
                               data_dir=cfg.paths.data_dir, mode=args.mode,
                               dry_run=args.dry_run)
    except BundleError as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 2

    head = "would import" if args.dry_run else ("replaced with" if args.mode == "replace"
                                                else "imported")
    print(f"\n{head}:")
    if report.runs_imported:
        print(f"  runs      {', '.join(report.runs_imported)}")
    if report.runs_skipped:
        print(f"  skipped   {', '.join(report.runs_skipped)}  "
              f"(already in this database)")
    for table, n in report.rows.items():
        print(f"  {table:<20} {n}")
    if not args.dry_run:
        print(f"  evidence images      {report.evidence_copied} copied"
              + (f", {report.evidence_skipped} already present"
                 if report.evidence_skipped else ""))
    if report.skipped_rows:
        detail = ", ".join(f"{t}: {n}" for t, n in report.skipped_rows.items())
        print(f"  rows not imported    {detail}")
    if report.camera_conflicts:
        print(f"\n  WARNING: these cameras exist here with different coordinates, "
              f"and this machine's version was kept: "
              f"{', '.join(report.camera_conflicts)}")
    if not report.runs_imported:
        print("\nnothing to do -- every run in this bundle is already here.")
        return 0
    if args.dry_run:
        print("\ndry run: nothing was written. Re-run without --dry-run to apply.")
    else:
        print(f"\n{report.total_rows} rows added. "
              f"open the dashboard, or: python -m app.cli db stats")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # --log-level lives on a shared parent so it is accepted both before and
    # after the subcommand. Small thing; saves a lot of "unrecognized
    # arguments" confusion at 2 a.m.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=None,
                        help="DEBUG | INFO | WARNING (default: from system.yaml)")

    parser = argparse.ArgumentParser(prog="app.cli", parents=[common],
                                     description="NETRA -- SIH PS 127")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cameras", parents=[common], help="list configured cameras")
    p.set_defaults(func=cmd_cameras)

    p = sub.add_parser("make-sample", parents=[common],
                       help="generate synthetic test footage")
    p.add_argument("--seconds", type=float, default=25.0)
    p.set_defaults(func=cmd_make_sample)

    p = sub.add_parser("preview", parents=[common],
                       help="play a camera feed with the live HUD")
    p.add_argument("--camera", required=True, help="camera id, e.g. CAM01")
    p.add_argument("--mode", choices=("realtime", "fast"), default=None)
    p.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--no-window", action="store_true", help="headless; log instead of display")
    p.add_argument("--source", default=None,
                   help="test against any video file, bypassing cameras.yaml")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("detect", parents=[common],
                       help="run vehicle detection on a camera feed")
    p.add_argument("--camera", required=True, help="camera id, e.g. CAM01")
    p.add_argument("--model", default=None, help="override the model, e.g. yolo11s.pt")
    p.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    p.add_argument("--mode", choices=("realtime", "fast"), default=None)
    p.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--save", action="store_true", help="write an annotated mp4 to data/out")
    p.add_argument("--no-window", action="store_true", help="headless; log instead of display")
    p.add_argument("--source", default=None,
                   help="test against any video file, bypassing cameras.yaml")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("track", parents=[common],
                       help="detect + track vehicles and count line crossings")
    p.add_argument("--camera", required=True, help="camera id, e.g. CAM01")
    p.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    p.add_argument("--mode", choices=("realtime", "fast"), default=None)
    p.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--save", action="store_true", help="write an annotated mp4 to data/out")
    p.add_argument("--no-window", action="store_true", help="headless; log instead of display")
    p.add_argument("--source", default=None,
                   help="test against any video file, bypassing cameras.yaml")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("process", parents=[common],
                       help="run the full pipeline and persist sightings")
    p.add_argument("--camera", required=True, help="camera id, e.g. CAM01")
    p.add_argument("--mode", choices=("realtime", "fast"), default=None)
    p.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--run-id", default=None,
                   help="reuse a run id; re-running with the same one is idempotent")
    p.add_argument("--save", action="store_true", help="write an annotated mp4 to data/out")
    p.add_argument("--no-window", action="store_true", help="headless; log instead of display")
    p.add_argument("--source", default=None,
                   help="test against any video file, bypassing cameras.yaml")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("db", parents=[common], help="inspect the event store")
    dbsub = p.add_subparsers(dest="db_command", required=True)
    dbsub.add_parser("init", parents=[common], help="create tables and mirror cameras.yaml")
    dbsub.add_parser("stats", parents=[common], help="row counts and breakdowns")
    dbsub.add_parser("cameras", parents=[common], help="cameras known to the database")
    q = dbsub.add_parser("sightings", parents=[common], help="most recent sightings")
    q.add_argument("--limit", type=int, default=25)
    q.add_argument("--camera", default=None)
    q = dbsub.add_parser("identities", parents=[common],
                         help="vehicles resolved across cameras")
    q.add_argument("--limit", type=int, default=25)
    q.add_argument("--min-cameras", type=int, default=1)
    q.add_argument("--verbose", action="store_true", help="show each hop")
    q = dbsub.add_parser("rejects", parents=[common],
                         help="links refused despite similar plates")
    q.add_argument("--limit", type=int, default=15)
    q = dbsub.add_parser("plate", parents=[common], help="every sighting of one plate")
    q.add_argument("value", help="plate text, e.g. KA01AB1234")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("run", parents=[common],
                       help="run every enabled camera, one process each")
    p.add_argument("--cameras", default=None, help="comma-separated subset, e.g. CAM01,CAM02")
    p.add_argument("--workers", type=int, default=None,
                   help="how many cameras to process at once (default: CPU count, max 8)")
    p.add_argument("--mode", choices=("realtime", "fast"), default=None)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resolve", parents=[common],
                       help="link sightings across cameras into vehicle identities")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("journey", parents=[common],
                       help="show a vehicle's reconstructed trip through the network")
    p.add_argument("--plate", default=None, help="plate to look up; omit for all journeys")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_journey)

    p = sub.add_parser("analytics", parents=[common],
                       help="traffic metrics and alert evaluation")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--verbose", action="store_true", help="show why each alert fired")
    p.set_defaults(func=cmd_analytics)

    p = sub.add_parser("doctor", parents=[common],
                       help="preflight check -- run this before you demo")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("preflight", parents=[common],
                       help="judge every video source: ready, placeholder, "
                            "corrupt, unsupported or missing")
    p.add_argument("--cameras", default=None, help="comma-separated subset")
    p.add_argument("--dir", default=None,
                   help="check every video under this directory instead of "
                        "the configured cameras")
    p.add_argument("--all", action="store_true",
                   help="include cameras marked enabled: false")
    p.add_argument("--deep", action="store_true",
                   help="decode every frame instead of sampling head and tail")
    p.add_argument("--head-frames", type=int, default=30,
                   help="frames to decode from the start when not --deep")
    p.add_argument("--json", default=None, help="also write the report to this path")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any source is not ready (for scripts)")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("seed", parents=[common],
                       help="process, resolve and analyse so the dashboard opens full")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--run-id", default=None)
    p.add_argument("--workers", type=int, default=None,
                   help="how many cameras to process at once (default: CPU count, max 8)")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("export", parents=[common],
                       help="pack results (database + evidence) into one archive")
    p.add_argument("--out", default="data/exports",
                   help="output file, or a directory to name one in")
    p.add_argument("--no-evidence", action="store_true",
                   help="skip snapshot images (much smaller, dashboard shows no photos)")
    p.add_argument("--note", default=None, help="free text stored in the manifest")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", parents=[common],
                       help="merge an archive produced on another machine")
    p.add_argument("bundle", help="path to the .tar.gz written by 'export'")
    p.add_argument("--mode", choices=("merge", "replace"), default="merge",
                   help="merge adds runs this machine does not have (default); "
                        "replace swaps the database, keeping a dated backup")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would happen and write nothing")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("serve", parents=[common], help="start the API and dashboard backend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    setup_logging(args.log_level or cfg.runtime.log_level)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
