"""The supervisor: one OS process per camera.

WHY PROCESSES AND NOT THREADS
Python's GIL means threaded workers serialise on inference -- three cameras
would run at one camera's throughput. Separate processes each own their model
instance and video decoder and genuinely run in parallel.

That is the honest engineering reason. The architectural one matters more: this
is exactly the boundary a real deployment cuts along. The same
``run_camera`` function runs on an edge box beside the physical camera, with a
Kafka bus instead of the local one. Nothing about the worker changes -- only
who starts it and where its events go.

WHY NOT ALL OF THEM AT ONCE
One process per camera is right. Every process at the same instant is not.
Each worker loads its own YOLO and its own two ONNX sessions, so ten cameras
on a four-core laptop do not run ten times faster -- they run slower than four
would, because the cores are oversubscribed and every model is resident in RAM
at once. On a single GPU they additionally compete for the same VRAM, and the
failure there is not slowness but an out-of-memory crash partway through.

So cameras are processed in batches, and a finished worker's slot is handed
straight to the next camera waiting. Cameras are independent until ``resolve``
runs -- nothing in a worker reads another camera's output -- so batching
changes nothing about what the system concludes. Only the peak load differs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass

from app.core.logging import get_logger, setup_logging

log = get_logger("supervisor")


@dataclass
class WorkerHandle:
    camera_id: str
    process: mp.Process

    @property
    def alive(self) -> bool:
        return self.process.is_alive()


def _worker_entry(camera_id: str, run_id: str, mode: str | None,
                  seconds: float | None, max_frames: int | None,
                  log_level: str) -> None:
    """Runs inside the child process.

    Config is re-loaded here rather than passed in: the child gets a clean,
    picklable start and there is no shared mutable state to go wrong.
    """
    setup_logging(log_level)
    from app.core.bus import LocalBus
    from app.core.config import load_config
    from app.pipeline.camera_worker import run_camera
    from app.store.db import get_sessionmaker
    from app.store.subscriber import SightingWriter

    cfg = load_config()
    camera = cfg.get_camera(camera_id)

    bus = LocalBus(camera_id)
    SightingWriter(get_sessionmaker(cfg.paths.database), run_id).register(bus)

    try:
        stats = run_camera(cfg, camera, bus, mode=mode, seconds=seconds,
                           max_frames=max_frames, show=False,
                           live_dir=cfg.paths.data_dir / "live",
                           progress_every=0, run_id=run_id)
        get_logger(f"worker.{camera_id}").info(
            "finished: %d frames, %d sightings, %d plates",
            stats.frames, stats.sightings, stats.plates)
    except Exception:
        get_logger(f"worker.{camera_id}").exception("worker crashed")


#: Measured resident memory of one camera worker, in GB: its own YOLO plus two
#: ONNX sessions plus decoder buffers. Measured at ~0.8 GB on 1280x720
#: footage; 1.0 leaves a little headroom rather than sailing close.
WORKER_MEMORY_GB = 1.0


def _total_ram_gb() -> float:
    """Physical RAM, or 0.0 when this platform will not say."""
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def default_workers(cfg=None) -> int:
    """How many cameras to process at once.

    Bounded by THREE things, because any one of them alone gets it wrong:

    * **cores** -- past the core count you stop buying parallelism and start
      paying for context switching;
    * **memory** -- each worker holds its own YOLO and two ONNX sessions, and
      a laptop reporting 8 logical cores may only have 8 GB of RAM. Counting
      cores alone is how eight workers end up asking for ~6.4 GB and pushing
      the machine into swap, where everything is slow and nothing says why.
      Half the physical RAM is left for the OS and everything else;
    * **8** -- a hard ceiling, so a 64-core box does not try to hold 64 copies
      of the model.

    Set ``performance.max_parallel_cameras`` in system.yaml, or --workers on
    the command line, to override any of this. On a single GPU the right
    number is usually 2-4 regardless of cores or system RAM, because VRAM is
    the binding constraint.
    """
    perf = ((cfg.raw.get("system", {}) or {}).get("performance", {}) or {}) if cfg else {}
    configured = int(perf.get("max_parallel_cameras", 0) or 0)
    if configured > 0:
        return configured

    try:
        limit = min(8, os.cpu_count() or 2)
    except Exception:
        limit = 2

    ram = _total_ram_gb()
    if ram > 0:
        by_memory = int((ram * 0.5) // WORKER_MEMORY_GB)
        limit = min(limit, max(1, by_memory))
    return max(1, limit)


def run_supervisor(cfg, cameras, run_id: str, *, mode: str | None = None,
                   seconds: float | None = None, max_frames: int | None = None,
                   log_level: str = "INFO", workers: int | None = None
                   ) -> list[WorkerHandle]:
    """Start a worker for every camera, at most ``workers`` running at once."""
    try:
        mp.set_start_method("spawn")     # required on macOS, safest everywhere
    except RuntimeError:
        pass                             # already set

    limit = max(1, workers or default_workers(cfg))
    cameras = list(cameras)
    if limit >= len(cameras):
        log.info("processing all %d cameras concurrently", len(cameras))
    else:
        log.info("processing %d cameras, %d at a time", len(cameras), limit)

    handles: list[WorkerHandle] = []
    pending = list(cameras)

    def _spawn(camera):
        proc = mp.Process(
            target=_worker_entry,
            args=(camera.id, run_id, mode, seconds, max_frames, log_level),
            name=f"netra-{camera.id}",
        )
        proc.start()
        h = WorkerHandle(camera.id, proc)
        handles.append(h)
        log.info("started %s (pid %d)", camera.id, proc.pid)
        return h

    running: list[WorkerHandle] = []
    while pending and len(running) < limit:
        running.append(_spawn(pending.pop(0)))

    stopping = False

    def _stop(_signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        log.info("stopping workers...")
        for h in handles:
            if h.alive:
                h.process.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Report a death rather than restarting. During a demo a worker that keeps
    # crash-looping is far more confusing than one camera tile going dark.
    reported: set[str] = set()
    while running or pending:
        for h in list(running):
            if h.alive:
                continue
            running.remove(h)
            if h.camera_id not in reported:
                reported.add(h.camera_id)
                if h.process.exitcode not in (0, None):
                    log.warning("%s exited (code %s) -- other cameras continue",
                                h.camera_id, h.process.exitcode)
                else:
                    log.info("%s finished", h.camera_id)
            # A slot freed up: start the next camera waiting for one.
            if pending and not stopping:
                running.append(_spawn(pending.pop(0)))
        if stopping:
            pending.clear()
        time.sleep(0.4)

    for h in handles:
        h.process.join(timeout=5)
    return handles
