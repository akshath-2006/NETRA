"""Video ingest -- the one way frames enter NETRA.

Every camera, whether it is a recorded clip today or an RTSP stream in a real
deployment, is read through this class. Two things make it more than a thin
wrapper around ``cv2.VideoCapture``:

**1. Virtual wall-clock timestamps.**
Cross-camera association is built entirely on "when did this vehicle pass this
camera". A recorded file has no real timestamps of its own, so we map the
video's internal timeline onto a chosen start time, plus the camera's
``clock_offset_s``. That is what lets three separately-recorded clips behave
like three cameras sharing one city clock -- and it is why a vehicle can leave
CAM01 at 10:42:13 and arrive at CAM02 at 10:46:51 in a way the trajectory
engine can reason about later.

**2. Playback pacing.**
In ``realtime`` mode we sleep so a 30-fps clip plays at 30 fps, which is what
makes the demo look live. In ``fast`` mode we drop the sleep and process as
quickly as the machine allows -- which is what we want when pre-seeding the
database before judging.

Frame rate is decoupled from the source: ``target_fps`` in cameras.yaml decides
how many frames we actually hand downstream. Detection does not need all 30.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from app.core.config import CameraConfig
from app.core.logging import get_logger

# Some containers report a nonsense FPS. Fall back to something sane rather
# than dividing by zero three frames into the demo.
DEFAULT_NATIVE_FPS = 25.0


class VideoSourceError(RuntimeError):
    """Raised when a camera source cannot be opened or read."""


class Frame:
    """One frame handed downstream, with everything the pipeline needs."""

    __slots__ = ("camera_id", "image", "index", "source_index", "timestamp", "lap")

    def __init__(
        self,
        camera_id: str,
        image: np.ndarray,
        index: int,
        source_index: int,
        timestamp: datetime,
        lap: int = 0,
    ) -> None:
        self.camera_id = camera_id
        self.image = image              # BGR, as OpenCV gives it
        self.index = index              # count of frames we emitted
        self.source_index = source_index  # position in the original video
        self.timestamp = timestamp      # virtual wall-clock time
        self.lap = lap                  # how many times the clip has looped

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"Frame({self.camera_id} #{self.index} "
            f"@{self.timestamp:%H:%M:%S.%f}"[:-3] + ")"
        )


class VideoSource:
    """Reads a camera source and yields paced, timestamped frames."""

    def __init__(
        self,
        camera: CameraConfig,
        mode: str = "realtime",
        virtual_start: datetime | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.camera = camera
        self.mode = mode
        self.virtual_start = virtual_start or datetime.now().replace(microsecond=0)
        self.max_frames = max_frames

        self.log = get_logger(f"video.{camera.id}")
        self._cap: cv2.VideoCapture | None = None

        self.native_fps: float = DEFAULT_NATIVE_FPS
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0
        self.stride: int = 1

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "VideoSource":
        source = self.camera.source

        if not self.camera.is_stream:
            path = Path(source)
            if not path.exists():
                raise VideoSourceError(
                    f"{self.camera.id}: video file not found: {path}\n"
                    f"  hint: run 'make sample' to generate test footage."
                )
            target: str | int = str(path)
        else:
            target = int(source) if source.isdigit() else source

        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise VideoSourceError(f"{self.camera.id}: could not open source {source!r}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = float(fps) if fps and fps > 0 else DEFAULT_NATIVE_FPS
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        want = self.camera.target_fps
        self.stride = max(1, round(self.native_fps / want)) if want > 0 else 1

        self._cap = cap
        self.log.info(
            "opened %s (%dx%d @ %.1f fps) -> processing every %d%s frame (~%.1f fps)",
            Path(source).name if not self.camera.is_stream else source,
            self.width, self.height, self.native_fps,
            self.stride, _ordinal_suffix(self.stride),
            self.native_fps / self.stride,
        )
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open() if self._cap is None else self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the interesting part ---------------------------------------------

    def frames(self) -> Iterator[Frame]:
        """Yield frames, paced and timestamped.

        Loops forever when ``camera.loop`` is set, so callers should bound the
        run with ``max_frames`` or simply stop iterating.
        """
        if self._cap is None:
            self.open()
        assert self._cap is not None

        wall_start = time.perf_counter()
        emitted = 0
        lap = 0
        lap_seconds = 0.0     # virtual time accumulated by completed laps
        source_index = -1

        while True:
            # Frames we are going to throw away still cost a full decode if we
            # read() them. grab() advances the stream and skips the expensive
            # decode-and-convert; retrieve() only pays for it on frames we keep.
            # At stride 2 that is half the decode work we were doing for nothing.
            if self.stride > 1:
                peek = source_index + 1
                if peek % self.stride:
                    if self._cap.grab():
                        source_index = peek
                        continue
                    ok, image = False, None
                else:
                    ok = self._cap.grab()
                    if ok:
                        ok, image = self._cap.retrieve()
                    else:
                        image = None
            else:
                ok, image = self._cap.read()

            if not ok:
                # End of file. Loop if asked, otherwise we are done.
                if self.camera.loop and not self.camera.is_stream and source_index >= 0:
                    lap += 1
                    lap_seconds += (source_index + 1) / self.native_fps
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    source_index = -1
                    self.log.debug("clip exhausted, looping (lap %d)", lap)
                    continue
                self.log.info("source ended after %d emitted frames", emitted)
                break

            source_index += 1
            if source_index % self.stride:
                continue  # not one of ours

            video_seconds = lap_seconds + source_index / self.native_fps

            # Pace to wall-clock so recorded footage behaves like a live feed.
            if self.mode == "realtime" and not self.camera.is_stream:
                sleep_for = (wall_start + video_seconds) - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # Live streams carry real time; files carry virtual time.
            if self.camera.is_stream:
                stamp = datetime.now()
            else:
                stamp = self.virtual_start + timedelta(
                    seconds=self.camera.clock_offset_s + video_seconds
                )

            yield Frame(
                camera_id=self.camera.id,
                image=image,
                index=emitted,
                source_index=source_index,
                timestamp=stamp,
                lap=lap,
            )
            emitted += 1

            if self.max_frames is not None and emitted >= self.max_frames:
                self.log.info("reached max_frames=%d", self.max_frames)
                break


def _ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
