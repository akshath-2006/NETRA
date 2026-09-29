"""Video preflight -- decide whether footage is usable BEFORE spending an hour on it.

WHY THIS EXISTS, in one sentence: a file that opens, decodes, and reports a
perfectly sane codec can still be worthless, and nothing in the pipeline used
to say so.

The case that forced it. Ten clips were pulled from TfL's open camera feed.
Two of them produced zero sightings. Every obvious explanation was wrong:

    codec       h264          same as the eight that worked
    resolution  352x288       same
    pixel fmt   yuv420p       same
    frame rate  25/1          same
    corruption  none          ffprobe and OpenCV both decode them cleanly

They were **1.000 second long, 25 frames, ~12.5 KB** -- the "camera
temporarily unavailable" card TfL's S3 endpoint serves when a camera is down.
A valid H.264 file containing no traffic. Not a codec problem, not a decoding
problem, not a path problem, not corruption. A *source availability* problem,
which is a different thing and has a different fix.

So this module refuses to answer "does it decode?" It answers the question
actually worth asking: **will this file produce useful results, and if not,
exactly why not?** Six verdicts, each with a different remedy:

    ready        process it
    placeholder  the camera was offline; re-capture, do not re-encode
    corrupt      truncated or damaged; re-fetch or repair
    unsupported  this OpenCV build has no decoder; transcode
    missing      the path in cameras.yaml points at nothing
    stream       live source, cannot be judged from disk

Everything here is measured from the file. Nothing is assumed from its name,
its extension, or its size alone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds. All overridable from the CLI, because "too short to be real" is a
# property of the source, not a universal truth.
# ---------------------------------------------------------------------------

#: A clip at or below this many seconds is treated as an offline placeholder
#: rather than footage. TfL's unavailable card is exactly 1.000 s; genuine
#: JamCam clips are 6-10 s. 2.0 separates them with room to spare.
PLACEHOLDER_MAX_SECONDS = 2.0

#: ...and it must also be small. A 2-second clip at 1080p is not a placeholder,
#: it is a short clip. Both conditions must hold.
PLACEHOLDER_MAX_BYTES = 60_000

#: Below this, a "video" is almost certainly an error page or a stub.
MIN_USABLE_BYTES = 2_000

#: A file whose header claims frames but which decodes fewer than this
#: fraction of them is damaged, not merely over-reported.
MIN_DECODE_RATIO = 0.5

#: Frame rates outside this band mean the header is lying or the clip is
#: variable-rate. Timestamps derived from frame index would then drift.
SANE_FPS = (1.0, 120.0)


VERDICTS = ("ready", "placeholder", "corrupt", "unsupported", "missing", "stream")


@dataclass
class VideoProbe:
    """What the file actually is, measured rather than assumed."""

    camera_id: str = ""
    path: str = ""
    exists: bool = False
    bytes: int = 0

    opened: bool = False
    tail_checked: bool = False      # did we actually manage to seek and look?
    width: int = 0
    height: int = 0
    fps: float = 0.0
    header_frames: int = 0          # what the container claims
    decoded_head: int = 0           # frames we actually decoded from the start
    decoded_tail: bool = False      # could we decode near the end?
    duration_s: float = 0.0

    fourcc: str = ""
    codec: str = ""                 # from ffprobe when available
    pix_fmt: str = ""
    avg_frame_rate: str = ""
    r_frame_rate: str = ""
    has_audio: bool = False
    ffprobe_ok: bool = False

    verdict: str = "missing"
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    # Planning numbers, so "can I run ten of these?" has an answer before the
    # hour is spent rather than after.
    megapixels_per_frame: float = 0.0
    frames_to_process: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def _ffprobe(path: Path) -> dict:
    """Container/codec detail, when ffprobe happens to be installed.

    Deliberately optional. ffmpeg is not a dependency of this project and the
    verdict must not change depending on whether a developer happens to have
    it -- OpenCV alone decides usable vs not. ffprobe only adds detail to the
    report (codec name, pixel format, whether the rate is variable).
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_streams", "-show_format",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return {}
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


#: Container magic numbers, as (offset, bytes, name). Enough to tell "a video
#: file that is damaged" from "not a video file at all" without ffmpeg, which
#: matters because the two have completely different remedies: re-download
#: versus transcode.
_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (4, b"ftyp", "mp4/mov"),
    (0, b"RIFF", "avi"),
    (0, b"\x1a\x45\xdf\xa3", "matroska/webm"),
    (0, b"OggS", "ogg"),
    (0, b"FLV", "flv"),
    (0, b"\x00\x00\x01\xba", "mpeg-ps"),
    (0, b"\x00\x00\x01\xb3", "mpeg-video"),
    (0, b"\x47", "mpeg-ts"),
)


def _container_signature(path: Path) -> str:
    """Which video container this file CLAIMS to be, from its first bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return ""
    for offset, magic, name in _SIGNATURES:
        if head[offset:offset + len(magic)] == magic:
            return name
    return ""


def _fourcc_str(value: float) -> str:
    n = int(value)
    if n <= 0:
        return ""
    try:
        s = "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))
    except ValueError:
        return ""
    return s if s.isprintable() else ""


def probe_file(path: Path, *, camera_id: str = "", head_frames: int = 30,
               deep: bool = False) -> VideoProbe:
    """Open, measure and decode enough of a file to judge it.

    ``head_frames`` frames are decoded from the start and one from near the
    end. That is enough to catch truncation, which is where damaged downloads
    actually fail, without paying to decode a 4K hour. ``deep`` decodes every
    frame instead -- slow, and only worth it when a file is under suspicion.
    """
    import cv2

    p = VideoProbe(camera_id=camera_id, path=str(path))
    if not path.exists():
        p.verdict = "missing"
        p.reason = "no file at this path"
        return p

    p.exists = True
    p.bytes = path.stat().st_size

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        # "Will not open" has two completely different causes and two
        # completely different fixes, so we work out which before answering.
        # A file whose header says "I am an MP4" and which still will not open
        # is a DAMAGED MP4 -- re-download it. A file with no video signature at
        # all is not our format -- transcode it or stop pointing at it.
        sig = _container_signature(path)
        meta = _ffprobe(path)
        has_video = any(s.get("codec_type") == "video"
                        for s in meta.get("streams", [])) if meta else False
        if meta:
            p.ffprobe_ok = True
        if has_video:
            for st in meta.get("streams", []):
                if st.get("codec_type") == "video":
                    p.codec = str(st.get("codec_name", ""))
                    p.pix_fmt = str(st.get("pix_fmt", ""))
                    break
            p.verdict = "unsupported"
            p.reason = (f"ffprobe reads this as {p.codec or 'video'}, but this "
                        f"OpenCV build has no decoder for it -- transcode it "
                        f"(ffmpeg -i in.ext -c:v libx264 out.mp4)")
        elif sig:
            p.verdict = "corrupt"
            p.reason = (f"the header says {sig} but nothing can decode it "
                        f"-- a truncated or damaged download. Fetch it again.")
        else:
            p.verdict = "unsupported"
            p.reason = ("no video container signature in the first bytes -- "
                        "this is not a video file, whatever the extension says")
        return p

    # The file is open, so "missing" is no longer the answer. Clearing the
    # default here is what lets classify() do its job; leaving it set meant
    # every successfully-opened file came back reported as missing.
    p.opened = True
    p.verdict = ""
    p.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    p.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    p.fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 3)
    p.header_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    p.fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))

    limit = p.header_frames if deep else head_frames
    decoded = 0
    while limit <= 0 or decoded < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded += 1
        if limit <= 0 and decoded > 100_000:     # unbounded header, stop somewhere
            break
    p.decoded_head = decoded

    # Truncation shows up at the END of a file, which is exactly the part a
    # head-only check never looks at.
    if not deep and p.header_frames > decoded:
        target = max(0, p.header_frames - 3)
        # cap.set() failing means this container does not support seeking --
        # which is NOT evidence of damage. Recording the difference is what
        # stops a perfectly good non-seekable file being called corrupt.
        if cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            ok, frame = cap.read()
            p.tail_checked = True
            p.decoded_tail = bool(ok and frame is not None)
    else:
        p.tail_checked = True
        p.decoded_tail = decoded > 0

    cap.release()

    frames = p.header_frames if p.header_frames > 0 else p.decoded_head
    if p.fps > 0 and frames > 0:
        p.duration_s = round(frames / p.fps, 3)

    p.megapixels_per_frame = round(p.width * p.height / 1e6, 4)
    p.frames_to_process = frames

    meta = _ffprobe(path)
    if meta:
        p.ffprobe_ok = True
        for st in meta.get("streams", []):
            if st.get("codec_type") == "video" and not p.codec:
                p.codec = str(st.get("codec_name", ""))
                p.pix_fmt = str(st.get("pix_fmt", ""))
                p.avg_frame_rate = str(st.get("avg_frame_rate", ""))
                p.r_frame_rate = str(st.get("r_frame_rate", ""))
            elif st.get("codec_type") == "audio":
                p.has_audio = True

    return p


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------

def classify(p: VideoProbe, *,
             placeholder_seconds: float = PLACEHOLDER_MAX_SECONDS,
             placeholder_bytes: int = PLACEHOLDER_MAX_BYTES,
             min_plate_width_px: int = 0) -> VideoProbe:
    """Turn measurements into one verdict and a written reason.

    Order matters. "Decodes nothing" must be decided before "is short",
    or a corrupt file gets filed as a placeholder and someone re-captures
    when they should be re-downloading.
    """
    if p.verdict in ("missing", "unsupported", "stream"):
        return p

    if p.decoded_head == 0:
        p.verdict = "corrupt"
        p.reason = ("the container opens but not a single frame decodes "
                    "-- damaged or truncated. Fetch it again.")
        return p

    if p.width <= 0 or p.height <= 0:
        p.verdict = "corrupt"
        p.reason = "decodes frames but reports no frame size"
        return p

    if p.bytes < MIN_USABLE_BYTES:
        p.verdict = "corrupt"
        p.reason = f"{p.bytes} bytes is too small to be video -- likely an error response"
        return p

    # Truncation: the header promises frames the file does not contain.
    # Only ever concluded from a tail read we actually performed.
    if (p.header_frames > 0 and p.tail_checked and not p.decoded_tail
            and p.decoded_head < p.header_frames):
        ratio = p.decoded_head / p.header_frames
        if ratio < MIN_DECODE_RATIO:
            p.verdict = "corrupt"
            p.reason = (f"header claims {p.header_frames} frames, "
                        f"only {p.decoded_head} decode and the tail is unreadable "
                        f"-- truncated download")
            return p
        p.warnings.append(f"tail unreadable past frame {p.decoded_head}")
    elif p.header_frames > 0 and not p.tail_checked:
        p.warnings.append("could not seek to the end -- truncation not ruled out; "
                          "re-check with --deep")

    # THE TfL CASE. Short AND small, both conditions, because either alone
    # produces false positives.
    if (p.duration_s > 0 and p.duration_s <= placeholder_seconds
            and p.bytes <= placeholder_bytes):
        p.verdict = "placeholder"
        p.reason = (f"{p.duration_s:g}s / {p.decoded_head or p.header_frames} frames "
                    f"/ {p.bytes/1000:.1f} KB -- a valid video file containing no "
                    f"footage. The camera was offline when this was captured. "
                    f"Re-capture it; re-encoding cannot add footage that was "
                    f"never there.")
        return p

    p.verdict = "ready"
    p.reason = "decodes cleanly at a sane size and frame rate"

    # ---- non-fatal warnings: usable, but you should know ------------------
    if not (SANE_FPS[0] <= p.fps <= SANE_FPS[1]):
        p.warnings.append(f"implausible frame rate {p.fps} -- timestamps may drift")

    if p.ffprobe_ok and p.avg_frame_rate and p.r_frame_rate \
            and p.avg_frame_rate != p.r_frame_rate:
        p.warnings.append(
            f"variable frame rate ({p.avg_frame_rate} avg vs {p.r_frame_rate} base) "
            f"-- frame-index timestamps will drift against wall clock")

    if p.duration_s and p.duration_s < 5:
        p.warnings.append(f"only {p.duration_s:g}s long -- few vehicles will cross")

    # ANPR reachability is arithmetic, not opinion: a plate can never be wider
    # than the frame, so a threshold above the frame width is unreachable, and
    # one above half the frame width needs the plate to fill half the picture.
    if min_plate_width_px and p.width:
        need = min_plate_width_px / p.width
        if need >= 1.0:
            p.warnings.append(
                f"NO PLATES POSSIBLE: anpr.min_plate_width_px is "
                f"{min_plate_width_px}px but the frame is only {p.width}px wide")
        elif need >= 0.4:
            p.warnings.append(
                f"plates unlikely: a plate must fill {need:.0%} of the frame width "
                f"to reach the {min_plate_width_px}px threshold")

    return p


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def preflight_sources(entries, *, head_frames: int = 30, deep: bool = False,
                      min_plate_width_px: int = 0,
                      placeholder_seconds: float = PLACEHOLDER_MAX_SECONDS,
                      placeholder_bytes: int = PLACEHOLDER_MAX_BYTES) -> list[VideoProbe]:
    """Judge many (camera_id, source) pairs. Streams are reported, not opened."""
    out: list[VideoProbe] = []
    for camera_id, source in entries:
        s = str(source)
        lowered = s.lower()
        if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://")) or lowered.isdigit():
            p = VideoProbe(camera_id=camera_id, path=s, verdict="stream",
                           reason="live source -- cannot be judged from disk; "
                                  "check it with 'app.cli preview'")
            out.append(p)
            continue
        p = probe_file(Path(s), camera_id=camera_id, head_frames=head_frames, deep=deep)
        classify(p, min_plate_width_px=min_plate_width_px,
                 placeholder_seconds=placeholder_seconds,
                 placeholder_bytes=placeholder_bytes)
        out.append(p)
    return out


def summarise(probes: list[VideoProbe]) -> dict[str, int]:
    counts = {v: 0 for v in VERDICTS}
    for p in probes:
        counts[p.verdict] = counts.get(p.verdict, 0) + 1
    return counts
