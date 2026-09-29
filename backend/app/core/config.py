"""Configuration loading and validation.

Everything the system can be told is loaded here, once, into frozen
dataclasses. Two reasons this is worth a proper module rather than a bare
``yaml.safe_load`` at each call site:

* **Typo protection.** A misspelled key in ``cameras.yaml`` at 2 a.m. should
  produce "camera CAM02: missing required field 'source'", not a
  ``KeyError`` fifteen frames into a video loop.
* **Path resolution.** Config files use paths relative to the project root, so
  the same config works no matter which directory you run from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(Exception):
    """Raised for any malformed or missing configuration."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float


@dataclass(frozen=True)
class CountingLine:
    """A virtual line vehicles are counted across.

    Coordinates are normalised (0-1 of frame width and height) so the same
    line keeps its meaning if the source resolution changes.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    positive_label: str = "inbound"
    negative_label: str = "outbound"

    def pixels(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((int(self.start[0] * width), int(self.start[1] * height)),
                (int(self.end[0] * width), int(self.end[1] * height)))


@dataclass(frozen=True)
class CameraConfig:
    """One physical (or simulated) camera in the network."""

    id: str
    name: str
    source: str                 # absolute path, or an rtsp:// URL, or "0"
    location: GeoPoint
    heading_deg: float = 0.0
    target_fps: float = 12.0
    clock_offset_s: float = 0.0
    loop: bool = True
    enabled: bool = True
    counting_line: CountingLine | None = None

    @property
    def is_stream(self) -> bool:
        """True when the source is live rather than a file on disk."""
        s = self.source.lower()
        return s.startswith(("rtsp://", "rtmp://", "http://", "https://")) or s.isdigit()


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    video_dir: Path
    model_dir: Path
    database: Path


@dataclass(frozen=True)
class Runtime:
    mode: str = "realtime"      # realtime | fast
    device: str = "auto"        # auto | cpu | mps | cuda
    log_level: str = "INFO"


@dataclass
class AppConfig:
    paths: Paths
    runtime: Runtime
    cameras: list[CameraConfig] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    def get_camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id.upper() == camera_id.upper():
                return cam
        known = ", ".join(c.id for c in self.cameras) or "(none configured)"
        raise ConfigError(f"unknown camera {camera_id!r}. configured cameras: {known}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Repository root, resolved from this file's location.

    Override with NETRA_ROOT when running from somewhere unusual.
    """
    env = os.environ.get("NETRA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # .../<root>/backend/app/core/config.py  ->  parents[3] == <root>
    return Path(__file__).resolve().parents[3]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping at the top level")
    return data


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _resolve(value: str | Path, root: Path) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (root / p)


def _parse_camera(entry: Any, index: int, root: Path) -> CameraConfig:
    if not isinstance(entry, dict):
        raise ConfigError(f"cameras[{index}]: expected a mapping, got {type(entry).__name__}")

    cam_id = str(_require(entry, "id", f"cameras[{index}]")).strip()
    context = f"camera {cam_id}"

    source = str(_require(entry, "source", context)).strip()
    loc = _require(entry, "location", context)
    if not isinstance(loc, dict) or "lat" not in loc or "lon" not in loc:
        raise ConfigError(f"{context}: 'location' must define both 'lat' and 'lon'")

    # File sources are resolved against the project root; streams stay as-is.
    lowered = source.lower()
    is_stream = lowered.startswith(("rtsp://", "rtmp://", "http://", "https://")) or lowered.isdigit()
    resolved_source = source if is_stream else str(_resolve(source, root))

    line_cfg = entry.get("counting_line")
    line = None
    if line_cfg:
        if not isinstance(line_cfg, dict):
            raise ConfigError(f"{context}: 'counting_line' must be a mapping")
        try:
            start = tuple(float(v) for v in _require(line_cfg, "start", context))
            end = tuple(float(v) for v in _require(line_cfg, "end", context))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{context}: counting_line start/end must be [x, y]") from exc
        if len(start) != 2 or len(end) != 2:
            raise ConfigError(f"{context}: counting_line start/end need exactly two values")
        for name, pt in (("start", start), ("end", end)):
            if not all(0.0 <= v <= 1.0 for v in pt):
                raise ConfigError(
                    f"{context}: counting_line {name} must be normalised 0-1, got {pt}")
        labels = line_cfg.get("labels") or ["inbound", "outbound"]
        line = CountingLine(start=start, end=end,
                            positive_label=str(labels[0]),
                            negative_label=str(labels[1] if len(labels) > 1 else "outbound"))

    try:
        return CameraConfig(
            id=cam_id,
            name=str(entry.get("name", cam_id)),
            source=resolved_source,
            location=GeoPoint(lat=float(loc["lat"]), lon=float(loc["lon"])),
            heading_deg=float(entry.get("heading_deg", 0.0)),
            target_fps=float(entry.get("target_fps", 12.0)),
            clock_offset_s=float(entry.get("clock_offset_s", 0.0)),
            loop=bool(entry.get("loop", True)),
            enabled=bool(entry.get("enabled", True)),
            counting_line=line,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: {exc}") from exc


def load_config(configs_dir: Path | None = None) -> AppConfig:
    """Load configs/system.yaml and configs/cameras.yaml into an AppConfig."""
    root = project_root()
    cdir = configs_dir or (root / "configs")

    system = _read_yaml(cdir / "system.yaml")
    cameras_doc = _read_yaml(cdir / "cameras.yaml")

    paths_cfg = system.get("paths", {}) or {}
    paths = Paths(
        root=root,
        data_dir=_resolve(paths_cfg.get("data_dir", "data"), root),
        video_dir=_resolve(paths_cfg.get("video_dir", "data/videos"), root),
        model_dir=_resolve(paths_cfg.get("model_dir", "data/models"), root),
        database=_resolve(paths_cfg.get("database", "data/netra.db"), root),
    )

    rt_cfg = system.get("runtime", {}) or {}
    mode = str(rt_cfg.get("mode", "realtime")).lower()
    if mode not in {"realtime", "fast"}:
        raise ConfigError(f"runtime.mode must be 'realtime' or 'fast', got {mode!r}")
    runtime = Runtime(
        mode=mode,
        device=str(rt_cfg.get("device", "auto")).lower(),
        log_level=str(rt_cfg.get("log_level", "INFO")).upper(),
    )

    entries: Iterable[Any] = cameras_doc.get("cameras") or []
    if not isinstance(entries, list):
        raise ConfigError("cameras.yaml: 'cameras' must be a list")

    cameras = [_parse_camera(e, i, root) for i, e in enumerate(entries)]

    seen: set[str] = set()
    for cam in cameras:
        key = cam.id.upper()
        if key in seen:
            raise ConfigError(f"duplicate camera id {cam.id!r} in cameras.yaml")
        seen.add(key)

    return AppConfig(paths=paths, runtime=runtime, cameras=cameras, raw={"system": system})
