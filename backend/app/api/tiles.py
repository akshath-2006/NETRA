"""Map tiles, cached on disk.

WHY THIS EXISTS
Venue wifi fails. A dashboard whose map is a blank grey rectangle in front of
judges is a dashboard that looks broken, however good the pipeline underneath
is. So the API serves tiles from a local cache and only reaches the network on
a miss -- which means:

* online, it works and fills the cache as you pan;
* after ``make cache-tiles``, it works with no network at all;
* if a tile is missing and the network is down, it returns a dark tile the same
  colour as the map background, so the map degrades to a plain dark canvas with
  the camera network still drawn on top, rather than to broken-image icons.

Attribution stays on the map either way -- these are CARTO basemap tiles built
from OpenStreetMap data, free to use at this volume, no key and no account.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from app.core.logging import get_logger

log = get_logger("tiles")

UPSTREAM = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
SUBDOMAIN_UPSTREAM = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
USER_AGENT = "NETRA-SIH-prototype/1.0 (student project; low volume)"
TIMEOUT_S = 6.0

# 1x1 PNG in the map's own background colour (#0A0F14). Returned when a tile is
# neither cached nor reachable, so the map goes plain rather than broken.
_FALLBACK = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "89000000106a464946000101010060000000000000000000194944415478da63"
    "6060606000000004000100b0d20e0000000049454e44ae426082"
)


def tile_path(cache_dir: Path, z: int, x: int, y: int) -> Path:
    return Path(cache_dir) / str(z) / str(x) / f"{y}.png"


def fetch_tile(z: int, x: int, y: int) -> bytes | None:
    """Fetch one tile upstream. Returns None on any failure -- never raises."""
    url = SUBDOMAIN_UPSTREAM.format(s="abcd"[(x + y) % 4], z=z, x=x, y=y)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            if response.status != 200:
                return None
            return response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.debug("tile %d/%d/%d unavailable: %s", z, x, y, exc)
        return None


def get_tile(cache_dir: Path, z: int, x: int, y: int,
             allow_network: bool = True) -> tuple[bytes, str]:
    """Return ``(png_bytes, source)`` where source is cache | network | fallback."""
    path = tile_path(cache_dir, z, x, y)
    if path.exists():
        try:
            return path.read_bytes(), "cache"
        except OSError:
            pass

    if allow_network:
        data = fetch_tile(z, x, y)
        if data:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            except OSError:
                pass          # serving the tile matters more than caching it
            return data, "network"

    return _FALLBACK, "fallback"


def cache_stats(cache_dir: Path) -> dict:
    cache = Path(cache_dir)
    if not cache.exists():
        return {"tiles": 0, "bytes": 0, "zooms": []}
    files = list(cache.rglob("*.png"))
    zooms = sorted({int(p.parent.parent.name) for p in files
                    if p.parent.parent.name.isdigit()})
    return {"tiles": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "zooms": zooms}
