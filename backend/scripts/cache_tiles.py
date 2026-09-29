"""Pre-download map tiles for the camera network.

Run this once on a good connection and the demo no longer needs the venue's
wifi. It covers the bounding box of every configured camera, with a margin, at
the zoom levels the dashboard actually uses.

    python backend/scripts/cache_tiles.py            # zooms 11-16
    python backend/scripts/cache_tiles.py --max-zoom 17

Tiles are CARTO basemaps built from OpenStreetMap data -- free at this volume,
no key, no account. Attribution stays on the map. The script rate-limits itself
and skips anything already cached, so re-running it is cheap.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.tiles import cache_stats, fetch_tile, tile_path   # noqa: E402
from app.core.config import load_config                        # noqa: E402


def deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0 * n)
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-cache map tiles for offline demo")
    ap.add_argument("--min-zoom", type=int, default=11)
    ap.add_argument("--max-zoom", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="degrees of padding around the camera bounding box")
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.cameras:
        print("no cameras configured", file=sys.stderr)
        return 2

    lats = [c.location.lat for c in cfg.cameras]
    lons = [c.location.lon for c in cfg.cameras]
    south, north = min(lats) - args.margin, max(lats) + args.margin
    west, east = min(lons) - args.margin, max(lons) + args.margin
    cache_dir = cfg.paths.data_dir / "tiles"

    print(f"area   {south:.4f},{west:.4f} -> {north:.4f},{east:.4f}")
    print(f"zooms  {args.min_zoom}-{args.max_zoom}")
    print(f"cache  {cache_dir}\n")

    fetched = skipped = failed = 0
    for z in range(args.min_zoom, args.max_zoom + 1):
        x0, y0 = deg2tile(north, west, z)
        x1, y1 = deg2tile(south, east, z)
        count = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
        print(f"  z{z:<3} {count} tile(s)", end="", flush=True)

        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                path = tile_path(cache_dir, z, x, y)
                if path.exists():
                    skipped += 1
                    continue
                data = fetch_tile(z, x, y)
                if data:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    fetched += 1
                else:
                    failed += 1
                time.sleep(args.delay)
        print("  done")

    stats = cache_stats(cache_dir)
    print(f"\nfetched {fetched}, already cached {skipped}, failed {failed}")
    print(f"cache now holds {stats['tiles']} tiles "
          f"({stats['bytes'] / 1e6:.1f} MB) at zooms {stats['zooms']}")
    if failed:
        print("\nsome tiles failed -- re-run on a better connection; the map still "
              "works, it just falls back to plain dark where a tile is missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
