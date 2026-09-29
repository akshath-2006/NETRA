#!/usr/bin/env python3
"""Build a real-footage test corridor from TfL's open traffic cameras.

WHAT THIS GIVES YOU, AND WHAT IT DOES NOT
-----------------------------------------
TfL publishes ~900 London traffic cameras under the Open Government Licence
v2.0, each with real coordinates and a short MP4 clip. That is real footage
from real fixed infrastructure cameras on a real road network, which is a
large step up from generated clips.

It is NOT a substitute for plate-readable footage. TfL's own documentation
describes JamCams as "low-resolution overviews of traffic conditions -- not
individual vehicle detail" and states they "do not record number plates".

    VALIDATES on this footage      DOES NOT validate
    ------------------------       -----------------
    vehicle detection              ANPR / plate reading
    multi-object tracking          plate consensus
    line crossing / counting       plate-driven cross-camera matching
    congestion from travel time    accuracy against ground truth
    camera health / coverage       (there is none -- see below)

GROUND TRUTH: there is none. No vehicle in this data carries a label saying
which other camera saw it. Any cross-camera result you get here is
EXPLORATORY and must be reported as such. Do not quote an accuracy figure
from it.

LICENCE
    Open Government Licence v2.0, with TfL's amendments. Commercial and
    non-commercial use are both permitted. Attribution is REQUIRED and must
    appear wherever the data or anything derived from it is shown:

        Powered by TfL Open Data
        Contains OS data (C) Crown copyright and database rights 2016
        Geomni UK Map data (C) and database rights 2019

    Use the Unified API (as this script does). The licence forbids automated
    scraping of TfL's other web properties.

ROAD DISTANCES
    Taken from OSRM routing over OpenStreetMap data where the router is
    reachable, and marked `source: osrm`. Where it is not, the straight-line
    distance times a detour factor is used and marked `source: estimated`.
    The two are never conflated -- see --calibrate, which measures the real
    detour factor for this corridor instead of assuming one.

USAGE
    python backend/scripts/fetch_tfl.py --list                 # find corridors
    python backend/scripts/fetch_tfl.py --corridor A406 --plan # inspect one
    python backend/scripts/fetch_tfl.py --corridor A406 --capture --rounds 6
    python backend/scripts/fetch_tfl.py --corridor A406 --write-config
    python backend/scripts/fetch_tfl.py --corridor A406 --calibrate
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TFL_PLACES = "https://api.tfl.gov.uk/Place/Type/JamCam"
OSRM = "http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
UA = {"User-Agent": "NETRA-SIH-prototype/0.12 (student research)"}

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "test" / "tfl"

ATTRIBUTION = [
    "Powered by TfL Open Data",
    "Contains OS data (C) Crown copyright and database rights 2016",
    "Geomni UK Map data (C) and database rights 2019",
]


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def load_cameras(cache: Path) -> list[dict]:
    """The camera registry, cached so repeated runs do not re-hit the API."""
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        raw = json.loads(cache.read_text())
    else:
        print(f"fetching camera registry from {TFL_PLACES}")
        raw = get_json(TFL_PLACES)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
        print(f"  cached {len(raw)} cameras -> {cache}")

    cams = []
    for c in raw:
        props = {p.get("key"): p.get("value") for p in c.get("additionalProperties", [])}
        vid = props.get("videoUrl")
        if not vid or c.get("lat") is None:
            continue
        cams.append({
            "tfl_id": c["id"],
            "name": c.get("commonName", ""),
            "lat": float(c["lat"]),
            "lon": float(c["lon"]),
            "video_url": vid,
            "image_url": props.get("imageUrl", ""),
            "available": str(props.get("available", "true")).lower() != "false",
        })
    return cams


# --------------------------------------------------------------------------
# geometry and corridors
# --------------------------------------------------------------------------

def haversine_km(a: dict, b: dict) -> float:
    R = 6371.0088
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dp = p2 - p1
    dl = math.radians(b["lon"] - a["lon"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


ROAD_RE = re.compile(r"^([AB]\d{1,4})\b")


def road_of(cam: dict) -> str | None:
    """The road a camera sits on, from its name. 'A406 Hanger Ln' -> 'A406'."""
    m = ROAD_RE.match(cam["name"].strip())
    return m.group(1) if m else None


def corridors(cams: list[dict], min_cameras: int = 3, max_gap_km: float = 3.0):
    """Group cameras into chains that actually sit along one road in sequence.

    Adjacency is the whole point. Two cameras on the same road 40 km apart are
    not a corridor -- no vehicle travels between them inside any sensible time
    window, so pairing them would manufacture a cross-camera problem that does
    not exist. We order by position along the road and cut wherever the gap is
    too large to be a plausible hop.
    """
    by_road: dict[str, list[dict]] = {}
    for c in cams:
        r = road_of(c)
        if r:
            by_road.setdefault(r, []).append(c)

    out = []
    for road, group in by_road.items():
        if len(group) < min_cameras:
            continue
        # Order along the dominant axis of the road, so "next camera" means
        # the next one you would physically drive past.
        lats = [c["lat"] for c in group]
        lons = [c["lon"] for c in group]
        spread_lat = max(lats) - min(lats)
        spread_lon = max(lons) - min(lons)
        key = (lambda c: c["lat"]) if spread_lat >= spread_lon else (lambda c: c["lon"])
        group = sorted(group, key=key)

        chain = [group[0]]
        for prev, cur in zip(group, group[1:]):
            if haversine_km(prev, cur) <= max_gap_km:
                chain.append(cur)
            else:
                if len(chain) >= min_cameras:
                    out.append((road, chain))
                chain = [cur]
        if len(chain) >= min_cameras:
            out.append((road, chain))

    out.sort(key=lambda rc: -len(rc[1]))
    return out


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def road_distance_km(a: dict, b: dict, detour: float = 1.35):
    """(distance_km, source). Real routing when we can get it, else estimated."""
    try:
        url = OSRM.format(lon1=a["lon"], lat1=a["lat"], lon2=b["lon"], lat2=b["lat"])
        d = get_json(url, timeout=25)
        if d.get("code") == "Ok" and d.get("routes"):
            return round(d["routes"][0]["distance"] / 1000.0, 3), "osrm"
    except Exception:
        pass
    return round(haversine_km(a, b) * detour, 3), "estimated"


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_list(cams, args):
    found = corridors(cams, args.min_cameras, args.max_gap_km)
    print(f"\n{len(cams)} cameras with video; {len(found)} corridors of "
          f">= {args.min_cameras} cameras within {args.max_gap_km} km hops\n")
    print(f"{'road':<8}{'cams':>5}{'span km':>9}  first -> last")
    for road, chain in found[:args.top]:
        span = sum(haversine_km(a, b) for a, b in zip(chain, chain[1:]))
        print(f"{road:<8}{len(chain):>5}{span:>9.1f}  "
              f"{chain[0]['name'][:32]} -> {chain[-1]['name'][:32]}")
    print("\ninspect one with:  --corridor <road> --plan")


def pick(cams, args):
    if args.cameras:
        want = [c.strip() for c in args.cameras.split(",")]
        chain = [c for w in want for c in cams if c["tfl_id"].endswith(w) or c["tfl_id"] == w]
        if len(chain) != len(want):
            sys.exit(f"could not resolve every id in {want}")
        return args.corridor or "CUSTOM", chain
    found = corridors(cams, args.min_cameras, args.max_gap_km)
    for road, chain in found:
        if road.upper() == (args.corridor or "").upper():
            return road, chain[:args.max_cameras]
    sys.exit(f"no corridor {args.corridor!r}; run --list to see what exists")


def plan_rows(chain, detour):
    """Per-hop distances and the travel time we will stagger captures by."""
    rows = []
    for a, b in zip(chain, chain[1:]):
        km, src = road_distance_km(a, b, detour)
        straight = haversine_km(a, b)
        secs = (km / max(1e-6, 30.0)) * 3600.0       # 30 km/h urban assumption
        rows.append({"from": a, "to": b, "km": km, "source": src,
                     "straight_km": round(straight, 3),
                     "ratio": round(km / straight, 3) if straight > 0 else None,
                     "travel_s": round(secs)})
    return rows


def cmd_plan(cams, args):
    road, chain = pick(cams, args)
    print(f"\ncorridor {road}: {len(chain)} cameras")
    for i, c in enumerate(chain, 1):
        flag = "" if c["available"] else "   [marked unavailable by TfL]"
        print(f"  CAM{i:02d}  {c['tfl_id']:<22} {c['name'][:44]:<44} "
              f"{c['lat']:.5f},{c['lon']:.5f}{flag}")
    print("\nhops:")
    for r in plan_rows(chain, args.detour):
        print(f"  {r['from']['name'][:26]:<26} -> {r['to']['name'][:26]:<26} "
              f"{r['km']:>6.2f} km  ({r['source']}, straight {r['straight_km']} km, "
              f"ratio {r['ratio']})   stagger capture by {r['travel_s']}s")
    print("\nNOTE: the stagger is what gives a vehicle a chance of appearing in")
    print("      two clips. It is an experiment design, not a guarantee, and")
    print("      there is no ground truth to confirm it either way.")


def cmd_capture(cams, args):
    road, chain = pick(cams, args)
    rows = plan_rows(chain, args.detour)
    outdir = DATA / road.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"\ncapturing {args.rounds} round(s) from {len(chain)} cameras -> {outdir}")
    for rnd in range(1, args.rounds + 1):
        print(f"\nround {rnd}/{args.rounds}")
        for i, cam in enumerate(chain):
            stamp = datetime.now(timezone.utc)
            name = f"r{rnd:02d}_CAM{i+1:02d}_{stamp:%Y%m%dT%H%M%SZ}.mp4"
            try:
                blob = get_bytes(cam["video_url"])
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"  CAM{i+1:02d} FAILED: {e}")
                continue
            (outdir / name).write_bytes(blob)
            manifest.append({
                "file": name, "camera_index": i + 1, "tfl_id": cam["tfl_id"],
                "camera_name": cam["name"], "lat": cam["lat"], "lon": cam["lon"],
                "fetched_utc": stamp.isoformat(), "bytes": len(blob),
                "round": rnd,
            })
            print(f"  CAM{i+1:02d} {len(blob)/1024:7.0f} KB  {name}")
            # Stagger by the travel time to the NEXT camera, so a vehicle
            # leaving this camera has time to reach the next before we sample it.
            if i < len(rows):
                wait = min(rows[i]["travel_s"], args.max_stagger_s)
                print(f"        waiting {wait}s (travel time to next camera)")
                time.sleep(wait)
        if rnd < args.rounds:
            time.sleep(args.round_gap_s)

    meta = {
        "dataset": f"TfL JamCams corridor {road}",
        "source_url": TFL_PLACES,
        "licence": "Open Government Licence v2.0 (TfL amendments)",
        "licence_url": "https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service",
        "attribution_required": ATTRIBUTION,
        "commercial_use_permitted": True,
        "cameras": len(chain),
        "videos": len(manifest),
        "synchronised": False,
        "synchronisation_note": (
            "Clips are pulled sequentially and staggered by estimated travel "
            "time. Capture times are recorded per file in UTC. This is NOT "
            "frame-synchronised multi-camera capture."),
        "same_vehicles_across_cameras": "unverified -- plausible, not labelled",
        "ground_truth": "NONE",
        "labelling_status": "unlabeled / exploratory",
        "usable_for": ["detection", "tracking", "counting", "congestion"],
        "NOT_usable_for": ["ANPR", "plate consensus",
                           "cross-camera accuracy measurement"],
        "resolution_note": (
            "TfL describes JamCams as low-resolution traffic overviews that "
            "do not record number plates. Expect zero readable plates."),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest,
    }
    (outdir / "dataset.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(manifest)} clips + dataset.json in {outdir}")
    print("REMEMBER the attribution wherever you show this footage:")
    for line in ATTRIBUTION:
        print(f"   {line}")


def cmd_write_config(cams, args):
    road, chain = pick(cams, args)
    rows = plan_rows(chain, args.detour)
    outdir = DATA / road.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    cam_lines = [
        "# GENERATED from TfL open data by backend/scripts/fetch_tfl.py",
        "# Powered by TfL Open Data.",
        "# Contains OS data (C) Crown copyright and database rights 2016.",
        "# Geomni UK Map data (C) and database rights 2019.",
        "#",
        "# Real cameras, real coordinates. clock_offset_s is 0 for every camera:",
        "# these clips were captured near-simultaneously in real time, so unlike",
        "# the generated demo clips they need no virtual offset to share a clock.",
        "cameras:",
    ]
    for i, c in enumerate(chain, 1):
        cam_lines += [
            f"  - id: CAM{i:02d}",
            f"    name: \"{c['name']}\"",
            f"    source: data/test/tfl/{road.lower()}/LATEST_CAM{i:02d}.mp4",
            f"    location: {{ lat: {c['lat']}, lon: {c['lon']} }}",
            f"    heading_deg: 0            # unknown; TfL does not publish bearing",
            f"    target_fps: 12",
            f"    clock_offset_s: 0",
            f"    loop: false",
            f"    enabled: true",
            f"    tfl_id: {c['tfl_id']}",
            "",
        ]
    (outdir / "cameras.generated.yaml").write_text("\n".join(cam_lines))

    g = ["# GENERATED from TfL open data + OSRM routing over OpenStreetMap.",
         "# Every edge records where its distance came from. An edge marked",
         "# `estimated` is a straight line times a detour factor, NOT a measured",
         "# road distance, and must not be presented as one.",
         "edges:"]
    for i, r in enumerate(rows, 1):
        g += [
            f"  - from: CAM{i:02d}",
            f"    to: CAM{i+1:02d}",
            f"    distance_km: {r['km']}",
            f"    distance_source: {r['source']}",
            f"    straight_line_km: {r['straight_km']}",
            f"    detour_ratio: {r['ratio']}",
            f"    typical_speed_kmph: 30      # urban assumption, NOT measured",
            f"    bidirectional: true",
            "",
        ]
    (outdir / "city_graph.generated.yaml").write_text("\n".join(g))
    print(f"wrote {outdir}/cameras.generated.yaml")
    print(f"wrote {outdir}/city_graph.generated.yaml")
    osrm = sum(1 for r in rows if r["source"] == "osrm")
    print(f"  {osrm}/{len(rows)} edges have REAL routed distances; "
          f"{len(rows)-osrm} are estimated and labelled as such")


def cmd_calibrate(cams, args):
    """Measure the detour factor instead of assuming one (Part 11)."""
    road, chain = pick(cams, args)
    rows = plan_rows(chain, args.detour)
    real = [r for r in rows if r["source"] == "osrm" and r["ratio"]]
    if not real:
        print("no routed hops available -- cannot calibrate. "
              "Keep the configured fallback and label edges `estimated`.")
        return
    ratios = sorted(r["ratio"] for r in real)
    mid = ratios[len(ratios) // 2] if len(ratios) % 2 else \
        (ratios[len(ratios)//2 - 1] + ratios[len(ratios)//2]) / 2
    print(f"\ndetour factor, measured over {len(real)} routed hops on {road}")
    for r in real:
        print(f"   {r['from']['name'][:28]:<28} -> {r['to']['name'][:28]:<28} "
              f"road {r['km']:.2f} / straight {r['straight_km']:.2f} = {r['ratio']}")
    print(f"\n  min {ratios[0]}   median {mid}   max {ratios[-1]}")
    print(f"  currently configured fallback: {args.detour}")
    print("\n  Use the median as the fallback ONLY where routing is unavailable.")
    print("  A measured road distance always beats a factor.")
    print(f"  Sample size is {len(real)} hops on ONE corridor -- widen it before")
    print("  treating this as a city-wide constant.")



def cmd_prepare(cams, args):
    """Turn one captured round into a runnable NETRA dataset.

    Three things have to happen here that --write-config could not do, because
    they depend on what the capture actually produced:

    1. JUNK FILES. TfL serves a tiny placeholder for a camera that is offline
       or covered. Those arrive as ~12 KB "mp4" files that OpenCV cannot open.
       We probe every file and drop the ones with no decodable frames, rather
       than letting a worker die on them mid-run.

    2. REAL CLOCK OFFSETS. --write-config wrote clock_offset_s: 0 everywhere,
       which is WRONG for staggered capture: it would place every camera at the
       same instant, so a vehicle would appear to be at CAM01 and CAM05 at once
       and the physics check would reject every real link. The clips were taken
       minutes apart on purpose, and the true offsets are the actual capture
       times recorded in dataset.json.

    3. HONEST SPEED BOUNDS. Distances come from routing; the speed fields are
       urban assumptions and are labelled as such.
    """
    import shutil
    try:
        import cv2
    except ImportError:
        sys.exit("opencv is needed to probe the clips: run this inside the project venv")

    road = (args.corridor or "custom").lower()
    outdir = DATA / road
    meta_path = outdir / "dataset.json"
    if not meta_path.exists():
        sys.exit(f"no {meta_path} -- run --capture first")
    meta = json.loads(meta_path.read_text())
    files = meta["files"]

    def probe(f):
        """Is this clip real footage, and how big is it?

        TfL serves a ~12 KB placeholder for a camera that is offline or
        covered. Those are VALID one-frame mp4 files, so "did it open?" and
        "did a frame come back?" both say yes. They have to be rejected on
        substance -- frame count and size -- or they enter the config as real
        cameras and the map ends up showing nodes that saw nothing.
        """
        path = outdir / f["file"]
        info = {"ok": False, "w": 0, "h": 0, "n": 0, "fps": 0.0}
        if not path.exists():
            return info
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            info["n"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = cap.get(cv2.CAP_PROP_FPS) or 0.0
            got, _frame = cap.read()
            info["ok"] = (bool(got) and info["w"] > 0
                          and info["n"] >= args.min_frames
                          and f["bytes"] >= args.min_bytes)
        cap.release()
        return info

    def span_of(fs):
        ts = [datetime.fromisoformat(x["fetched_utc"]) for x in fs]
        return (max(ts) - min(ts)).total_seconds() if len(ts) > 1 else 0.0

    def worst_adjacent_gap(fs):
        """The biggest hop between consecutive cameras.

        This -- not the total span -- is what decides whether links are even
        attempted. CAM01 and CAM10 being 32 min apart is fine; CAM03 and CAM04
        being 98 min apart because the laptop slept is not.
        """
        ts = sorted(datetime.fromisoformat(x["fetched_utc"]) for x in fs)
        return max(((b - a).total_seconds() for a, b in zip(ts, ts[1:])), default=0.0)

    rounds = sorted({f["round"] for f in files})
    scored = []
    for r in rounds:
        fs = sorted([f for f in files if f["round"] == r], key=lambda f: f["camera_index"])
        good = []
        for f in fs:
            info = probe(f)
            if info["ok"]:
                f.update({k: info[k] for k in ("w", "h", "n", "fps")})
                good.append(f)
        scored.append({"round": r, "all": fs, "good": good,
                       "gap": worst_adjacent_gap(good), "span": span_of(good)})

    window_s = args.max_gap_minutes * 60
    print(f"\nround   usable   worst adjacent gap   total span")
    for sc in scored:
        flag = "" if sc["gap"] <= window_s else "   <- gap exceeds the linking window"
        print(f"  {sc['round']}    {len(sc['good']):>2}/{len(sc['all'])}"
              f"      {sc['gap']/60:>8.1f} min      {sc['span']/60:>6.1f} min{flag}")

    if args.round:
        chosen = next((sc for sc in scored if sc["round"] == args.round), None)
        if chosen is None:
            sys.exit(f"no round {args.round}")
    else:
        # Most usable cameras wins; among equals, the tightest adjacent gap.
        # A round the laptop slept through is worse than one with fewer
        # cameras, because a 98-minute hop links to nothing at all.
        ok_rounds = [sc for sc in scored if sc["gap"] <= window_s] or scored
        chosen = max(ok_rounds, key=lambda sc: (len(sc["good"]), -sc["gap"]))

    rnd = chosen["round"]
    usable = chosen["good"]
    print(f"\nusing round {rnd}: {len(usable)} usable cameras, "
          f"worst adjacent gap {chosen['gap']/60:.1f} min")
    for f in chosen["all"]:
        kept = f in usable
        print(f"  CAM{f['camera_index']:02d}  {f['bytes']/1024:6.0f} KB  "
              f"{f.get('w',0)}x{f.get('h',0)} {f.get('n',0):>4} frames  "
              f"{'keep' if kept else 'DROP'}   {f['camera_name'][:34]}")

    if len(usable) < 2:
        sys.exit("\nfewer than 2 usable clips -- nothing to link across. "
                 "Capture more rounds, or pick another corridor.")

    # Real offsets, from real capture times.
    t0 = min(datetime.fromisoformat(f["fetched_utc"]) for f in usable)
    for f in usable:
        f["offset_s"] = round((datetime.fromisoformat(f["fetched_utc"]) - t0).total_seconds())

    cdir = ROOT / "configs"
    for name in ("cameras.yaml", "city_graph.yaml"):
        src = cdir / name
        bak = cdir / f"{name}.demo-backup"
        if src.exists() and not bak.exists():
            shutil.copy2(src, bak)
            print(f"\nbacked up {name} -> {bak.name}")

    lines = [
        "# GENERATED by backend/scripts/fetch_tfl.py --prepare",
        "# REAL FOOTAGE from TfL open traffic cameras.",
        "#   Powered by TfL Open Data",
        "#   Contains OS data (C) Crown copyright and database rights 2016",
        "#   Geomni UK Map data (C) and database rights 2019",
        "#",
        "# clock_offset_s below are MEASURED capture-time differences, not",
        "# invented. The clips were pulled minutes apart on purpose so a vehicle",
        "# leaving one camera had time to reach the next.",
        "#",
        "# EXPECT NO PLATES. TfL describes these as low-resolution traffic",
        "# overviews that do not record number plates. Detection, tracking and",
        "# counting are what this footage tests.",
        "cameras:",
    ]
    for i, f in enumerate(usable, 1):
        lines += [
            f"  - id: CAM{i:02d}",
            f"    name: \"{f['camera_name']}\"",
            f"    source: data/test/tfl/{road}/{f['file']}",
            f"    location: {{ lat: {f['lat']}, lon: {f['lon']} }}",
            f"    heading_deg: 0",
            f"    target_fps: 12",
            f"    clock_offset_s: {f['offset_s']}",
            f"    loop: false",
            f"    enabled: true",
            "",
        ]
    (cdir / "cameras.yaml").write_text("\n".join(lines))

    rows = []
    for a, b in zip(usable, usable[1:]):
        km, src = road_distance_km(a, b, args.detour)
        rows.append((a, b, km, src))
    g = [
        "# GENERATED by backend/scripts/fetch_tfl.py --prepare",
        "# Distances routed over OpenStreetMap via OSRM where available.",
        "# Every edge states its source. `estimated` is a straight line times a",
        "# detour factor and is NOT a measured road distance.",
        "defaults:",
        f"  detour_factor: {args.detour}",
        "  max_speed_kmph: 90        # motorway-capable arterial",
        "  speed_limit_kmph: 48      # 30 mph, the London default",
        "  min_speed_kmph: 3",
        "  typical_speed_kmph: 30    # ASSUMPTION, not measured",
        "  max_gap_minutes: 30",
        "",
        "# Off: this corridor is a chain, and deriving edges between every pair",
        "# would invent shortcuts that do not exist on the road.",
        "auto_edges: false",
        "",
        "edges:",
    ]
    for i, (a, b, km, src) in enumerate(rows, 1):
        g += [
            f"  - from: CAM{i:02d}",
            f"    to: CAM{i+1:02d}",
            f"    distance_km: {km}",
            f"    # distance_source: {src}",
            f"    typical_speed_kmph: 30",
            f"    max_speed_kmph: 90",
            f"    speed_limit_kmph: 48",
            f"    bidirectional: true",
            "",
        ]
    (cdir / "city_graph.yaml").write_text("\n".join(g))

    routed = sum(1 for *_r, s in rows if s == "osrm")
    span = max(f["offset_s"] for f in usable)
    print(f"\nwrote configs/cameras.yaml      {len(usable)} real cameras")
    print(f"wrote configs/city_graph.yaml   {len(rows)} edges "
          f"({routed} routed, {len(rows)-routed} estimated)")
    print(f"worst adjacent gap {chosen['gap']/60:.1f} min "
          f"(links are attempted within {args.max_gap_minutes} min)")
    print(f"total span {span/60:.1f} min -- the first and last camera may be too")
    print(f"far apart to link directly, which is expected and harmless.")
    print("\nnext:")
    print("  make seed && make resolve && make analytics")
    print("  restore the demo config with: --restore")


def cmd_restore(args):
    import shutil
    cdir = ROOT / "configs"
    done = []
    for name in ("cameras.yaml", "city_graph.yaml"):
        bak = cdir / f"{name}.demo-backup"
        if bak.exists():
            shutil.copy2(bak, cdir / name)
            done.append(name)
    print("restored:", ", ".join(done) if done else "nothing to restore")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="show candidate corridors")
    p.add_argument("--plan", action="store_true", help="show one corridor in detail")
    p.add_argument("--capture", action="store_true", help="download clips")
    p.add_argument("--write-config", action="store_true", help="emit NETRA config")
    p.add_argument("--calibrate", action="store_true", help="measure the detour factor")
    p.add_argument("--prepare", action="store_true",
                   help="turn a captured round into a runnable NETRA config")
    p.add_argument("--restore", action="store_true",
                   help="put the original demo configs back")
    p.add_argument("--round", type=int, default=0, help="which capture round to use")
    p.add_argument("--min-frames", type=int, default=25,
                   help="reject clips with fewer decodable frames (TfL placeholders)")
    p.add_argument("--min-bytes", type=int, default=30000,
                   help="reject clips smaller than this (TfL placeholders are ~12 KB)")
    p.add_argument("--max-gap-minutes", type=int, default=30,
                   help="must match city_graph defaults.max_gap_minutes")
    p.add_argument("--corridor", help="road id, e.g. A406")
    p.add_argument("--cameras", help="explicit comma-separated TfL ids")
    p.add_argument("--min-cameras", type=int, default=3)
    p.add_argument("--max-cameras", type=int, default=10)
    p.add_argument("--max-gap-km", type=float, default=3.0)
    p.add_argument("--detour", type=float, default=1.35,
                   help="fallback detour factor when routing is unavailable")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--round-gap-s", type=int, default=300,
                   help="TfL refreshes every ~5 min; shorter gaps re-fetch the same clip")
    p.add_argument("--max-stagger-s", type=int, default=240)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    if args.restore:
        return cmd_restore(args)

    cams = load_cameras(DATA / "cameras.cache.json")
    if args.list:
        return cmd_list(cams, args)
    if not (args.corridor or args.cameras):
        return cmd_list(cams, args)
    if args.plan:
        return cmd_plan(cams, args)
    if args.capture:
        return cmd_capture(cams, args)
    if args.write_config:
        return cmd_write_config(cams, args)
    if args.calibrate:
        return cmd_calibrate(cams, args)
    if args.prepare:
        return cmd_prepare(cams, args)
    cmd_plan(cams, args)


if __name__ == "__main__":
    main()
