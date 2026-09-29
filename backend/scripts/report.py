"""Generate the results slide from live data.

THE RULE THIS FILE ENFORCES: no number on a slide is typed by hand.

Every figure in the presentation comes from here, computed from the database as
it stands. Where a measurement has not been made yet, this says
**"not yet measured"** and names the command that would produce it -- it never
fills the gap with something plausible. A judge who asks "where did 94% come
from?" should get an answer, and "we ran this script" is one.

    python backend/scripts/report.py              # print
    python backend/scripts/report.py --write      # also write docs/RESULTS.md
    python backend/scripts/report.py --benchmark  # measure throughput too
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_config          # noqa: E402
from app.store.db import get_sessionmaker, init_db  # noqa: E402

NOT_MEASURED = "not yet measured"


def section(title: str) -> str:
    return f"\n## {title}\n"


def collect(cfg, benchmark: bool) -> dict:
    from sqlalchemy import func, select

    from app.store.models import (Alert, IdentityLink, Journey, JourneyHop,
                                  PlateRead, Sighting, VehicleIdentity)
    from app.store.repository import stats as db_stats

    init_db(cfg.paths.database)
    Session = get_sessionmaker(cfg.paths.database)
    out: dict = {"generated_at": datetime.now(), "database": str(cfg.paths.database)}

    with Session() as s:
        st = db_stats(s)
        out["dataset"] = st
        out["identities"] = s.scalar(select(func.count(VehicleIdentity.id))) or 0
        out["multi_camera"] = s.scalar(
            select(func.count(VehicleIdentity.id))
            .where(VehicleIdentity.camera_count > 1)) or 0
        out["journeys"] = s.scalar(select(func.count(Journey.id))) or 0
        out["hops"] = s.scalar(select(func.count(JourneyHop.id))) or 0
        out["links_accepted"] = s.scalar(
            select(func.count(IdentityLink.id))
            .where(IdentityLink.accepted.is_(True))) or 0
        out["links_rejected"] = s.scalar(
            select(func.count(IdentityLink.id))
            .where(IdentityLink.accepted.is_(False))) or 0
        out["alerts"] = s.scalar(select(func.count(Alert.id))) or 0
        out["coverage_gaps"] = s.scalar(
            select(func.count(Journey.id))
            .where(Journey.coverage_gaps != "")) or 0

        # Reads per plate is the temporal-consensus story in one number.
        plated = s.scalar(
            select(func.count(Sighting.id))
            .where(Sighting.plate_text.is_not(None))) or 0
        reads = s.scalar(select(func.count(PlateRead.id))) or 0
        out["reads_per_plate"] = (reads / plated) if plated else 0.0

        out["mean_journey_km"] = s.scalar(select(func.avg(Journey.total_distance_km))) or 0.0
        out["mean_confidence"] = s.scalar(select(func.avg(Journey.confidence))) or 0.0

    # -- ground-truth association score ------------------------------------
    truth = cfg.paths.data_dir / "groundtruth" / "journeys.csv"
    rows = [ln for ln in truth.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")] if truth.exists() else []
    if len(rows) > 1:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "eval_association.py")],
            capture_output=True, text=True, cwd=str(cfg.paths.root),
            env={"PYTHONPATH": "backend", "PATH": "/usr/bin:/bin"})
        out["association"] = proc.stdout.strip() if proc.returncode == 0 else NOT_MEASURED
    else:
        out["association"] = NOT_MEASURED

    # -- throughput --------------------------------------------------------
    out["throughput"] = NOT_MEASURED
    if benchmark:
        from app.core.bus import LocalBus
        from app.pipeline.camera_worker import run_camera
        from app.vision.detector import resolve_device

        camera = next((c for c in cfg.enabled_cameras
                       if c.is_stream or Path(c.source).exists()), None)
        if camera:
            print("  benchmarking one camera at full speed...", flush=True)
            began = time.perf_counter()
            stats = run_camera(cfg, camera, LocalBus("bench"), mode="fast",
                               max_frames=60, show=False, progress_every=0)
            out["throughput"] = {
                "device": resolve_device(cfg.runtime.device),
                "fps": round(stats.fps, 1),
                "frames": stats.frames,
                "seconds": round(time.perf_counter() - began, 1),
            }
    return out


def render(data: dict, cfg) -> str:
    st = data["dataset"]
    lines = [
        "# NETRA — measured results",
        "",
        f"Generated {data['generated_at']:%Y-%m-%d %H:%M} from `{data['database']}`.",
        "",
        "**Every number here is computed from the system's own output.** Where a",
        "measurement has not been made, this file says so rather than estimating.",
    ]

    lines += [section("Dataset")]
    window = (f"{st['first_seen']:%H:%M:%S} → {st['last_seen']:%H:%M:%S}"
              if st["first_seen"] else "—")
    lines += [
        "| | |", "|---|---|",
        f"| Cameras | {st['cameras']} |",
        f"| Vehicle sightings | {st['sightings']} |",
        f"| Time window | {window} |",
        f"| Vehicle mix | {', '.join(f'{k} {v}' for k, v in st['by_class'].items()) or '—'} |",
        f"| Line crossings | {', '.join(f'{k} {v}' for k, v in st['by_crossing'].items()) or '—'} |",
    ]

    lines += [section("ANPR and temporal consensus")]
    rate = (st["with_plate"] / st["sightings"]) if st["sightings"] else 0
    lines += [
        "| | |", "|---|---|",
        f"| Sightings with an agreed plate | {st['with_plate']} of {st['sightings']} ({rate:.0%}) |",
        f"| Individual OCR attempts | {st['plate_reads']} |",
        f"| Reads behind each agreed plate | {data['reads_per_plate']:.1f} |",
        "",
        f"That last row is the point: a plate is agreed from **{data['reads_per_plate']:.0f} "
        "reads on average**, not one. No single frame decides a plate.",
    ]

    lines += [section("Cross-camera association")]
    lines += [
        "| | |", "|---|---|",
        f"| Vehicle identities resolved | {data['identities']} |",
        f"| Seen at more than one camera | {data['multi_camera']} |",
        f"| Links accepted | {data['links_accepted']} |",
        f"| Links refused (recorded near-misses) | {data['links_rejected']} |",
    ]
    if data["association"] == NOT_MEASURED:
        lines += [
            "",
            "**Precision and recall: not yet measured.** They need ground truth —",
            "one row per pass in `data/groundtruth/journeys.csv`, written down while",
            "you film. Then `make eval`. Until that exists, quote the counts above",
            "and nothing else.",
        ]
    else:
        lines += ["", "```", data["association"], "```"]

    lines += [section("Trajectory reconstruction")]
    lines += [
        "| | |", "|---|---|",
        f"| Journeys reconstructed | {data['journeys']} |",
        f"| Hops (camera-to-camera legs) | {data['hops']} |",
        f"| Mean journey length | {data['mean_journey_km']:.1f} km |",
        f"| Mean journey confidence | {data['mean_confidence']:.2f} |",
        f"| Journeys with a coverage gap | {data['coverage_gaps']} |",
        "",
        "Confidence is the **weakest hop** in a journey, not the average.",
    ]

    lines += [section("Throughput")]
    if data["throughput"] == NOT_MEASURED:
        lines += ["Not measured in this run. `python backend/scripts/report.py --benchmark`."]
    else:
        t = data["throughput"]
        lines += [
            "| | |", "|---|---|",
            f"| Device | {t['device']} |",
            f"| End-to-end rate, one camera | {t['fps']} fps |",
            f"| Frames measured | {t['frames']} |",
            "",
            "Detection, tracking and ANPR together, on one camera. The pipeline runs",
            "one OS process per camera, so N cameras scale with cores rather than",
            "dividing this figure.",
        ]

    lines += [section("Alerts")]
    lines += [f"{data['alerts']} alert(s) currently raised. Rules: congestion from "
              "travel time, over-speed against the legal limit, fuzzy watchlist match "
              "(flagged for review, never asserted), and coverage gaps."]

    lines += [
        section("What these numbers are not"),
        "- They are measured on the footage in `data/videos/`, not at city scale.",
        "- Plate accuracy depends on camera placement and resolution; a plate needs",
        "  roughly 80 px of width to be read reliably.",
        "- Association confidence is calibrated on this data and would need",
        "  re-calibration for a real deployment.",
        "",
        "Quote these figures with the sample size attached, and nothing more.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Measured results for the presentation")
    ap.add_argument("--write", action="store_true", help="write docs/RESULTS.md")
    ap.add_argument("--benchmark", action="store_true", help="measure throughput too")
    args = ap.parse_args()

    cfg = load_config()
    data = collect(cfg, benchmark=args.benchmark)
    text = render(data, cfg)
    print(text)

    if args.write:
        out = cfg.paths.root / "docs" / "RESULTS.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
