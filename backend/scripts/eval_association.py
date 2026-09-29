"""Measure cross-camera association against your own ground truth.

"How do you know it works?" is the question SIH judges ask and almost nobody
can answer. This answers it with precision and recall on a set you collected
yourself.

GROUND TRUTH FORMAT -- data/groundtruth/journeys.csv:

    vehicle,camera,time
    KA01AB1234,CAM01,10:42:13
    KA01AB1234,CAM02,10:46:51
    KA01AB1234,CAM03,10:52:09
    KA05MJ8842,CAM01,10:44:02
    ...

One row per pass. ``vehicle`` is any stable label you choose (the real plate is
ideal). Write it down as you film -- five minutes at the shoot saves the whole
argument later.

    python backend/scripts/eval_association.py

We score PAIRS, not clusters: for every pair of sightings, did the system put
them together, and should it have? That is the honest metric, because a single
wrong merge that joins two journeys is far worse than one missed hop, and pair
counting reflects that.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import load_config          # noqa: E402
from app.store.db import get_sessionmaker        # noqa: E402
from app.store.models import Sighting            # noqa: E402


def load_truth(path: Path) -> dict[tuple[str, str], str]:
    """(camera, HH:MM:SS) -> vehicle label."""
    truth: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vehicle = (row.get("vehicle") or "").strip()
            camera = (row.get("camera") or "").strip().upper()
            time = (row.get("time") or "").strip()
            if vehicle and camera and time:
                truth[(camera, time[:8])] = vehicle
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description="Score cross-camera association")
    ap.add_argument("--truth", default="data/groundtruth/journeys.csv")
    ap.add_argument("--tolerance", type=int, default=3,
                    help="seconds of slack when matching a sighting to a truth row")
    args = ap.parse_args()

    cfg = load_config()
    truth_path = Path(args.truth)
    if not truth_path.is_absolute():
        truth_path = cfg.paths.root / truth_path
    if not truth_path.exists():
        print(f"no ground truth at {truth_path}\n\n{__doc__.split('GROUND TRUTH')[1][:400]}",
              file=sys.stderr)
        return 2

    truth = load_truth(truth_path)
    Session = get_sessionmaker(cfg.paths.database)
    with Session() as session:
        rows = session.query(Sighting).order_by(Sighting.first_seen).all()

    # Attach each sighting to a truth row within the tolerance.
    labelled = []
    unmatched = 0
    for r in rows:
        found = None
        base = r.first_seen
        for delta in range(-args.tolerance, args.tolerance + 1):
            key = (r.camera_id.upper(),
                   (base.replace(microsecond=0)).strftime("%H:%M:%S"))
            key = (key[0], (base.replace(microsecond=0)
                            .replace(second=(base.second + delta) % 60)).strftime("%H:%M:%S")
                   if 0 <= base.second + delta < 60 else key[1])
            if key in truth:
                found = truth[key]
                break
        if found is None:
            unmatched += 1
        else:
            labelled.append((r, found))

    if len(labelled) < 2:
        print(f"only {len(labelled)} sighting(s) matched ground truth "
              f"({unmatched} unmatched). check your timestamps and tolerance.",
              file=sys.stderr)
        return 2

    tp = fp = fn = tn = 0
    for (ra, va), (rb, vb) in combinations(labelled, 2):
        same_truth = va == vb
        same_pred = ra.identity_id is not None and ra.identity_id == rb.identity_id
        if same_truth and same_pred:
            tp += 1
        elif same_truth and not same_pred:
            fn += 1
        elif not same_truth and same_pred:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    vehicles = len({v for _, v in labelled})
    print(f"\nground truth  {truth_path}")
    print(f"sightings     {len(labelled)} matched, {unmatched} unmatched")
    print(f"vehicles      {vehicles}")
    print(f"\npairs         {tp} true positive   {fp} false positive")
    print(f"              {fn} false negative  {tn} true negative")
    print(f"\nprecision     {precision:.1%}   (of the links we asserted, how many were right)")
    print(f"recall        {recall:.1%}   (of the links that existed, how many we found)")
    print(f"f1            {f1:.1%}")
    print(f"\nquote these numbers, with the sample size, and nothing more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
