# NETRA — measured results

Generated 2026-09-05 13:17 from `/mnt/user-data/uploads/SIH/data/netra.db`.

**Every number here is computed from the system's own output.** Where a
measurement has not been made, this file says so rather than estimating.

## Dataset

| | |
|---|---|
| Cameras | 3 |
| Vehicle sightings | 9 |
| Time window | 05:06:32 → 05:17:19 |
| Vehicle mix | bus 9 |
| Line crossings | eastbound 1, entry 1 |

## ANPR and temporal consensus

| | |
|---|---|
| Sightings with an agreed plate | 6 of 9 (67%) |
| Individual OCR attempts | 90 |
| Reads behind each agreed plate | 15.0 |

That last row is the point: a plate is agreed from **15 reads on average**, not one. No single frame decides a plate.

## Cross-camera association

| | |
|---|---|
| Vehicle identities resolved | 2 |
| Seen at more than one camera | 2 |
| Links accepted | 6 |
| Links refused (recorded near-misses) | 0 |

**Precision and recall: not yet measured.** They need ground truth —
one row per pass in `data/groundtruth/journeys.csv`, written down while
you film. Then `make eval`. Until that exists, quote the counts above
and nothing else.

## Trajectory reconstruction

| | |
|---|---|
| Journeys reconstructed | 2 |
| Hops (camera-to-camera legs) | 4 |
| Mean journey length | 4.7 km |
| Mean journey confidence | 0.82 |
| Journeys with a coverage gap | 0 |

Confidence is the **weakest hop** in a journey, not the average.

## Throughput

Not measured in this run. `python backend/scripts/report.py --benchmark`.

## Alerts

3 alert(s) currently raised. Rules: congestion from travel time, over-speed against the legal limit, fuzzy watchlist match (flagged for review, never asserted), and coverage gaps.

## What these numbers are not

- They are measured on the footage in `data/videos/`, not at city scale.
- Plate accuracy depends on camera placement and resolution; a plate needs
  roughly 80 px of width to be read reliably.
- Association confidence is calibrated on this data and would need
  re-calibration for a real deployment.

Quote these figures with the sample size attached, and nothing more.
