# NETRA — bootup (3-camera demo footage)

Configured right now for the three clips in `data/videos/`:
`cam01.mp4`, `cam02.mp4`, `cam03.mp4` — 1280×720, 350 frames, 14 s each.
Preflight: **3 ready**.

These are the clips that actually produce plates. TfL's real footage is
352×288 and yields none, so this is the config to demo ANPR, consensus,
cross-camera identity and journeys on.

Nothing to install: Python 3.11 venv, fast-alpr 0.4.0, numpy 1.26.4,
onnxruntime 1.23.2, opencv 4.14, frontend `node_modules` all present.

Three terminals. Every one starts with the `cd`.

---

## Terminal 1 — check, then process

```bash
cd ~/Akshath/SIH
make doctor
make preflight
```

Expect `ready: 3`.

```bash
cd ~/Akshath/SIH
rm -f data/netra.db data/netra.db-wal data/netra.db-shm
rm -rf data/evidence/sightings
make seed
```

The reset matters here: the database currently holds four runs, two of them
from TfL cameras that produced zero plates. Mixing them into a plate-driven
demo makes the numbers confusing. `make seed` finishes on its own — roughly a
minute, and the first 20 seconds look idle while models load.

Expect around 5 sightings per camera, each with a plate, and CAM01→CAM02→CAM03
resolving into journeys.

```bash
cd ~/Akshath/SIH
make db
```

---

## Terminal 2 — API

```bash
cd ~/Akshath/SIH
make api
```

Stays running. http://localhost:8000 · docs at http://localhost:8000/docs

---

## Terminal 3 — dashboard

```bash
cd ~/Akshath/SIH
make dash
```

Stays running. http://localhost:5173

---

## Live tiles

```bash
cd ~/Akshath/SIH
make run
```

**This one does not stop on its own.** `loop: true` in the demo
`cameras.yaml` plus `mode: realtime` in `system.yaml` means it behaves as a
live feed — the clips restart forever and the tiles keep moving. That is the
point of it, and it is what you want running behind the dashboard while you
present.

Ctrl-C in that terminal, or from anywhere:

```bash
cd ~/Akshath/SIH
make stop
```

Do not wait for it to finish. It will not.

---

## What to show, in order

1. **Live** — tiles moving, boxes and track IDs, counts climbing.
2. **Trace** — pick a plate, watch the route animate CAM01 → CAM02 → CAM03.
3. **Double-click a vehicle** — the case view, with every OCR read behind the
   consensus plate and why the confidence is what it is.
4. **Unclear Possible Leads** — the pairs the resolver refused, still kept and
   still inspectable. This is the honesty point: nothing is deleted just
   because it scored low.
5. **Analytics** — corridor travel times, congestion index, alerts.

---

## Switching footage

To the real TfL corridor (8 cameras, detection and tracking, **no plates**):

```bash
cd ~/Akshath/SIH
cp configs/cameras.yaml.tfl-backup configs/cameras.yaml
cp configs/city_graph.yaml.tfl-backup configs/city_graph.yaml
make preflight
```

Back to these three clips:

```bash
cd ~/Akshath/SIH
.venv/bin/python backend/scripts/fetch_tfl.py --restore
make preflight
```

Reset the database and re-seed after either switch.

---

## Checks

```bash
cd ~/Akshath/SIH
make test
```

9 suites. Expect 280 passing and **one** known failure in `test_identity.py` —
"and says it is capped" — which predates this work and fails identically on an
untouched copy of the code.

---

## If something is wrong

| symptom | cause |
|---|---|
| `make: *** No rule to make target` | you are not in `~/Akshath/SIH` |
| `make run` never finishes | correct — `loop: true`. Use `make stop` |
| dashboard loads but is empty | `make seed` has not run, or the API is not up |
| plates look wrong | check `anpr.ocr_model` is `cct-s-v2-global-model`; a 9-slot model truncates every 10-character plate |
| first 20 s look frozen | models loading |
