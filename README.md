# NETRA — Network-wide Trajectory & Recognition Analytics

City-wide multi-camera ANPR, cross-camera vehicle trajectory reconstruction and
urban traffic analytics. Built for **Smart India Hackathon — PS 127**.

The full technical plan (architecture, roadmap, data strategy, demo script)
lives in [`docs/NETRA-Build-Plan.pdf`](docs/NETRA-Build-Plan.pdf).

---

## Quick start

```bash
make setup                  # create .venv, install backend deps
make sample                 # generate synthetic test footage
make cameras                # list configured cameras
make preview CAM=CAM01      # play a camera feed with the live HUD
make detect CAM=CAM01       # run vehicle detection on a feed
make track CAM=CAM01        # detect + track + count line crossings
make process CAM=CAM01      # full pipeline: detect, track, count, persist
make db                     # what is in the event store
make test                   # logic tests -- no video needed
```

### Before you demo

```bash
make doctor          # preflight -- fix every FAIL, run it again before you present
make warm            # download all models now, not in front of judges
make cache-tiles     # map works without the venue's wifi
make seed            # dashboard opens populated, not empty
```

### The demo

```bash
make demo            # seeds if empty, starts workers, runs the API
make dash            # second terminal: dashboard on :5173
```

Or three terminals: `make run`, `make api`, `make dash`.

**The full runbook, with the six-minute script and what to do when something
breaks, is in [`docs/DEMO.md`](docs/DEMO.md).**

Press `q` or `Esc` to close the preview window.

Everything is free and offline — no API keys, no accounts, no paid services.

---

## Where things live

```
configs/          every tunable number in the system (never hardcode)
  system.yaml       paths, runtime mode, device, log level
  cameras.yaml      the camera registry — add a camera, no code changes
data/videos/      camera footage (gitignored)
docs/             the build plan and reference screenshots
backend/app/
  core/             config loading, logging, timing helpers
  vision/           video ingest, HUD overlay, sample footage generator
  cli.py            the command line entry point
```

## Two ideas worth knowing before you read the code

**1. Config drives everything.** Adding a camera to the city means adding a
block to `configs/cameras.yaml`. No code changes, ever. That is what makes the
"could this run on 50 cameras?" answer honest.

**2. Every frame carries a virtual wall-clock timestamp.** Recorded clips have
no real timestamps, but cross-camera association is built entirely on *when* a
vehicle passed a camera. So `VideoSource` maps each clip's internal timeline
onto a shared city clock, plus that camera's `clock_offset_s`. Three separately
recorded clips then behave like three cameras watching one timeline:

```
CAM01  offset   +0s   →  clock 07:17:55.880
CAM02  offset  +45s   →  clock 07:18:41.880
CAM03  offset +110s   →  clock 07:19:46.880
```

This is why a vehicle can leave CAM01 at 10:42:13 and arrive at CAM02 at
10:46:51 in a way the trajectory engine can reason about later.

---

## Vehicle detection (M2)

```bash
make detect CAM=CAM01
python -m app.cli detect --camera CAM01 --source clip.mp4 --save
```

The model is a config value (`configs/system.yaml` → `detection.model`), so
`yolo11n.pt` → `yolo26n.pt` → a fine-tuned model is a one-line change. Weights
download once into `data/models/` (~5 MB, free, no account).

Nothing outside `vision/detector.py` imports ultralytics. The rest of the
pipeline sees only our own `Detection` objects, so replacing the detector later
touches exactly one file.

**Known limitation, stated plainly:** the pretrained weights are trained on
COCO, which has **no auto-rickshaw class**. India's most common three-wheeler is
absorbed into car / truck / motorcycle or missed. Fixing it means fine-tuning on
an Indian vehicle dataset — a tracked nice-to-have. We report what the model
actually predicts and say so in the presentation rather than papering over it.

---

## Tracking and counting (M3)

```bash
make track CAM=CAM01
make test                  # 24 logic tests, no video or model needed
```

Detection answers *what is in this frame*. Tracking answers *is this the same
vehicle as last frame* — and everything downstream depends on that second
answer. Without stable IDs there is no such thing as a vehicle **sighting**,
only unrelated boxes: the plate consensus in M5 has nothing to vote over, and
the cross-camera association in M7 has nothing to associate.

**ByteTrack via ultralytics**, not a hand-rolled tracker. One config value,
genuinely strong, and writing our own would cost days and still lose.

Two pieces of our own logic sit on top:

- **Confidence-weighted class voting.** A vehicle flickers between "car" and
  "truck" across frames. We keep weighted votes and report the winner, so one
  bad frame cannot relabel a vehicle. Same principle as plate consensus in M5,
  introduced early because it matters just as much here.
- **Track lifecycle.** Tracks are born, live, go unseen, then close out. A
  closed track is exactly one sighting record — `Track.to_sighting()` is the
  contract M4 persists.

Counting lines live in `cameras.yaml` in **normalised coordinates** (0–1), so
the same line keeps its meaning if the source resolution changes. A vehicle is
counted once and once only, with a signed direction — which is what makes the
origin–destination matrix possible in M9.

**Expect ID switches.** In testing, 3 vehicles produced 11 raw track IDs; the
`min_frames` filter reduced that to 5 completed tracks. This is normal tracker
behaviour under occlusion, not a bug — and it is exactly why M7 associates on
plate + appearance + physics rather than trusting track IDs.

---

## The event store and the bus (M4)

```bash
make process CAM=CAM01      # detect -> track -> count -> publish -> persist
make db                     # row counts and breakdowns
python -m app.cli db sightings --limit 20
python -m app.cli db plate KA01AB1234
```

### Why there is a bus

Every hackathon scalability slide says "we would add Kafka here", and every
judge knows that means a rewrite. `core/bus.py` is about eighty lines and it is
the difference between claiming that and showing it.

The camera worker publishes a sighting and does not know a database exists.
Today a `LocalBus` hands it to whoever subscribed — the store writer now, the
WebSocket feed in M6. In production a `KafkaBus` implementing the same two
methods puts it on a topic instead. **Nothing above or below changes.** That is
also why workers are separate processes: the boundary is already where the
network would go.

### Why SQLite is genuinely enough

SQLite allows one writer at a time, which sounds fatal for N camera workers. It
is not, because of a Milestone 0 decision: **consensus happens in the worker**,
so a camera writes one row per *completed track*, not per frame. Three cameras
produce single-digit writes per second. WAL mode plus a busy timeout absorbs
that easily, and the models are unchanged when the URL points at Postgres.

### The sighting is the atom

One vehicle, seen once, by one camera. Everything downstream — identity
resolution, trajectories, analytics — is built by relating sightings to each
other. Each carries its **virtual wall-clock times** from M1, which is what
makes cross-camera reasoning possible at all.

`run_id` makes re-processing safe: track IDs restart at 1 every run, so without
it a second run would silently double your vehicle counts. Re-run with the same
`--run-id` and every row is recognised as a duplicate and skipped.

---

## ANPR with temporal consensus (M5)

```bash
make process CAM=CAM01                              # ANPR runs inside the pipeline
python backend/scripts/ocr_bakeoff.py data/plates   # pick the OCR model by measurement
python -m app.cli db plate KA01AB1234
```

### We never trust a single frame

That sentence is the milestone. A vehicle is in view for 40–90 frames; OCR on
each gives a spread of readings, most right, some off by a character, a few
nonsense. Taking the single highest-confidence read throws away what every
other frame knew.

Instead we vote **per character position, weighted by the OCR's own
confidence** — and fast-alpr gives a confidence *per character*, so one shaky
character does not drag down the eight the model was sure of. Twelve frames
saying "4" at position 8 outvote one frame saying "A".

The output is not a string, it is evidence: consensus text, per-character
confidence, how many reads backed it, whether it satisfies the plate format,
and what the vote said *before* repair. M7 needs to know how much to trust a
plate, not merely what it says.

### The plate grammar is the cheapest big win

`vision/plate_grammar.py` is ~200 lines, no model, no training, and does more
for accuracy than swapping OCR engines would.

- **Format as a constraint.** `AA 00 A(A)(A) 0000`. A "0" where a letter must
  be is almost certainly "O". Positional repair turns near-misses into hits:
  `KA0LAB12E4 → KA01AB1234`, two fixes in one read.
- **OCR errors are not random.** "0"/"O" are confused constantly; "0"/"W"
  never. Weighted edit distance charges 0.25 for a known-confusable swap and
  1.0 otherwise. Near-miss similarity 0.95; genuinely different plates 0.60.
  That gap is what M7 associates on.
- **We only repair towards plates that could exist.** A candidate must have a
  real state code. Without that rule, repair converts `ZZZZZZZZZZ` into
  `ZZ22ZZ2222` — structurally valid, entirely invented. Confident nonsense is
  the exact failure this project exists to avoid.

### Every OCR attempt is kept

`plate_reads` stores each individual read against its sighting. Being able to
show *"we read this plate 37 times, here is the vote"* is worth far more to a
judge than a bare string — and it is what makes a wrong answer diagnosable
instead of mysterious.

### Running it on your own footage

ANPR reads the **vehicle crop**, not the whole frame — faster and more
accurate. Vehicles narrower than `min_vehicle_width_px` and plates narrower
than `min_plate_width_px` are skipped outright: a 40px plate wastes the call
and poisons the vote. Tune both in `system.yaml` against your real camera
angles, and use the read count shown on each track box to check you are getting
enough evidence per vehicle.

---

## Multi-camera orchestration and the dashboard (M6)

**From here the system is always demoable.** Every later milestone adds
capability to something already running, rather than working towards a first
run. If time runs out, the demo degrades instead of disappearing.

### One OS process per camera

Python's GIL means threaded workers serialise on inference — three cameras
would run at one camera's throughput. Separate processes each own their model
and decoder and genuinely run in parallel. That is the honest engineering
reason; the architectural one matters more: **this is the boundary a real
deployment cuts along.** The same `run_camera` function runs on an edge box
beside the physical camera, with a Kafka bus instead of the local one.

A worker that dies is reported and the others carry on. During a demo, one dark
camera tile is far better than a crash-looping restart storm.

### How live video crosses the process boundary

Each worker writes its latest annotated frame to `data/live/<CAM>.jpg`
(write-then-rename, so a reader never sees half a frame). The API re-serves it
as MJPEG. Crude, but it crosses a process boundary with **no IPC to fail in
front of judges** — and in a real deployment this is exactly where an RTSP or
WebRTC relay would sit.

### How the event feed works

The WebSocket tails the database for rows newer than the last one it sent,
rather than wiring an in-memory queue between processes. So a browser that
connects late still sees a backlog, a worker restart never drops the socket,
and the whole thing costs one indexed query every half second. The client
reconnects with backoff — a demo has to survive the backend restarting.

### API

`http://127.0.0.1:8000/docs` is generated automatically and is worth showing a
judge.

| Endpoint | Returns |
|---|---|
| `GET /api/health` | liveness and database path |
| `GET /api/cameras` | the network, with a live flag per camera |
| `GET /api/stats` | KPI row: sightings, plates, OCR attempts, breakdowns |
| `GET /api/sightings` | recent sightings, filterable by camera |
| `GET /api/plates/{plate}` | every sighting of one plate, in time order |
| `GET /api/live/{camera}` | MJPEG preview |
| `WS /ws/events` | new sightings pushed as they land |

`/api/plates/{plate}` already returns ordered hops across cameras — M8 keeps
the endpoint and adds distances, durations and route confidence.

---

## Cross-camera identity resolution (M7)

```bash
make resolve                          # link sightings into vehicles
python -m app.cli db identities --verbose
python -m app.cli db rejects          # what we refused, and why
make eval                             # precision/recall vs your ground truth
```

### The problem

A vehicle leaving CAM01 and arriving at CAM03 is one physical object, but the
system sees two unrelated sightings and two plate strings that may differ by
two characters. Naive matching fails **both ways**: it misses real matches
(`MH12DE1433` read as `MH12OE1433`) and invents false ones (two different
vehicles degrading to the same garbage).

### Three signals, one of which can veto

| Signal | Role | What it is |
|---|---|---|
| **Plate** | evidence | confusion-aware similarity, discounted by how sure the OCR was |
| **Appearance** | corroboration | vehicle class, dominant colour, box aspect |
| **Physics** | **constraint** | does the camera graph permit this journey in this time? |

Physics is not a third opinion — it is a veto. A vehicle cannot cross 4.4 km in
eight seconds, and no amount of plate agreement makes that link real:

```
plausible  CAM01->CAM02 in 4 min  ->  LINK   0.97
      plate 1.00 (trust 0.88), appearance 1.00, 24 km/h over 1.6 km

impossible CAM01->CAM03 in 8 s    ->  REJECT 0.00
      plate similarity 1.00, but implies 3168 km/h over 4.4 km,
      above the 70 km/h ceiling

near-miss  1 char, 4 min          ->  LINK   0.95
      plate 0.97 (trust 0.88), appearance 1.00, 24 km/h over 1.6 km
```

**Show the rejection during the demo.** It is the clearest evidence that the
system reasons rather than pattern-matches, and it is stored in
`identity_links` precisely so you can pull it up on demand.

### We never assert an identity without a plate

Appearance is corroboration, not identification — two silver hatchbacks are not
the same car. `plateless_cap` therefore sits **below** `accept_threshold`, so a
plate-less pair is scored, recorded and visible, but never merged.

This was not the original design. The first end-to-end run merged nine
sightings into one "vehicle" because the cap sat *above* the threshold and did
nothing. The unit test had checked that a cap existed, not that it bit. Both
the code and the test are fixed, and `require_plate: false` is what a learned
re-ID embedding would earn us later — it should not be flipped before then.

### Journey confidence is the weakest hop

Not the average. A three-camera journey with one shaky link is reported as
shaky, because that is what it is.

### Keep the config coherent

`clock_offset_s` in `cameras.yaml` implies travel speeds against the distances
in `city_graph.yaml`. If they disagree, every link is correctly rejected as
impossible and the demo shows nothing but refusals. Measure your real road
distances on a map — ten minutes, and it is the difference between a
plausibility gate that means something and one that does not.

### Proving it works

`make eval` scores **pairs**, not clusters, against
`data/groundtruth/journeys.csv` — one row per pass, written down while you
film. Pair counting is the honest metric: one wrong merge that joins two
journeys is far worse than one missed hop.

```
precision     of the links we asserted, how many were right
recall        of the links that existed, how many we found
```

Quote those numbers with the sample size, and nothing more.

---

## Trajectory reconstruction (M8)

```bash
make resolve                              # identities AND journeys
make journey PLATE=KA01AB1234
python -m app.cli journey                 # every journey
```

M7 gives a *set* of sightings believed to be one vehicle. That is not yet a
journey. Three things have to happen:

### 1. Chain, don't fan out

For a vehicle seen at CAM01, CAM02 and CAM03 the resolver accepts three links —
01→02, 02→03 **and** 01→03. Only the first two are hops. A journey is the chain
of *consecutive* sightings in time order; the shortcut link is corroborating
evidence, not a leg of the trip.

### 2. Route, don't just connect

Two consecutive sightings are joined by a **road**, and routing over the
declared network recovers the cameras in between:

```
01AB12334   confidence 0.83
  16:55:59  CAM01 -> CAM03  via CAM02 (unobserved)
             10.5 min     4.7 km    27 km/h (usual 28)   score 0.83
  route     CAM01 -> CAM02 -> CAM03
  total     4.7 km   10 min 32 s   27 km/h average
  gaps      CAM02 on the route but never saw it
```

That last line is a **coverage gap** — a camera the vehicle certainly passed
that recorded nothing. Miscalibration, a missed detection, or an unreadable
plate. It is a finding a city would act on, and it falls out of the graph for
free.

Routing uses **declared edges only**. A derived edge is a straight-line guess
between two cameras, not a road; routing over one would invent junctions that
do not exist. So declare road *segments* in `city_graph.yaml`, not every camera
pair — that is both more realistic and what makes gap detection possible.

### 3. Measure, don't just draw

Each hop carries distance, duration, implied speed, and how that compares to
the corridor's typical speed. `delay_ratio ≥ 1.5` marks a leg taken markedly
slower than usual — **congestion observed from real travel times, not inferred
from vehicle counts.** That is the raw material for M9, and it is a
better signal than counting: forty vehicles moving freely is not congestion,
and four vehicles crawling is.

Journey confidence is the **weakest hop**, not the average. A three-camera trip
with one shaky link is reported as shaky, because that is what it is.

---

## Analytics and alerts (M9)

```bash
make analytics                    # metrics, congestion, alerts
python -m app.cli analytics --verbose
```

Every figure here comes from something the system observed. Nothing is
modelled, extrapolated or filled in, and where there is not enough data to say
something the report **says so** instead of producing a confident number from
three samples.

### Congestion is measured from travel time, not vehicle count

Forty vehicles moving freely is not congestion; four crawling is. Counting
cameras can only tell you how many passed. A system that reconstructs journeys
can tell you how long they took — which is the thing a traffic controller
actually needs.

The free-flow baseline is taken **from the data itself** (the 15th percentile
of observed travel times on that corridor), falling back to the configured
typical speed when there are too few trips to trust a percentile. Which one was
used is printed, because a baseline from four trips is not a baseline.

```
CORRIDOR          TRIPS    KM   MEDIAN  FREEFLOW  INDEX  LEVEL
CAM01->CAM02          2   1.6     3.9m      3.7m   1.06  free flow  (baseline from config)
CAM02->CAM03          2   3.1     6.6m      5.8m   1.13  free flow  (baseline from config)
```

### Plate rate is a camera-quality signal, not a traffic one

A camera reading 20% while its neighbours read 70% is badly sited or out of
focus. That belongs in the report, and it is the kind of finding a city would
act on.

### Four alert rules, each with its value and threshold

| Rule | Fires when | Note |
|---|---|---|
| `congestion` | median travel time ≥ 1.5× free flow | never from fewer than 3 trips |
| `overspeed` | hop above the **legal** limit | distinct from the plausibility ceiling |
| `watchlist` | plate similar to a listed vehicle | fuzzy, flagged for review |
| `coverage_gap` | a camera repeatedly missed vehicles that passed it | a camera fault, not traffic |

Two distinctions worth stating to a judge:

**Impossible vs illegal.** `max_speed_kmph` is the plausibility ceiling used to
*reject* journeys in M7; `speed_limit_kmph` is the legal limit that raises an
*alert*. They are different numbers doing different jobs, and the alert says
plainly that its figure is a segment average, not a radar reading.

**A watchlist hit is flagged, never asserted.** Matching is fuzzy because OCR
misreads a character often enough that exact matching would miss real hits —
but that makes every hit a hypothesis. Verified behaviour:

```
read 01AB12334    -> HIT 01AB12334 at 1.00
read 01AB1233     -> HIT 01AB12334 at 0.89   (a dropped character)
read O1AB12334    -> HIT 01AB12334 at 0.97   (O/0 confusion)
read 122DE143     -> no match
```

Every alert reports the similarity that produced it and says a human should
confirm. A system that announces "stolen vehicle detected" on a 0.86 match is a
system that gets someone stopped for nothing.

### Refusing to fire is a feature

On the demo dataset the congestion rule raises **nothing**, because each
corridor has only two trips and the minimum is three. That is the rule working.
A dashboard that cries wolf is worse than one that says nothing.

---

## The command dashboard (M10)

```bash
make dash-setup      # once: npm install (adds leaflet + recharts)
make run             # terminal 1
make api             # terminal 2
make dash            # terminal 3 -> http://localhost:5173
```

Three views:

**Live** — city map with camera nodes sized by volume and corridors coloured by
measured congestion, camera feeds below, alerts and the detection feed on the
right.

**Trace** — type a plate, watch the journey animate across the network, with a
hop-by-hop timeline: departure and arrival, distance, implied speed against the
corridor's usual pace, and the confidence of each link. Coverage gaps are called
out in red on the map and in the timeline.

**Analytics** — volume over time, corridor congestion, camera utilisation,
origin→destination, and the **links the system refused**.

### Colour is computed, not chosen

The vehicle-class palette was validated against this dashboard's own surface
(`#141C24`) rather than picked by eye:

```
lightness band PASS · chroma floor PASS · CVD separation PASS (worst
adjacent ΔE 8.4) · normal-vision floor PASS (19.8) · contrast PASS
```

The same hues are used in the **video overlay and the charts**, so a bus is the
same colour wherever you see it. Congestion and alert severity use a reserved
status palette that is never used for a series, and always ships with a text
label — meaning never rests on hue alone.

### Free, keyless map

CARTO dark basemap tiles: no API key, no account, no card. That was the M0
reason for rejecting Mapbox and Google Maps, and it still holds.

### Two things worth knowing before you demo

**Workers take 15–20 seconds to load models** before the first frame appears.
Start `make run` before you open the dashboard, or the first thing a judge sees
is three idle tiles.

**The map needs internet for tiles.** M11 pre-caches them so the demo survives
venue wifi.

---

## Demo hardening (M11)

Four things that protect the demo, each aimed at something that actually went
wrong during development.

### `make doctor` — the preflight

Twenty checks in ten seconds. Dependencies, inference device, camera footage,
declared road edges, cached model weights, database writability, whether the
database is seeded, map tile cache, ports already in use, disk space.

One check is worth calling out: it verifies that **`clock_offset_s` in
`cameras.yaml` implies physically possible speeds against the distances in
`city_graph.yaml`.** That exact misconfiguration made every cross-camera link
get correctly rejected during M7, and the symptom — a demo that finds no
journeys at all — looks nothing like its cause.

### `make seed` — never open on an empty screen

Processes every camera at full speed, resolves identities and journeys, computes
analytics and alerts. One command from empty database to a dashboard full of
data. Then run the live pipeline on top for the "watch it happen" moment.

### `make cache-tiles` — the map survives venue wifi

The API serves map tiles from a local cache, fetching and storing on a miss. So
it works online, works fully offline once cached, and when a tile is neither
cached nor reachable it returns a dark tile the colour of the map background —
the map goes **plain instead of broken**, with the camera network and journeys
still drawn on top. Verified with the network blocked:

```
GET /api/tiles/13/5891/3648.png  ->  200, x-tile-source: fallback
```

### `GET /api/health` — is this thing ready?

```json
{ "seeded": true, "sightings": 9, "tiles_cached": 0, "offline_ready": false }
```

### One thing no script can fix

Workers take **15–20 seconds to load models** before the first frame appears.
Start them before you open the dashboard, or the first thing a judge sees is
three idle tiles.

---

## The presentation package (M12)

```
docs/NETRA-SIH-PS127.pptx     16-slide deck, editable
docs/architecture.png          the pipeline diagram, for reuse
docs/scaling.png               prototype → production
docs/RESULTS.md                measured numbers — regenerate with `make report`
docs/DEMO.md                   the six-minute runbook
```

### No number on a slide is typed by hand

`make report` computes every figure from the database as it stands and writes
`docs/RESULTS.md`. Where a measurement has not been made it prints **"not yet
measured"** and names the command that would produce it — it never fills the
gap with something plausible. A judge who asks "where did that number come
from?" should get an answer, and *"we ran this script"* is one.

Right now it correctly reports that **association precision and recall are not
yet measured**, because that needs your ground-truth file. Everything else —
15 OCR reads behind each agreed plate, mean journey confidence, links accepted
and refused — comes straight from the system's output.

### The deck's argument, in order

1. Every camera is an island — 400 cameras, none of which can trace a journey
2. The hard part is **identity continuity under uncertainty**, not detection or OCR
3. Three signals, one of which can **veto**: plate, appearance, physics
4. Architecture — every boundary is where the network goes in production
5. We never trust a single frame
6. The plate grammar: 200 lines, no model, more accuracy than swapping OCR engines
7. Measured results, with the gaps named
8. Impact, scalability, feasibility (₹0)
9. **Limitations, stated before anyone asks**

The limitations slide is deliberate. Being caught by a limitation reads as
naive; naming it first reads as rigour.

---

## Playback modes

| Mode | What it does | Used for |
|---|---|---|
| `realtime` | sleeps so a 25-fps clip plays at 25 fps | the live demo |
| `fast` | no sleeping, runs as fast as the machine allows | pre-seeding the database |

Set the default in `configs/system.yaml`, or override per run with `--mode`.

`target_fps` in `cameras.yaml` controls how many frames actually reach the
pipeline. Detection does not need all 25 — 10–12 is plenty and much faster.

---

## About the synthetic footage

`make sample` writes three short clips with vehicles carrying legible number
plates. It exists so Milestones 2–5 can be built and debugged before anyone has
shot real footage, and four vehicles deliberately appear in all three clips so
there are cross-camera journeys to test against.

**It is scaffolding, not data.** It proves the pipeline is wired correctly. It
proves nothing about real-world accuracy, it never appears in the demo, and no
number in the presentation comes from it. Real recorded footage replaces it.

### It also cannot test detection

Worth knowing before you waste an hour on it: YOLO returns **zero detections**
on the synthetic clips, and that is the correct answer. The vehicles are flat
coloured rectangles, not cars. The synthetic footage tests timing, looping,
config and the virtual clock — everything *around* the model. It cannot test
the model.

**From Milestone 2 onward you need real video.** The fastest way to get some:
stand at a roadside or a window and record 60 seconds on your phone at 1080p.
That is a five-minute job and it unblocks everything.

```bash
# drop any clip in and point a command at it -- no config edit needed
python -m app.cli detect --camera CAM01 --source ~/Downloads/traffic.mp4
```

Once you are happy with a clip, move it to `data/videos/` and set it as the
`source` in `configs/cameras.yaml`.

---

## Status

- [x] **M1** — skeleton, config, logging, video pipeline
- [x] **M2** — vehicle detection
- [x] **M3** — tracking and line counting
- [x] **M4** — event store and bus interface
- [x] **M5** — ANPR with temporal consensus
- [x] **M6** — multi-camera orchestration, first demoable build
- [x] **M7** — cross-camera identity resolution
- [x] **M8** — trajectory engine
- [x] **M9** — analytics and alerts
- [x] **M10** — command dashboard
- [x] **M11** — demo hardening
- [x] **M12** — presentation package
