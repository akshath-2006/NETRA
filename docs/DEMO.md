# NETRA — demo runbook

Six minutes, one uninterrupted narrative. Rehearse it three times before you
present; the third run is where the surprises show up.

---

## The night before

```bash
make doctor          # fix every FAIL. warnings are weaker-demo, not broken-demo
make warm            # download all models now, not in front of judges
make cache-tiles     # map works without the venue's wifi
make seed            # dashboard opens populated, not empty
make test            # 262 logic tests, ~5 seconds
```

`make doctor` again the morning of, and once more ten minutes before you go on.
It catches the specific things that have actually broken during development:
missing dependencies, uncached models, a port still held by yesterday's process,
and clock offsets that imply impossible speeds.

## Starting up

Three terminals, in this order:

```bash
make run     # 1 — camera workers. WAIT for the first frames (15-20s of model loading)
make api     # 2 — API on :8000
make dash    # 3 — dashboard on :5173
```

Or one command: `make demo` (seeds if needed, starts workers, runs the API),
then `make dash` in a second terminal.

**Do not open the dashboard before the workers have warmed up.** The first thing
a judge should see is three streaming tiles, not three idle ones.

---

## The run

| Time | Beat | What you do |
|---|---|---|
| 0:00 | **The gap** | "This city has 400 cameras and 400 blind spots. Each one sees a moment. None of them sees a journey." One line, no preamble. |
| 0:30 | **Open live** | Dashboard already populated — three tiles with detection boxes, KPIs counting, plate reads scrolling in with confidence scores. Never an empty state. |
| 1:15 | **The pipeline** | Point at one tile: detected → tracked → plate read → **consensus across ~60 frames** → one sighting. Say the line: *"we never trust a single frame."* |
| 2:00 | **The money shot** | Trace tab. Ask a judge to pick a plate from the feed. Search it. The journey animates across the map with per-hop times, distances, speeds and confidence. |
| 3:00 | **The credibility moment** | Analytics tab → *Links the system refused*. "Plate similarity 1.00, but that link implies 3168 km/h over 4.4 km. Physically impossible, so we didn't assert it." |
| 3:45 | **Why a city cares** | Corridor congestion measured from **travel time, not counts**. Camera utilisation — plate rate is a camera-quality signal. Coverage gaps: a camera the vehicle certainly passed that saw nothing. |
| 4:45 | **The numbers** | "On our own N-camera recorded dataset with ground truth: X% plate character accuracy, Y% association precision, Z% recall." *(`make eval` — see below.)* |
| 5:15 | **Scale and limits** | Prototype→production table. Then state the limitations before anyone asks. |

---

## The three things that win it

**1. Show a rejection.** Any system can link things. Showing what the system
*refused* to assert, and why, is the clearest evidence there is reasoning under
the dashboard rather than a lookup table. Almost no competing team demos a
negative result on purpose.

**2. Quote real accuracy numbers.** `make eval` scores association against
`data/groundtruth/journeys.csv`. Judges ask "how do you know it works?" and
almost nobody can answer. Quote the numbers *with the sample size*, and nothing
more.

**3. State the limitations first.** Plate readability depends on camera
placement; the prototype is evaluated on a small self-collected set; appearance
re-identification degrades badly at night; association confidence would need
re-calibration per deployment; a real deployment carries privacy and retention
obligations this prototype does not address. Saying it yourself reads as
confidence. Being caught by it does not.

---

## If something breaks

| Symptom | Do this |
|---|---|
| Camera tile idle | It is still loading models, or that worker died. `tail data/logs/workers.log`. The other cameras keep running — say so and carry on. |
| Map is plain dark | Tiles not cached and no network. The camera network and journeys still draw. `make cache-tiles` beforehand avoids it. |
| Dashboard empty | `make seed`. |
| "Address already in use" | `make stop`, then check `make doctor`. |
| Search finds nothing | `make resolve` — journeys are built by that command, not by the workers. |

**Never restart everything mid-demo.** Talk over the gap: the architecture slide
and the limitations both fill sixty seconds comfortably.

---

## What is real and what is not — say this plainly

Everything on screen is produced by the pipeline from the footage you fed it.
Nothing is mocked, and no number is invented.

The synthetic clips from `make sample` are **scaffolding for development**, not
demo material — YOLO correctly detects nothing in them. Demo with real recorded
footage, and say that it is recorded footage standing in for live CCTV. That is
the honest framing, and it is also the strong one: the architecture is the
claim, and it does not change when the source becomes RTSP.
