# NETRA — first run

Everything below is copy-paste. If a step fails, `make doctor` will tell you why.

---

## 0. What you need first

| | |
|---|---|
| **Python 3.10 or newer** | `python3 --version` |
| **Node 18 or newer** | `node --version` |
| **make** | `make --version` — on macOS, `xcode-select --install` if missing |
| **Internet** | only for the one-time installs and model downloads |

Everything the project uses is free. No API keys, no accounts, no cards.

---

## 1. Install — once, about 5 minutes

```bash
cd ~/Akshath/SIH

make setup        # creates .venv and installs the Python side
make dash-setup   # installs the dashboard's packages
```

`make setup` downloads PyTorch, which is large. Let it finish.

---

## 2. Get some footage

You have two options.

**A. Test footage, right now** — proves the plumbing works:

```bash
make sample
```

Writes three synthetic clips to `data/videos/`.

> **Important:** YOLO detects **nothing** in these clips, and that is correct —
> they are coloured rectangles, not cars. They exist so the pipeline can be run
> and debugged. They are not demo material.

**B. Real footage** — what you actually demo with. See section 7.

---

## 3. Check everything before you run it

```bash
make doctor
```

Twenty checks in ten seconds. Fix anything marked **FAIL**. Warnings are
"weaker demo", not "broken demo".

---

## 4. Download the models — once

```bash
make warm
```

Pulls the vehicle detector (~5 MB) and the ANPR models (~15 MB) into
`data/models/`. Do this now, not while a judge is watching.

---

## 5. Fill the database

```bash
make seed
```

Processes every camera at full speed, links vehicles across cameras, builds
journeys, computes analytics and alerts. Takes a couple of minutes.

This is why the dashboard opens **full of data** instead of empty.

---

## 6. Start it — three terminals

Open three terminal tabs, all in `~/Akshath/SIH`:

```bash
# Terminal 1 — the cameras
make run
```
Wait for the first log lines. **Models take 15–20 seconds to load.**

```bash
# Terminal 2 — the API
make api
```

```bash
# Terminal 3 — the dashboard
make dash
```

Then open **http://localhost:5173**

To stop: `Ctrl-C` in terminals 2 and 3, then `make stop`.

---

## 7. Using the dashboard

**Live** — the city map with your cameras, corridors coloured by congestion,
the camera feeds, alerts, and detections arriving in real time.

**Trace** — the important one. Click any journey in the right-hand list, or
type a plate and press **Trace**. The vehicle's path animates across the map and
you get a hop-by-hop breakdown: when it left, when it arrived, how far, how fast
against the corridor's usual pace, and how confident each link is.

**Analytics** — traffic volume, corridor congestion measured from travel time,
camera utilisation, origin→destination, and **the links the system refused**.
That last panel is your best slide: it shows what the system would *not* assert
and why.

---

## 8. Using your own footage

1. Put your clips in `data/videos/`.
2. Open `configs/cameras.yaml` and for each camera set:
   - `source:` the path to your clip
   - `location:` the real lat/lon where you filmed
   - `clock_offset_s:` how far apart the passes were, in seconds
3. Open `configs/city_graph.yaml` and set `distance_km` to the **real road
   distance** between each pair — measure it on a map, it takes ten minutes.

   > These two files must agree. If the offsets imply an impossible speed for
   > the distance, every cross-camera link is correctly rejected and you will
   > see no journeys at all. `make doctor` checks this for you.

4. Then:

```bash
rm -f data/netra.db*    # start clean
make doctor
make seed
```

---

## 9. Proving it works

Write down every vehicle pass while you film, into
`data/groundtruth/journeys.csv`:

```csv
vehicle,camera,time
KA01AB1234,CAM01,10:42:13
KA01AB1234,CAM02,10:46:51
KA01AB1234,CAM03,10:52:09
```

Then:

```bash
make eval      # precision and recall for cross-camera association
make report    # regenerates docs/RESULTS.md with every measured number
```

`make report` never invents a figure — where something has not been measured it
says so. That is what lets you quote numbers to a judge and defend them.

---

## 10. Before you present

```bash
make doctor        # again, ten minutes before
make cache-tiles   # map works without the venue's wifi
```

Then follow **`docs/DEMO.md`** — the six-minute script, and what to say if
something breaks.

---

## Every command

```
make help          list everything

SETUP     setup · dash-setup · sample · warm
CHECK     doctor · test
DATA      seed · resolve · journey · analytics · report · eval
RUN       run · api · dash · demo · stop
DEBUG     cameras · preview CAM=CAM01 · detect CAM=CAM01 · track CAM=CAM01 · db
```
