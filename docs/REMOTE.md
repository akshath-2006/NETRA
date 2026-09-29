# Processing somewhere more powerful

How to run NETRA's expensive half on another machine and get the results back
onto the dashboard — and why it is done this way rather than another way.

---

## The finding that shaped the whole approach

NETRA was already split down the right line, and nobody had noticed.

| command | what it does | what it touches |
|---|---|---|
| `run` | decode, detect, track, ANPR | **writes** `data/netra.db`, `data/evidence/` |
| `resolve` | cross-camera identity, journeys | **writes** the same |
| `analytics` | metrics, congestion, alerts | **writes** the same |
| `serve` + dashboard | shows all of it | **reads only** |

The expensive half and the visible half already communicate through files on
disk. Not a socket, not a shared process, not an in-memory queue. So "do the
heavy work somewhere else" needed **no architectural change at all** — no new
service, no job queue, no RPC layer, no rewrite. It needed three things:

1. a way to know the footage is worth processing before starting (`preflight`),
2. a way to run the four commands headlessly on any machine (`remote_batch.py`),
3. a way to carry the results back (`export` / `import`).

That is the entire remote-execution feature. The pipeline, the schema, the
event bus, the API and the frontend are untouched.

---

## Why not just pick Colab

Colab was the obvious answer, which is the reason it got checked rather than
assumed. What the free options actually offer:

| option | GPU | cost | the catch |
|---|---|---|---|
| **Google Colab** free | T4, when available | ₹0 | no guarantee of a GPU; idle disconnects; ~12 h ceiling; storage vanishes unless you mount Drive |
| **Kaggle Notebooks** | P100 or 2×T4 | ₹0 | 30 GPU-hours/week, but **guaranteed** when you have quota; 12 h sessions; datasets persist |
| **A lab or college GPU box over SSH** | whatever it has | ₹0 | you need one |
| **Rented GPU** (vast.ai, RunPod) | anything | costs money | ruled out — the budget is zero |
| **This MacBook** | none (Intel, no CUDA, no MPS) | ₹0 | it is the machine we are trying to get off |

None of them is reliably the best, and two of them can refuse you a GPU on the
day. So the design deliberately **does not choose**:

* `remote_batch.py` and `remote_setup.sh` are plain Python and bash. They run
  identically on a laptop, an SSH box, Colab, Kaggle or a VM.
* `notebooks/NETRA_remote.ipynb` is a thin wrapper over those two scripts that
  detects whether it is on Colab or Kaggle and adjusts only the upload and
  download steps.

If a provider disappears tomorrow, one notebook cell changes. Nothing in
`backend/` knows a provider exists.

---

## Running it

### On any machine with a terminal (SSH box, VM, your own laptop)

```bash
bash backend/scripts/remote_setup.sh              # installs deps, proves the GPU is visible
python backend/scripts/remote_batch.py --workers 4 --max-frames 0
```

`remote_batch.py` runs `preflight → run → resolve → analytics → export`,
times each phase, and writes `data/exports/netra-<timestamp>.tar.gz` plus a
JSON report. It calls the same `cmd_*` functions the CLI calls, so a remote
run cannot drift from a local one.

### On Colab or Kaggle

Open `notebooks/NETRA_remote.ipynb`, turn the GPU on, and work down the cells.
`make bundle-src` on your laptop produces the zip the notebook asks for.

### Back on the laptop

```bash
python -m app.cli import data/exports/netra-20260924-1830.tar.gz
make api      # terminal 1
make dash     # terminal 2
```

---

## What `export` / `import` actually guarantee

**Export** takes a consistent snapshot with SQLite's `VACUUM INTO`, not a file
copy. The database runs in WAL mode, so copying `netra.db` alone silently
loses the tail of the run — the worst kind of data loss, because everything
still opens and looks fine. The archive holds the database, every evidence
snapshot, the configs that produced them, and a manifest recording which
machine ran it and with what device.

**Import merges by run, and is idempotent.** Every table except `cameras` has
an autoincrement primary key, so a row's id on the remote machine means
nothing on the laptop. Import renumbers everything and rewrites every foreign
key as it goes; a row whose target did not come across is **dropped, never
repointed**, because a journey assembled from a link with one end missing is a
fabricated journey. A run already in your database is skipped rather than
duplicated, so importing the same bundle twice is safe.

Verified in `backend/tests/test_bundle.py` (32 checks) by fingerprinting
*meaning* rather than ids: that every link still joins the same two sightings,
every journey still owns its own hops, every OCR read is still attached to the
vehicle it was read from, and the local results are untouched.

`--dry-run` reports what would happen and writes nothing.
`--mode replace` swaps the database instead, keeping a dated backup first.

---

## Getting the GPU actually used

Two engines, two different stories, and one of them fails silently.

**Vehicle detection** follows the GPU by itself. `resolve_device("auto")` asks
torch, and ultralytics does the rest. Nothing to configure.

**ANPR does not.** It runs on ONNX Runtime, whose CPU build (`onnxruntime`)
and GPU build (`onnxruntime-gpu`) are **different packages providing the same
`import onnxruntime`**. Install the wrong one — or both — and ANPR runs on the
CPU while reporting no error at all. Since ANPR is 31% of measured runtime,
that turns a "GPU run" into a mostly-CPU run with no warning.

So:

* `remote_setup.sh` uninstalls the CPU wheel before installing the GPU one,
  and prints the providers it ends up with.
* `remote_batch.py` prints both engines' devices before it starts, and says
  outright when detection is on the GPU but ANPR is not.
* `anpr.providers: auto` in `system.yaml` intersects what you asked for with
  what the installed runtime actually has, so a config copied back from the
  GPU box does not stop ANPR dead on the laptop.

---

## Choosing `--workers` on a GPU

On the laptop, workers are limited by cores. On a single GPU they are limited
by **VRAM**: each worker holds its own YOLO plus two ONNX sessions. Two to
four is usually right on one T4 regardless of how many cores the box reports,
and the failure mode for going too high is not slowness but an out-of-memory
crash partway through the run.

Set it with `--workers`, or `performance.max_parallel_cameras` in
`system.yaml`.

---

## What this does not do

* It does not stream results back live. The remote machine finishes, you
  import. A live feed would need the Kafka bus the event-bus abstraction was
  written for, and that is a real project, not a flag.
* It does not upload your footage for you. Video is the big thing; how it gets
  to the other machine (Drive, a Kaggle dataset, `scp`) is yours to choose.
* It does not make a free GPU appear. If the notebook hands you a CPU runtime,
  `remote_batch.py` says so in its first five lines rather than letting you
  find out from the clock.
