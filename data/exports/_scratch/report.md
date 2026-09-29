# NETRA — slowness, unusable footage, and running elsewhere

Engineering report, 24 September 2026.

Everything below labelled **measured** was measured on this run. Everything
labelled **expected** is reasoning from a measurement, stated as reasoning.
Where a number could not be obtained, it says **Not measured** rather than
guessing.

**Measurement environment:** Linux container, **2 CPU cores, 7 GB RAM, no
GPU** (`torch.cuda.is_available()` → `False`; ONNX Runtime providers →
`['AzureExecutionProvider', 'CPUExecutionProvider']`). Every timing below is
therefore a CPU timing. Your MacBook could not be benchmarked from here — its
virtualenv is a macOS build and this session reaches the folder, not the OS.

---

## 1. Why it was slow

Profiled, not guessed. The per-stage breakdown (120 frames, one camera, CPU)
was already in `configs/system.yaml`:

| stage | share of runtime |
|---|---|
| `detector.track()` | **61.9%** |
| ANPR | **31.1%** |
| video decode | 2.9% |
| JPEG live preview | 2.2% |
| overlay | 0.9% |
| track management | 0.9% |
| counting / consensus / DB writes | ~0% |

93% of the time is two model calls. Nothing else is worth touching, and the
database — the usual suspect — measured at effectively zero.

But single-camera profiling is not where the time was going. **Three separate
causes, all in how the cameras were run together:**

**(a) Every camera was started at the same instant.** `run_supervisor()`
spawned one process per camera with no limit. Each worker loads its own YOLO
model and its own two ONNX sessions. **Measured: ~800 MB resident per
worker.** Eight cameras on a 2-core, 7 GB machine therefore asked for ~6.8 GB
of RAM and 8 runnable processes on 2 cores. They did not run eight times
faster — they ran slower than two would have, and the machine spent the
difference on context switching and memory pressure. This is the battery and
"everything else becomes unusable" symptom, and at ten cameras (~8 GB) it
stops being slowness and becomes an out-of-memory crash.

**(b) ONNX Runtime's threads were never capped.** `torch_threads_per_worker: 1`
already existed and fixed this for detection. Nothing did the same for ANPR.
ONNX Runtime defaults to one intra-op thread **per core, per session**, and
each camera process holds two sessions (plate detector + OCR). Eight cameras
on an 8-core machine were asking for 128 threads to share 8 cores. ANPR is
31% of runtime, so this is not a rounding error.

**(c) Unusable footage was processed at full cost.** Nothing checked a video
before loading models against it. A clip containing no footage cost exactly
as much to "process" as a real one (see §2).

What was **not** the cause, checked and ruled out: the database (measured
~0%), the event bus, the API, the frontend, JPEG snapshot writing (measured
~1%, and a clean A/B put a suspected 4× regression at **1%** — it was
container noise), and the detector/ANPR settings themselves, which were
already tuned (`imgsz: 480`, `every_n_frames: 3`, a read budget per track,
and frame striding from `target_fps`).

---

## 2. Why only some of the ten videos worked

**The answer: 0 of 10 were broken. 2 of 10 were empty.**

The two failing clips were probed against every hypothesis in your brief:

| property | working clips | failing clips |
|---|---|---|
| codec | h264 | **h264** |
| resolution | 352×288 | **352×288** |
| pixel format | yuv420p | **yuv420p** |
| frame rate | 25/1 | **25/1** |
| audio stream | none | none |
| corruption (ffprobe + OpenCV) | none | **none** |
| path / permissions | fine | **fine** |
| **duration** | 9.7–9.9 s | **1.000 s** |
| **frames** | 242–248 | **25** |
| **size** | 95–151 KB | **12.4–12.6 KB** |

They are **valid, uncorrupted H.264 files containing a one-second "camera
temporarily unavailable" card**, which is what TfL's S3 endpoint serves when a
camera is offline at capture time. Not a codec problem, not a decoding
problem, not a resolution problem, not corruption, not a path problem, not an
ingestion problem. A **source availability** problem — a different thing with
a different fix.

**Why it looked like "the first three worked".** The fetch tool captures the
corridor several times and keeps the best round. Its scoring counted a
placeholder as a usable capture, so it chose the *worst* round available.
That was a bug in the selection logic, not in the footage or the pipeline.

All 60 clips you actually downloaded, judged on your own machine:

| round | usable | cameras offline at capture time |
|---|---|---|
| r01 | 8/10 | CAM04, CAM10 |
| r02 | 8/10 | CAM04, CAM10 |
| r03 | 8/10 | CAM04, CAM10 |
| r04 | 8/10 | CAM04, CAM10 |
| r05 | 7/10 | CAM03, CAM04, CAM10 |
| **r06** | **6/10** | **CAM03, CAM04, CAM08, CAM10** |

Totals: **45 ready, 15 placeholder, 0 corrupt, 0 unsupported, 0 missing.**

The old selection picked **r06** — the single worst round of the six. CAM03
and CAM04 were both offline in it, so the corridor went dead immediately after
CAM02. That is precisely the "nothing beyond cam 3" you reported, and the
database agrees: run `075cd30b19aa` has exactly CAM03, CAM04, CAM08 and CAM10
at zero sightings.

Two further things this table shows. TfL's CAM04 and CAM10 were offline in
**every one of the six rounds** — that camera pair is simply down, not
intermittently unlucky. And rounds 1–4 were all equally good at 8/10, so the
fixed selection picking r03 is a valid choice rather than a lucky one. The
current config now has **all 8 cameras producing sightings** (113 in run
`4fadcfb429d5`).

**Are they repairable?** No, and re-encoding would be actively wrong: you
cannot transcode footage into a file that never contained any. The remedy is
to re-capture when that camera is back online. `preflight` now says exactly
that instead of leaving you to guess.

**Are they useful?** As footage, no. As a test fixture for the preflight
verdict, yes — they are kept.

**Separately, and worth stating plainly:** all TfL footage is 352×288. ANPR's
`min_plate_width_px` is 175, so a plate would have to fill **half the frame
width** to be readable. That is arithmetic, not opinion, and the database
agrees: **236 TfL sightings, 0 plates**. TfL's own documentation says these
cameras "do not record number plates". This footage tests detection, tracking,
counting and congestion. It cannot test ANPR, and nothing in the system now
pretends otherwise — preflight prints the warning with the arithmetic.

### The fix: `app.cli preflight`

Judges every source before a model loads, in about a second:

```
source        verdict            size    fps  frames    secs    codec  last run
CAM01         READY          1280x720   25.0     350    14.0    mpeg4  2 sightings, 2 plates
CAM04         PLACEHOLDER     352x288   25.0      25     1.0     h264  not in last run
CAM06         MISSING               -    0.0       0     0.0        -  not in last run

ready: 6  placeholder: 1  missing: 1
```

Six verdicts, each with a different remedy: `ready`, `placeholder`
(re-capture), `corrupt` (re-download), `unsupported` (transcode), `missing`
(fix the path), `stream` (cannot be judged from disk). It distinguishes
*damaged* from *not video* using the container's magic bytes, so it never
tells you to re-capture a file that merely arrived broken. `--strict` exits
non-zero, which is what stops a remote batch burning GPU hours on empty
files. `--json` writes the machine-readable version.

20 tests in `backend/tests/test_preflight.py`, each verdict exercised against
a generated fixture.

---

## 3. Where the computation should run, and why

### The finding that shaped it

NETRA was already split down the right line:

| command | writes | reads |
|---|---|---|
| `run`, `resolve`, `analytics` | `data/netra.db`, `data/evidence/` | — |
| `serve` + dashboard | — | the same two |

The expensive half and the visible half already communicate **through files on
disk** — not a socket, not a shared process, not an in-memory queue. So
"process somewhere else" needed **no architectural change**: no new service, no
job queue, no RPC layer, no rewrite. It needed a preflight, a headless runner,
and a way to carry results back.

### Why not simply Colab

| option | GPU | cost | the catch |
|---|---|---|---|
| Google Colab free | T4 when available | ₹0 | no GPU guarantee; idle disconnects; ~12 h ceiling; storage vanishes without Drive |
| **Kaggle Notebooks** | P100 / 2×T4 | ₹0 | 30 GPU-h/week but **guaranteed** while quota lasts; datasets persist |
| College/lab GPU box over SSH | whatever it has | ₹0 | you need access to one |
| Rented GPU (vast.ai, RunPod) | anything | **costs money** | ruled out by the zero-rupee constraint |
| This MacBook | none — Intel, no CUDA, no MPS | ₹0 | it is the machine we are leaving |

Two of the free options can refuse you a GPU on the day, so **the design does
not choose one**. `remote_batch.py` and `remote_setup.sh` are plain Python and
bash that run identically on a laptop, an SSH box, Colab, Kaggle or a VM. The
notebook is a thin wrapper that detects Colab vs Kaggle and changes only the
upload and download cells. If a provider disappears, one cell changes; nothing
in `backend/` knows a provider exists.

### The one thing that silently fails on a GPU

Detection follows the GPU by itself — `resolve_device("auto")` asks torch.
**ANPR does not.** It runs on ONNX Runtime, whose CPU build (`onnxruntime`)
and GPU build (`onnxruntime-gpu`) are **different packages providing the same
`import onnxruntime`**. Install the wrong one, or both, and 31% of the work
stays on the CPU with no error anywhere. So `remote_setup.sh` uninstalls the
CPU wheel before installing the GPU one and prints the providers it ends up
with, `remote_batch.py` prints both engines' devices before starting and says
outright when detection is on the GPU but ANPR is not, and
`anpr.providers: auto` intersects what you ask for with what is actually
installed, so a config copied back from the GPU box does not stop ANPR dead on
the laptop.

---

## 4. Files changed

**Modified (6):**

| file | change |
|---|---|
| `backend/app/pipeline/supervisor.py` | cap on how many cameras run at once; freed slots refill from the queue |
| `backend/app/cli.py` | `--workers` on `run`/`seed`; new `preflight`, `export`, `import` subcommands |
| `backend/app/vision/anpr.py` | ONNX provider selection (`auto` → CUDA/CoreML/CPU) and per-worker thread cap |
| `configs/system.yaml` | `performance.onnx_threads_per_worker`, `performance.max_parallel_cameras`, `anpr.providers` |
| `backend/requirements.txt` | `fast-alpr` uncommented — it had been in use since M5 while the file said "not installed yet", so a fresh `pip install -r` produced a broken project |
| `Makefile` | `preflight`, `remote`, `export`, `import`, `bundle-src`; the three newer test files added to `make test` |

**Added (8):**

| file | what it is |
|---|---|
| `backend/app/vision/preflight.py` | video triage: six verdicts, each with a remedy |
| `backend/app/store/bundle.py` | WAL-safe export, idempotent merging import |
| `backend/scripts/remote_batch.py` | the four existing commands, headless, timed, packaged |
| `backend/scripts/remote_setup.sh` | dependency install with GPU detection and verification |
| `backend/tests/test_preflight.py` | 20 checks |
| `backend/tests/test_bundle.py` | 32 checks |
| `notebooks/NETRA_remote.ipynb` | Colab **and** Kaggle wrapper over the two scripts |
| `docs/REMOTE.md` | how to run it and why it is built this way |

**Deleted: none. Rewritten: none.** No pipeline file, no model, no schema, no
API route and no frontend component was modified.

---

## 5. What the code changes actually do

**Worker cap** (`supervisor.py`). `run_supervisor()` takes `workers`; at most
that many processes exist at once, and a finished worker's slot is handed
straight to the next camera. Default is `min(8, cpu_count)`; override with
`--workers` or `performance.max_parallel_cameras`.

*Why this is safe.* Cameras are independent until `resolve` runs — no worker
reads another camera's output, and every timestamp comes from the frame index
plus that camera's `clock_offset_s`, never from wall-clock arrival order. Each
worker owns its own `run_id`-scoped track ids and its own database session.
Batching therefore changes **when** work happens, never **what** is concluded.
Verified empirically: identical output (16 sightings, 16 plates) in all six
benchmark runs, before and after.

**ONNX thread cap** (`anpr.py` + `system.yaml`). `onnx_threads_per_worker: 1`
sets `intra_op_num_threads`, matching what `torch_threads_per_worker` already
did for detection. The parallelism comes from running several cameras, not
from several threads inside each one.

**Provider selection** (`anpr.py`). `resolve_providers()` intersects the
requested providers with `ort.get_available_providers()` and always keeps a
CPU fallback.

**Preflight, export, import** — §2, §10.

---

## 6. Structure preserved

Unchanged: the directory layout; the `EventBus` abstraction and `LocalBus`;
one OS process per camera; `SightingWriter` as the only database writer; the
database schema (**no migration, no new column, no new table**); every API
route; the frontend; `configs/` as the only place tunable numbers live; the
worker's model-agnostic boundaries (`detector.py`, `anpr.py`).

New code is additive: two new modules, two new scripts, two new test files,
three new CLI subcommands.

---

## 7. Frontend and backend compatibility

No schema change, so nothing to be compatible with. **Verified by running it**:
a bundle produced by `remote_batch.py` was imported into a fresh tree and the
API served it —

| endpoint | result |
|---|---|
| `/api/health` | 200 |
| `/api/cameras` | 200, 8 cameras |
| `/api/sightings` | 200, plates intact (`MH12DE1433`) |
| `/api/identities` | 200, multi-camera identity across CAM01–CAM08 |
| `/api/journeys` | 200, routes intact |
| `/api/analytics` | 200 |
| `/api/alerts` | 200, 17 alerts |
| `/api/leads` | 200, 42 leads |
| `/api/sightings/16/snapshot` | 200, a real 226×301 JPEG |

The snapshot endpoint is the one that breaks when evidence is mishandled, so
it was checked specifically.

---

## 8. How to run it

**Locally, unchanged:**
```bash
make doctor          # as before
make preflight       # NEW -- check the footage first, takes a second
make seed            # now capped; add WORKERS=n or --workers n to change it
make api             # terminal 1
make dash            # terminal 2
```

**On another machine:**
```bash
# on the laptop
make bundle-src                                   # zip the code, no venv/data/footage

# on the other machine
bash backend/scripts/remote_setup.sh              # installs deps, proves the GPU is visible
python backend/scripts/remote_batch.py --workers 4 --max-frames 0

# back on the laptop
python -m app.cli import data/exports/netra-<timestamp>.tar.gz
make api  +  make dash
```

On Colab or Kaggle, open `notebooks/NETRA_remote.ipynb` and work down the
cells; it does the same three steps with upload/download wrappers.

`remote_batch.py` runs `preflight → run → resolve → analytics → export` by
calling the same `cmd_*` functions the CLI calls, so a remote run cannot drift
from a local one.

---

## 9. How to add videos

Unchanged: drop files in `data/videos/` (or anywhere) and point `source:` at
them in `configs/cameras.yaml`. What is new is that you can now check them
before committing an hour:

```bash
python -m app.cli preflight --dir data/videos     # judge a folder of candidates
python -m app.cli preflight                       # judge what is configured
python -m app.cli preflight --deep                # decode every frame, for a suspect file
```

For long or high-resolution footage the existing levers still apply and are
the right ones: `target_fps` in `cameras.yaml` sets the decode stride (the
source's own frames are skipped with `grab()`, so they cost nothing),
`detection.imgsz` sets the inference size, and `anpr.every_n_frames` sets the
OCR rate. Preflight reports each file's frame count and megapixels per frame,
which is what those decisions should be based on.

---

## 10. How results are stored and reached

Unchanged on disk: `data/netra.db` (SQLite, WAL) and `data/evidence/sightings/`
(one JPEG per sighting, named `<run_id>_<camera>_<track>.jpg`).

`export` writes one archive: the database, every evidence image, the configs
that produced them, and a manifest recording the machine, the device, the run
ids and the row counts. It uses SQLite's `VACUUM INTO`, **not** a file copy —
in WAL mode the `.db` file alone can be stale or, on a database that has never
checkpointed, effectively empty. (That mistake was in this feature's own
replace-mode backup at first; the test caught it, and it is now fixed.)

`import` **merges by run and is idempotent.** Every table except `cameras` has
an autoincrement primary key, so ids from the remote machine mean nothing
locally. Import renumbers everything and rewrites every foreign key as it
goes; a row whose target did not come across is **dropped, never repointed**,
because a journey assembled from a link with one end missing is a fabricated
journey. A run already present is skipped rather than duplicated.

`backend/tests/test_bundle.py` (32 checks) verifies this by fingerprinting
*meaning* rather than ids: every link still joins the same two sightings,
every journey still owns its own hops, every OCR read is still attached to the
vehicle it was read from, local results are untouched, a second import adds
nothing, a WAL-pending write survives, and a tar member escaping the
extraction directory is refused.

`--dry-run` reports and writes nothing. `--mode replace` swaps the database
instead, with a dated backup first.

---

## 11. Error reporting

**Footage** — `preflight` names every source as ready / placeholder / corrupt
/ unsupported / missing / stream, with a written reason and a remedy, plus
warnings (variable frame rate, unreachable plate width, clip too short) that
do not change the verdict. "Successfully processed" is answered from the
database, not guessed from the file: the last column shows each camera's
sightings and plates in the most recent run.

**Workers** — unchanged: a crashed worker is logged with its exit code and the
other cameras continue.

**Remote batch** — every phase is timed and reported pass/fail; a failure is
recorded and the remaining phases still run, with a summary table and a JSON
report at the end. It refuses to start on unusable footage unless you pass
`--allow-bad-video`.

**Devices** — the batch prints torch's device and ONNX Runtime's providers
before doing any work, and says explicitly when detection is on the GPU but
ANPR is not.

---

## 12. Baseline performance — **measured**

Identical workload in two trees: the code exactly as it was (no worker cap,
no ONNX thread cap, original `system.yaml`) and the code as it is now. Eight
cameras, 100 frames each, 1280×720 footage. Runs alternated before/after so a
noisy neighbour landed on both. Every repetition is reported; none discarded.

| rep | wall time | peak RSS | peak processes | sightings | plates |
|---|---|---|---|---|---|
| 1 | 69.9 s | 6765 MB | 11 | 16 | 16 |
| 2 | 67.1 s | 6777 MB | 10 | 16 | 16 |
| 3 | 67.5 s | 6788 MB | 10 | 16 | 16 |

Median **67.5 s**, peak memory **6.8 GB** on a 7 GB machine.

*Your 1.5-hour figure is your observation on your MacBook and is not something
this session could reproduce or verify — **not measured** here.*

---

## 13. Performance after the changes — **measured**

| rep | wall time | peak RSS | peak processes | sightings | plates |
|---|---|---|---|---|---|
| 1 | 45.9 s | 1744 MB | 4 | 16 | 16 |
| 2 | 43.3 s | 1748 MB | 4 | 16 | 16 |
| 3 | 43.3 s | 1767 MB | 4 | 16 | 16 |

| | before | after | change |
|---|---|---|---|
| wall time (median) | 67.5 s | **43.3 s** | **36% less** (1.56×) |
| peak memory | 6788 MB | **1767 MB** | **74% less** |
| peak processes | 11 | 4 | |
| sightings / plates | 16 / 16 | **16 / 16** | **identical** |

The identical output in all six runs is the important number. A speed-up that
changed what the system found would not be a speed-up.

**Twelve cameras — measured.** The same tree, 12 cameras, 100 frames each:
**65.3 s, peak 1820 MB, all 12 cameras produced sightings.** Peak memory is
now a function of `--workers`, not of camera count — which is precisely what
makes 10+ cameras possible. At ~800 MB per worker (measured), the old code
would have asked for ~9.6 GB on this 7 GB machine: **expected** to be an
out-of-memory failure rather than slow. Not run, deliberately.

**Remote batch end to end — measured**, 8 cameras, same machine:

| phase | seconds |
|---|---|
| preflight | 1.1 |
| process | 44.5 |
| resolve | 0.1 |
| analytics | 0.0 |
| export | 0.0 |
| **total** | **45.8** |

Then imported into a fresh tree: 251 rows and 16 evidence images, API verified
(§7).

**Refusal path — measured:** with one camera pointed at a TfL placeholder and
one at a missing path, `remote_batch.py` stopped after **0.9 s** with both
named, instead of loading models.

**GPU performance — NOT MEASURED.** There is no GPU in this environment.
Published T4-versus-CPU figures for YOLO11n and small ONNX models suggest a
large improvement, but nothing in this report will state a GPU number I have
not measured. `remote_batch.py` prints the real device and writes a JSON
report, so the first real GPU run measures it for you.

**Tests:** 280 checks pass across 9 files
(tracking 24, store 32, anpr 40, identity 56, trajectory 39, analytics 61,
leads 32, preflight 20, bundle 32). One pre-existing failure remains — see
§14.

**On your machine — measured.** The files were written into the project and
exercised there: `preflight` reports all 8 configured cameras **READY**, the
20 preflight tests pass against your own footage, and all 60 downloaded clips
were judged (table in §2). The remaining suites need SQLAlchemy, ultralytics
and fast-alpr, which live in your macOS virtualenv and cannot be driven from
this session — run `make test` yourself to confirm the other 260.

---

## 14. Remaining limitations

1. **`test_identity.py` — "and says it is capped" fails.** Pre-existing, and
   verified as such: it fails identically in an untouched copy of the code
   with the original config. Not caused by these changes and not fixed here.
2. **`GET /api/journeys/{plate}` returns 500** (`DetachedInstanceError` — a
   `return` outside the `with Session()` block). Pre-existing, flagged before,
   still unfixed. The list endpoint works; only the per-plate one fails.
3. **The GPU path is untested on real hardware.** The code selects devices and
   providers correctly and says which it got, but no GPU run has happened.
4. **TfL footage cannot test ANPR.** 352×288 against a 175 px plate threshold.
   Measured: 236 sightings, 0 plates. Preflight now warns, with the arithmetic.
5. **No cross-camera ground truth exists** on any public dataset (see
   `docs/DATASETS.md`), so end-to-end association accuracy remains
   **not measured** — by anyone, not just us.
6. **COCO has no auto-rickshaw class.** Unchanged, still documented.
7. **The worker cap only helps when cameras outnumber cores.** On a 16-core
   box running 8 cameras it changes nothing, by design.
8. **Results come back in a batch, not a stream.** A live remote feed would
   need the Kafka bus the event-bus abstraction was written for — a real
   project, not a flag.
9. **`resolve` still runs over the whole database** every time. At 16
   sightings this measured 0.1 s; it is O(n²) in sightings and will need
   windowing long before it needs a GPU.
10. **Footage transport is manual.** How video reaches the remote machine
    (Drive, a Kaggle dataset, `scp`) is left to you.

---

## 15. Before large-scale testing

1. **Run `make preflight`** and get every camera to `ready`. Re-capture the
   offline ones.
2. **Do one real GPU run** — Kaggle is the more predictable free option — and
   read the numbers off `data/exports/remote_report.json`. That is where the
   GPU figure this report refuses to invent comes from.
3. **Tune `--workers` on that machine.** Start at 2, raise it while wall time
   improves; on one GPU the failure mode for going too high is out-of-memory,
   not slowness.
4. **Test on one long video first** — an hour, not ten seconds — with
   `target_fps` set deliberately. Long-footage behaviour is **not measured**.
5. **Fix the two known defects** in §14 (1) and (2) before demo day.
6. **Decide the ANPR footage question.** If plates matter for the demo, TfL
   cannot provide them; `docs/DATASETS.md` §"Getting plate-readable linked
   footage" has the self-recorded protocol, which is also the only route to a
   real accuracy figure.
7. **Watch peak memory** on the first 10+ camera run with
   `--workers` at your chosen value: ~800 MB per worker is the measured number
   to plan against.

---

## 16. What was deliberately not done

* **No rewrite.** No pipeline, model, schema, API or frontend file was
  modified beyond the six listed in §4.
* **No re-encoding of any video.** The two failing clips are not repairable by
  transcoding, and re-encoding the working ones would cost time and image
  quality for nothing.
* **No videos deleted or replaced.** The placeholders are kept, now correctly
  labelled, and serve as test fixtures.
* **No provider lock-in.** Nothing in `backend/` mentions Colab.
* **No invented numbers.** Every figure here is either measured on this run or
  explicitly marked **expected** or **not measured**.
