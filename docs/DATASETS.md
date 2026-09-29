# Test data: what is legally and practically obtainable

Verified 20–22 September 2026. Every licence below was read from an
authoritative page; anything I could not verify is marked **UNVERIFIED** rather
than guessed. This file exists so that "where did your data come from?" has a
written answer.

---

## The finding that shapes everything

**No public dataset has both readable number plates and cross-camera vehicle
identity ground truth.** Every serious multi-camera vehicle dataset redacts
plates for privacy:

| Dataset | Their own words |
|---|---|
| CityFlowV2 | *"The privacy issue has been addressed by redacting vehicle license plates and human faces."* |
| VeRi-776 | *"Due to the privacy issue we will not provide the license plates in the future."* |
| VERI-Wild | *"License plates are masked in the dataset."* |

So NETRA's headline capability — plate-driven cross-camera association —
**cannot be validated end-to-end on any public data**. Not by us, not by
anyone. The two halves must be tested separately:

* plate reading → ANPR datasets (single camera, no identities)
* cross-camera association → re-ID datasets (identities, no plates)

This is worth saying out loud rather than hiding. A judge who asks why we
have no end-to-end accuracy figure gets an answer that shows we read the terms.

---

## Tier 1 — open, verified, usable today

### UVH-26 — Indian vehicle detection
* **Source** https://huggingface.co/datasets/iisc-aim/UVH-26
* **Licence** CC BY 4.0 (verified on the dataset card and the IISc release note).
  Commercial use permitted. Attribution required.
* **Access** open download, no agreement, no gating.
* **Content** 26,646 anonymised 1080p frames from ~2,800 Bengaluru "Safe City"
  CCTV cameras, Feb 2025, 06:00–18:00 IST. ~1.8M boxes, COCO JSON.
* **Classes** 14 India-specific: Cycle, 2-Wheeler, **3-Wheeler (Auto-rickshaw)**,
  LCV, Van, **Tempo-traveller**, Hatchback, Sedan, SUV, MUV, Mini-bus, Bus,
  Truck, Other.
* **Cameras / sync / cross-camera GT** images only, not video. No camera ids in
  the release, no synchronisation, no cross-camera identities.
* **Useful for** detection and classification on real Indian CCTV. **Not** for
  tracking, ANPR (anonymised) or cross-camera work.
* **Why it matters** this is the direct fix for our documented COCO limitation:
  COCO has no auto-rickshaw class, so our detector currently absorbs India's
  most common three-wheeler into car/truck/motorcycle or misses it. UVH-26 is
  the dataset that would close that gap, with zero legal friction.

### TfL JamCams — real multi-camera corridors
* **Source** https://api.tfl.gov.uk/ (free key at https://api-portal.tfl.gov.uk/)
* **Licence** Open Government Licence v2.0 with TfL amendments. Verbatim: you may
  *"Exploit the Information commercially and non-commercially."*
* **Attribution REQUIRED**, wherever shown:
  * `Powered by TfL Open Data`
  * `Contains OS data © Crown copyright and database rights 2016`
  * `Geomni UK Map data © and database rights 2019`
* **Content** 900+ geolocated cameras; stills refresh every 5–10 min, short MP4
  clips per camera.
* **Cross-camera GT** none. Unlabelled — physically the same vehicles do pass
  adjacent cameras, but nothing says which.
* **Resolution** TfL's own documentation: *"low-resolution overviews of traffic
  conditions — not individual vehicle detail"* and they *"do not record number
  plates"*. **Expect zero readable plates.**
* **Useful for** detection, tracking, counting, congestion, camera health, and a
  realistic multi-camera topology on a real road network.
* **Restriction** use the Unified API. The licence forbids automated scraping of
  TfL's other web properties.
* **Tooling** `backend/scripts/fetch_tfl.py` — finds genuine corridors, staggers
  capture by travel time, writes config with routed distances, records provenance.

### CCPD — ANPR volume
* **Source** https://github.com/detectRecog/CCPD
* **Licence** MIT (stated in README and LICENSE).
* **Content** 300,000+ images, Chinese plates, one per image. Subsets for blur,
  rotation, tilt, night, challenge. Ground truth encoded in filenames: plate
  box, **four corner vertices**, characters, tilt, brightness, blurriness.
* **Useful for** ANPR only. No camera ids, no video, no cross-camera identity.
* **Caution** MIT is a *software* licence applied to real unredacted plates with
  no documented consent basis. Acting in good faith is fine, but **do not show
  unblurred real plates from it in a presentation** — transcribe or blur.

---

## Tier 2 — obtainable with a college email

Use your institutional address, not Gmail. All are **non-commercial, research
only, no redistribution**. An SIH prototype qualifies; a commercialised product
would not, and the data would have to be swapped out.

| Dataset | Access | Gets you |
|---|---|---|
| **RodoSol-ALPR** | email `rblsantos@inf.ufpr.br` with the signed paragraph | 20,000 images; **5,000 motorcycle plates** — the closest public proxy for India's two-wheeler ANPR problem, which is our hardest case |
| **VeRi-776** | email `xinchenliu@bupt.cn` | 20 cameras, 776 identities, each vehicle on **2–18 cameras**, with timestamps and inter-camera distances. Plates no longer distributed. |
| **VERI-Wild** | email `yanbai@pku.edu.cn` | **174 cameras**, 40,671 identities, camera ids + timestamps + track relations. Plates masked. |
| **UFPR-ALPR** | email `rblsantos@inf.ufpr.br` | 4,500 annotated images, per-character labels. Onboard camera, not fixed infrastructure. |

> **Do not** take VeRi-776 from Kaggle mirrors. The authors state they *"will not
> give it to any third party or publish it publicly anywhere"* — using a mirror
> contradicts their terms. Request it properly.

---

## Tier 3 — avoid, with reasons

| Dataset | Why not |
|---|---|
| **AI City Challenge / CityFlowV2** | The only real cross-camera GT that exists (46 cameras, 880 vehicles each on ≥2 cameras, synchronised by start-time offsets). Download is open with no form — but the licence PDF at `aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf` redirect-loops and **could not be read**. Licence **UNVERIFIED**. Read it in a browser before downloading; do not assume it is permissive. Evaluation-server registration additionally refuses Gmail. |
| **UA-DETRAC** | Official distribution is **defunct** — the download URL now redirects to a lab page with no data and no licence. Kaggle/Roboflow mirrors have unverifiable provenance and user-declared licences the authors never granted. Also single-camera only, so useless for cross-camera work. |
| **VRIC** | Derived from UA-DETRAC; *"copyright belongs to the original owners"* of a dataset whose licence cannot be established. |
| **BDD100K** | Licence internally inconsistent (GitHub BSD covers the *toolkit*; Zenodo says CC BY 3.0 US; the official licence page is unreachable). Dashcam footage anyway — no fixed cameras, no cross-camera ids, no plates. |
| **Indian_LPR** | The paper advertises an *"open-source dataset with 16192 images"*; the repo says *"We can't make dataset public because of legalities involved in making Indian Road data public."* The data was never released. The pretrained Indian-plate weights are still useful. |
| **Glasgow CCTV** | 2.65 TB of exactly the right footage. Record states: *"CCTV traffic video data is not available for research sharing."* Metadata only. |
| **Roboflow Universe plate sets** | Licences are self-declared by uploaders who often do not own the images. Tiny. Smoke tests only. |

---

## Getting plate-readable linked footage

There is no public source. The only route that produces footage with **both**
readable plates **and** known cross-camera identities is to record it yourself.
That is not a workaround — it is what every dataset author did.

**Protocol** (three phones on tripods is enough):

1. Pick three points along one road, 300–800 m apart, with no major junction
   between them — so a vehicle passing point 1 must pass 2 and 3.
2. Frame so a plate is **≥ 80 px wide** in frame. Measured on our data, plate
   reads below that width were wrong 100% of the time. Zoom in; a wide
   establishing shot is useless for ANPR.
3. Start all three recording, then film a clap or a phone showing the same clock
   at each camera. That is your synchronisation reference.
4. Record 10–15 minutes simultaneously.
5. **Write down the ground truth as it happens**: plate, camera, wall-clock
   time. A notebook is fine. This is the part nobody else can give you, and it
   is what turns "it looked right" into a measured number.
6. Fill `data/groundtruth/journeys.csv`, then `make eval` and `make report`
   produce real precision/recall instead of "not yet measured".

Fifteen minutes of self-shot footage with a written pass log is worth more to a
judge than any amount of borrowed data, because it is the only thing that lets
you state an accuracy figure you actually measured.

---

## Honest status

| Claim | Status |
|---|---|
| Detection on real Indian CCTV | **NOT YET MEASURED** — UVH-26 obtainable, not yet run |
| Detection/tracking on real fixed cameras | **NOT YET MEASURED** — TfL tooling written, not yet run |
| ANPR accuracy on real plates | **NOT YET MEASURED** — needs CCPD or RodoSol |
| Cross-camera accuracy | **NOT MEASURABLE on public data** — no dataset has plates + identities |
| Current accuracy figures | From our own generated clips with known ground truth. Real, measured, and **not representative of field conditions**. |
