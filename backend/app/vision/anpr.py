"""Automatic Number Plate Recognition.

Plate detection and OCR sit behind one small interface for the same reason the
vehicle detector does: nothing else in the pipeline should know which engine we
picked, so swapping one is a config change rather than a refactor.

The default engine is **fast-alpr** (MIT licence, ONNX Runtime, CPU-friendly,
installs cleanly on Apple Silicon). It bundles a plate detector and a global
plate-OCR model, both free, both about 7 MB, neither needing an account.

Two things worth knowing:

* We run ANPR on the **vehicle crop**, not the whole frame. It is faster and
  much more accurate, because the plate detector is looking at a region that
  definitely contains one vehicle instead of a whole junction.
* fast-alpr returns **per-character** OCR confidences. We keep them, because
  the consensus vote in ``consensus.py`` can then weight each character
  position by how sure the model was about *that* character rather than the
  string as a whole.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.core.logging import get_logger
from app.vision.plate_grammar import normalise


@dataclass
class PlateHit:
    """One plate found and read in one image."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]
    detection_confidence: float = 0.0
    char_confidence: list[float] = field(default_factory=list)

    @property
    def width_px(self) -> int:
        return max(0, self.box[2] - self.box[0])


class ANPREngine(ABC):
    name = "abstract"

    @abstractmethod
    def load(self) -> "ANPREngine":
        ...

    @abstractmethod
    def read(self, image: np.ndarray) -> list[PlateHit]:
        ...


class FastALPREngine(ANPREngine):
    """fast-alpr: bundled plate detector + global plate OCR, all ONNX."""

    name = "fast-alpr"

    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "cct-s-v2-global-model",
        min_ocr_slots: int = 10,
        providers: str | list[str] = "auto",
        threads: int = 0,
    ) -> None:
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self.min_ocr_slots = int(min_ocr_slots)
        self.providers_pref = providers
        self.threads = int(threads or 0)
        self.providers: list[str] = []
        self.max_plate_slots: int | None = None
        self._alpr = None
        self.last_ms = 0.0
        self.log = get_logger("anpr")

    def load(self) -> "FastALPREngine":
        if self._alpr is not None:
            return self
        import onnxruntime as ort
        from fast_alpr import ALPR

        self.providers = resolve_providers(self.providers_pref)

        # THREAD CAPPING, for exactly the reason torch needs it in the worker.
        # ONNX Runtime defaults to one intra-op thread per core, PER SESSION,
        # and each camera process holds two sessions (plate detector + OCR).
        # Eight cameras on eight cores therefore asked for 128 threads to
        # share 8 cores. They do not go faster; the difference is spent on
        # context switching. ANPR is 31% of measured runtime, so this is not
        # a rounding error -- it is the second-biggest cost in the pipeline.
        opts = None
        if self.threads > 0:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.threads
            opts.inter_op_num_threads = 1

        self.log.info("loading %s + %s on %s (first run downloads ~15 MB)",
                      self.detector_model, self.ocr_model, ", ".join(self.providers))
        self._alpr = ALPR(detector_model=self.detector_model,
                          ocr_model=self.ocr_model,
                          detector_providers=self.providers,
                          ocr_providers=self.providers,
                          detector_sess_options=opts,
                          ocr_sess_options=opts)

        # THE CHECK THAT SHOULD HAVE EXISTED FROM DAY ONE.
        #
        # Every fast-plate-ocr model emits a FIXED number of character slots.
        # Pick one with fewer slots than the plate format needs and it does not
        # error -- it quietly truncates, drops the leading characters, and
        # still reports ~1.00 confidence on the mutilated string. That failure
        # is invisible from the outside: the pipeline runs, the dashboard
        # fills, and every plate is subtly wrong.
        #
        # An Indian plate (KA01AB1234) is 10 characters. The v1 global models
        # have 9 slots and scored 0/22 against ground truth. We refuse to start
        # quietly on a model that cannot represent the answer.
        try:
            self.max_plate_slots = int(self._alpr.ocr.ocr_model.config.max_plate_slots)
        except Exception:
            self.log.warning("could not read max_plate_slots from %s -- "
                             "cannot verify it fits the plate format", self.ocr_model)
        else:
            self.log.info("ocr model %s emits %d character slots",
                          self.ocr_model, self.max_plate_slots)
            if self.min_ocr_slots and self.max_plate_slots < self.min_ocr_slots:
                raise ValueError(
                    f"OCR model {self.ocr_model!r} emits only "
                    f"{self.max_plate_slots} characters, but anpr.min_ocr_slots "
                    f"is {self.min_ocr_slots}. It cannot read a full "
                    f"{self.min_ocr_slots}-character plate and will silently "
                    f"truncate every read.\n"
                    f"  fix: set anpr.ocr_model to a model with enough slots "
                    f"(cct-s-v2-global-model has 10), or lower "
                    f"anpr.min_ocr_slots if your plate format really is shorter."
                )

        # Warm-up, for the same reason as the vehicle detector.
        self._alpr.predict(np.zeros((160, 320, 3), dtype=np.uint8))
        self.log.info("anpr ready")
        return self

    def read(self, image: np.ndarray) -> list[PlateHit]:
        if self._alpr is None:
            self.load()
        if image is None or image.size == 0:
            return []

        t0 = time.perf_counter()
        try:
            results = self._alpr.predict(image)
        except Exception:
            # A malformed crop must never take the pipeline down mid-demo.
            self.log.exception("anpr failed on a crop")
            return []
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        hits: list[PlateHit] = []
        for r in results:
            ocr = getattr(r, "ocr", None)
            if ocr is None or not ocr.text:
                continue
            char_conf = list(ocr.confidence) if isinstance(ocr.confidence, (list, tuple)) \
                else [float(ocr.confidence or 0.0)]
            bb = r.detection.bounding_box
            hits.append(PlateHit(
                text=normalise(ocr.text),
                confidence=float(sum(char_conf) / len(char_conf)) if char_conf else 0.0,
                box=(int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)),
                detection_confidence=float(getattr(r.detection, "confidence", 0.0) or 0.0),
                char_confidence=[float(c) for c in char_conf],
            ))
        return hits


class NullEngine(ANPREngine):
    """Used when ANPR is switched off in config. Keeps the pipeline runnable."""

    name = "disabled"

    def load(self) -> "NullEngine":
        return self

    def read(self, image: np.ndarray) -> list[PlateHit]:
        return []


def resolve_providers(preference: str | list[str] = "auto") -> list[str]:
    """Which ONNX Runtime backends to use, checked against what is installed.

    'auto' means "GPU if this machine actually has one". The check matters:
    naming a provider that is not built into the installed onnxruntime makes
    newer versions raise rather than quietly fall back, so a config copied
    back from the GPU box would stop ANPR dead on the laptop.

    Moving ANPR onto a GPU is therefore `pip install onnxruntime-gpu` and
    nothing else: the CPU and GPU wheels expose the same API and differ only
    in which providers they report.
    """
    try:
        import onnxruntime as ort
        available = list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    if isinstance(preference, str) and preference.lower() in ("auto", ""):
        wanted = ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                  "CPUExecutionProvider"]
    elif isinstance(preference, str):
        wanted = [preference]
    else:
        wanted = list(preference)

    chosen = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")      # always keep a fallback
    return chosen


def build_anpr(cfg) -> ANPREngine:
    sysd = cfg.raw.get("system", {}) or {}
    a = sysd.get("anpr", {}) or {}
    if not a.get("enabled", True):
        return NullEngine()

    perf = sysd.get("performance", {}) or {}
    engine = str(a.get("engine", "fast-alpr")).lower()
    if engine in ("fast-alpr", "fastalpr", "fast_alpr"):
        return FastALPREngine(
            detector_model=a.get("detector_model", "yolo-v9-t-384-license-plate-end2end"),
            ocr_model=a.get("ocr_model", "cct-s-v2-global-model"),
            min_ocr_slots=int(a.get("min_ocr_slots", 10)),
            providers=a.get("providers", "auto"),
            threads=int(perf.get("onnx_threads_per_worker", 1) or 0),
        )
    raise ValueError(
        f"unknown anpr engine {engine!r}. supported: fast-alpr. "
        f"set anpr.enabled: false in system.yaml to run without ANPR."
    )
