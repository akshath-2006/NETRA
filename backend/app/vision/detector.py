"""Vehicle detection.

Wraps a pretrained YOLO model behind a small, stable interface. Two reasons
this is a class rather than three lines at each call site:

**1. Nothing else in the pipeline imports ultralytics.**
Tracking, ANPR, the workers and the analytics all deal in our own ``Detection``
objects. Swapping the detector later -- a newer YOLO, a model fine-tuned on
Indian traffic, an ONNX export running on an edge box -- changes exactly this
one file. That is the whole point of the boundary.

**2. Device selection and warm-up belong somewhere sensible.**
The first inference on any backend is several times slower than the rest
(weights load, kernels compile). Running one throwaway pass at load time keeps
the FPS number on the HUD honest instead of flattering.

KNOWN LIMITATION -- worth stating out loud rather than hiding:
COCO, which the pretrained weights are trained on, has no auto-rickshaw class.
India's most common three-wheeler is absorbed into car / truck / motorcycle or
missed entirely. The fix is fine-tuning on an Indian vehicle dataset. Until
then we report what the model actually predicts and say so in the presentation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.logging import get_logger

# COCO ids -> the label we display. Anything not listed is ignored.
COCO_VEHICLES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    track_id: int | None = None   # set by Detector.track(), None from detect()

    @property
    def box(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


def resolve_device(preference: str = "auto") -> str:
    """Turn 'auto' into the best backend actually available on this machine."""
    pref = (preference or "auto").lower()
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:  # pragma: no cover - torch arrives with ultralytics
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"          # Apple Silicon
    return "cpu"


class Detector:
    """Pretrained YOLO vehicle detector."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        model_dir: Path | None = None,
        confidence: float = 0.35,
        iou: float = 0.50,
        imgsz: int = 640,
        classes: Sequence[int] | None = None,
        device: str = "auto",
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else Path("data/models")
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes else [2, 3, 5, 7]
        self.device = resolve_device(device)
        self.tracker = tracker

        self.log = get_logger("detector")
        self._model = None
        self.last_inference_ms: float = 0.0

    # -- loading -----------------------------------------------------------

    def _weights_path(self) -> Path:
        """Keep downloaded weights in data/models instead of the repo root.

        Ultralytics downloads into the current working directory, so we briefly
        chdir into the model directory to catch the file where we want it.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        target = self.model_dir / self.model_name
        if target.exists():
            return target

        from ultralytics import YOLO

        self.log.info("downloading %s into %s (one time, ~5 MB)",
                      self.model_name, self.model_dir)
        cwd = os.getcwd()
        try:
            os.chdir(self.model_dir)
            YOLO(self.model_name)
        finally:
            os.chdir(cwd)
        return target if target.exists() else Path(self.model_name)

    def load(self) -> "Detector":
        if self._model is not None:
            return self

        from ultralytics import YOLO

        weights = self._weights_path()
        self._model = YOLO(str(weights))
        self.log.info("loaded %s on %s (imgsz=%d, conf=%.2f)",
                      weights.name, self.device, self.imgsz, self.confidence)

        # Warm-up: the first pass is always slower. Do it now so the HUD's FPS
        # reading reflects steady state rather than start-up cost.
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        self._raw(blank)
        self.log.info("warm-up pass took %.0f ms", (time.perf_counter() - t0) * 1000)
        return self

    # -- inference ---------------------------------------------------------

    def _raw(self, image: np.ndarray):
        return self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

    @staticmethod
    def _parse(results) -> list[Detection]:
        """Turn an ultralytics result into our own Detection objects.

        This is the only place the two representations meet. Everything
        downstream sees Detection and nothing else.
        """
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = results[0].names or {}
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        # boxes.id is None on frames where the tracker assigned nothing.
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

        for i, ((x1, y1, x2, y2), conf, cid) in enumerate(zip(xyxy, confs, clss)):
            label = COCO_VEHICLES.get(int(cid)) or names.get(int(cid), str(cid))
            detections.append(
                Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=float(conf), class_id=int(cid), label=label,
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on one BGR frame. No identity across frames."""
        if self._model is None:
            self.load()
        t0 = time.perf_counter()
        results = self._raw(image)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse(results)

    def track(self, image: np.ndarray) -> list[Detection]:
        """Detect and assign persistent track IDs.

        ``persist=True`` is what makes the tracker carry state between calls,
        so one Detector instance belongs to exactly one camera. That is also
        why the pipeline runs one process per camera.
        """
        if self._model is None:
            self.load()
        t0 = time.perf_counter()
        results = self._model.track(
            image,
            imgsz=self.imgsz, conf=self.confidence, iou=self.iou,
            classes=self.classes, device=self.device,
            persist=True, tracker=self.tracker, verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse(results)


def build_detector(cfg) -> Detector:
    """Construct a Detector from an AppConfig, so callers never poke at YAML."""
    sysd = cfg.raw.get("system", {}) or {}
    d = sysd.get("detection", {}) or {}
    t = sysd.get("tracking", {}) or {}
    return Detector(
        model_name=d.get("model", "yolo11n.pt"),
        model_dir=cfg.paths.model_dir,
        confidence=d.get("confidence", 0.35),
        iou=d.get("iou", 0.50),
        imgsz=d.get("imgsz", 640),
        classes=d.get("classes", [2, 3, 5, 7]),
        device=cfg.runtime.device,
        tracker=t.get("tracker", "bytetrack.yaml"),
    )
