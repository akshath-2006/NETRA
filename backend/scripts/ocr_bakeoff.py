"""OCR bake-off: choose the plate model with a measurement, not a guess.

Point it at a folder of cropped plate images whose FILENAME is the ground
truth, e.g. ``KA01AB1234.jpg``. It runs every candidate OCR model over them and
reports character accuracy, exact-match rate and milliseconds per plate.

    python backend/scripts/ocr_bakeoff.py data/plates
    python backend/scripts/ocr_bakeoff.py data/plates --models cct-s-v1-global-model,cct-xs-v1-global-model

Why it matters beyond the code: "we chose cct-s because it scored 94.2% on our
own 47-plate set, against 91.8% for the model that runs twice as fast" is a
presentation slide. "We used the popular one" is not.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.vision.plate_grammar import inspect as inspect_plate  # noqa: E402
from app.vision.plate_grammar import normalise, repair          # noqa: E402

DEFAULT_MODELS = ["cct-s-v1-global-model", "cct-xs-v1-global-model"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def char_accuracy(truth: str, guess: str) -> float:
    """Fraction of characters correct, penalising length mismatch."""
    if not truth:
        return 0.0
    correct = sum(1 for a, b in zip(truth, guess) if a == b)
    return correct / max(len(truth), len(guess))


def load_set(folder: Path) -> list[tuple[str, Path]]:
    items = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        truth = normalise(path.stem.split("_")[0])
        if truth:
            items.append((truth, path))
    return items


def run_model(model: str, items, detector: str) -> dict:
    from fast_alpr import ALPR

    alpr = ALPR(detector_model=detector, ocr_model=model)
    alpr.predict(cv2.imread(str(items[0][1])))          # warm-up

    exact = exact_repaired = valid = found = 0
    acc_total = 0.0
    ms_total = 0.0
    misses: list[tuple[str, str]] = []

    for truth, path in items:
        image = cv2.imread(str(path))
        if image is None:
            continue
        t0 = time.perf_counter()
        results = alpr.predict(image)
        ms_total += (time.perf_counter() - t0) * 1000.0

        guess = ""
        if results and results[0].ocr and results[0].ocr.text:
            guess = normalise(results[0].ocr.text)
            found += 1

        repaired, _ = repair(guess)
        acc_total += char_accuracy(truth, guess)
        exact += guess == truth
        exact_repaired += repaired == truth
        valid += inspect_plate(repaired).valid
        if repaired != truth:
            misses.append((truth, f"{guess} -> {repaired}"))

    n = len(items)
    return {
        "model": model, "n": n, "found": found,
        "char_accuracy": acc_total / n,
        "exact": exact / n,
        "exact_repaired": exact_repaired / n,
        "format_valid": valid / n,
        "ms": ms_total / n,
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare plate OCR models on your own crops")
    ap.add_argument("folder", help="folder of plate crops named <PLATE>.jpg")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--detector", default="yolo-v9-t-384-license-plate-end2end")
    ap.add_argument("--show-misses", type=int, default=8)
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2

    items = load_set(folder)
    if not items:
        print(f"no usable images in {folder}.\n"
              f"name each crop after its plate, e.g. KA01AB1234.jpg", file=sys.stderr)
        return 2

    print(f"{len(items)} plate crops from {folder}\n")
    rows = [run_model(m, items, args.detector) for m in args.models.split(",") if m.strip()]

    print(f"{'MODEL':<26} {'CHAR ACC':>9} {'EXACT':>7} {'+REPAIR':>8} "
          f"{'VALID':>7} {'DETECT':>7} {'MS':>7}")
    print("-" * 76)
    for r in rows:
        print(f"{r['model']:<26} {r['char_accuracy']:>8.1%} {r['exact']:>7.1%} "
              f"{r['exact_repaired']:>8.1%} {r['format_valid']:>7.1%} "
              f"{r['found']/r['n']:>7.1%} {r['ms']:>7.1f}")

    best = max(rows, key=lambda r: r["exact_repaired"])
    print(f"\nbest exact-match after grammar repair: {best['model']} "
          f"({best['exact_repaired']:.1%})")
    print("set it in configs/system.yaml -> anpr.ocr_model")

    gain = best["exact_repaired"] - best["exact"]
    if gain > 0:
        print(f"grammar repair alone added {gain:+.1%} on this set.")

    if args.show_misses and best["misses"]:
        print(f"\nworst cases for {best['model']}:")
        for truth, got in best["misses"][:args.show_misses]:
            print(f"  truth {truth:<12} got {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
