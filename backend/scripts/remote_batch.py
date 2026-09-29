#!/usr/bin/env python3
"""Run a whole NETRA batch headlessly, anywhere, and pack the results.

    python backend/scripts/remote_batch.py --workers 4 --max-frames 0

WHY THIS SCRIPT AND NOT A NEW SERVICE
NETRA already splits into an expensive half and a visible half. ``run``,
``resolve`` and ``analytics`` write ``data/netra.db`` and ``data/evidence/``;
``serve`` and the dashboard only read them. Those two halves already talk
through files, never through a socket or a shared process. So "do the heavy
work somewhere else" needs no new architecture, no service, no queue and no
API -- it needs the four existing commands run in order on the other machine,
and the results brought back.

That is all this script is: the four existing commands, in order, with timing
around each, plus a preflight that refuses to start and an export that ends
it. It calls the same ``cmd_*`` functions the local CLI calls. Nothing is
reimplemented, so a remote run cannot drift from a local one.

DELIBERATELY NOT TIED TO ANY PROVIDER. This runs unchanged on a laptop, a
university GPU box over SSH, a Colab notebook, a Kaggle notebook, or a spot
instance. The only thing that varies is who installs the dependencies, and
``remote_setup.sh`` does that part the same way everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import load_config                              # noqa: E402
from app.core.logging import setup_logging                           # noqa: E402


def describe_machine() -> dict:
    """What this box actually is. Measured, not assumed.

    Printed at the top of every batch because the single most common way to
    waste an afternoon on a "GPU" run is to discover afterwards that torch
    was the CPU wheel, or that ONNX Runtime never had a CUDA provider built
    in, and every number you collected was a CPU number.
    """
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": None, "torch_device": "cpu", "gpu": None, "gpu_memory_gb": None,
        "onnxruntime": None, "onnx_providers": [],
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["torch_device"] = "mps"
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["onnx_providers"] = list(ort.get_available_providers())
    except Exception as exc:
        info["onnx_error"] = str(exc)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1e6, 1)
                    break
    except OSError:
        pass
    return info


def print_machine(info: dict) -> None:
    print("machine")
    print(f"  {info['platform']}  ({info['cpu_count']} cores"
          + (f", {info['ram_gb']} GB RAM" if info.get("ram_gb") else "") + ")")
    print(f"  torch {info['torch'] or 'MISSING'} -> {info['torch_device']}"
          + (f"  [{info['gpu']}, {info['gpu_memory_gb']} GB]" if info["gpu"] else ""))
    print(f"  onnxruntime {info['onnxruntime'] or 'MISSING'} -> "
          f"{', '.join(info['onnx_providers']) or 'none'}")
    if info["torch_device"] == "cpu" and "CUDAExecutionProvider" not in info["onnx_providers"]:
        print("  NOTE: no GPU is visible to either engine. This run will be a "
              "CPU run whatever the machine is advertised as.")
    elif info["torch_device"] == "cuda" and "CUDAExecutionProvider" not in info["onnx_providers"]:
        print("  NOTE: detection will use the GPU but ANPR will not -- "
              "onnxruntime-gpu is not installed. See remote_setup.sh.")


class Phase:
    """Times a phase and keeps going, so one failure does not lose the rest."""

    def __init__(self, name: str, results: dict):
        self.name, self.results = name, results

    def __enter__(self):
        print(f"\n=== {self.name} " + "=" * (56 - len(self.name)))
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.t0
        self.results[self.name] = {"seconds": round(elapsed, 2),
                                   "ok": exc_type is None}
        if exc_type is not None:
            self.results[self.name]["error"] = f"{exc_type.__name__}: {exc}"
            print(f"--- {self.name} FAILED after {elapsed:.1f}s: {exc}")
            return True                       # swallow, report at the end
        print(f"--- {self.name} finished in {elapsed:.1f}s")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=None,
                    help="cameras processed at once (default: CPU count, max 8)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="frames per camera; 0 means the whole clip")
    ap.add_argument("--cameras", default=None, help="comma-separated subset")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default="data/exports",
                    help="where to write the results bundle")
    ap.add_argument("--no-export", action="store_true",
                    help="stop after analytics and leave the database in place")
    ap.add_argument("--allow-bad-video", action="store_true",
                    help="process even if preflight finds unusable sources")
    ap.add_argument("--report", default="data/exports/remote_report.json")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    os.environ.setdefault("NETRA_ROOT", str(ROOT))
    setup_logging(args.log_level)
    cfg = load_config()

    import app.cli as cli

    started = datetime.now()
    results: dict = {}
    machine = describe_machine()

    print(f"\nNETRA remote batch   {started:%Y-%m-%d %H:%M:%S}")
    print(f"project {ROOT}\n")
    print_machine(machine)

    # --- 1. preflight -------------------------------------------------------
    # Before any model loads. An hour of GPU time spent on ten "camera offline"
    # placeholders produces a clean run, zero sightings, and no explanation --
    # which is exactly how this failure went unnoticed the first time.
    with Phase("preflight", results):
        rc = cli.cmd_preflight(argparse.Namespace(
            cameras=args.cameras, dir=None, all=False, deep=False,
            head_frames=30, json=str(ROOT / "data" / "exports" / "preflight.json"),
            strict=not args.allow_bad_video, log_level=args.log_level), cfg)
        if rc != 0:
            print("\nrefusing to process unusable footage. Fix the sources above, "
                  "or pass --allow-bad-video to run anyway.")
            return 2

    # --- 2. process ---------------------------------------------------------
    with Phase("process", results):
        cli.cmd_run(argparse.Namespace(
            cameras=args.cameras, workers=args.workers, mode="fast",
            seconds=None, max_frames=args.max_frames or None,
            run_id=args.run_id, log_level=args.log_level), cfg)

    # --- 3. resolve ---------------------------------------------------------
    with Phase("resolve", results):
        cli.cmd_resolve(argparse.Namespace(limit=0, log_level=args.log_level), cfg)

    # --- 4. analytics -------------------------------------------------------
    with Phase("analytics", results):
        cli.cmd_analytics(argparse.Namespace(limit=0, verbose=False,
                                             log_level=args.log_level), cfg)

    # --- 5. export ----------------------------------------------------------
    bundle = None
    if not args.no_export:
        with Phase("export", results):
            from app.store.bundle import export_bundle
            manifest = export_bundle(
                database=cfg.paths.database, data_dir=cfg.paths.data_dir,
                configs_dir=ROOT / "configs", out=Path(args.out),
                note=f"remote batch on {machine['platform']} "
                     f"[{machine['torch_device']}]")
            bundle = manifest["bundle"]
            print(f"\nbundle  {bundle}  ({manifest['bundle_bytes']/1e6:.1f} MB)")

    # --- summary ------------------------------------------------------------
    total = sum(p["seconds"] for p in results.values())
    failed = [k for k, v in results.items() if not v["ok"]]

    print("\n" + "=" * 64)
    print(f"{'phase':<14}{'seconds':>10}   status")
    for name, p in results.items():
        print(f"{name:<14}{p['seconds']:>10.1f}   {'ok' if p['ok'] else p.get('error','failed')}")
    print(f"{'TOTAL':<14}{total:>10.1f}")

    counts = {}
    try:
        from app.store.db import get_sessionmaker
        from app.store.repository import stats as db_stats
        with get_sessionmaker(cfg.paths.database)() as s:
            counts = db_stats(s)
        print("\nproduced: " + "  ".join(f"{k} {v}" for k, v in counts.items()
                                         if isinstance(v, int)))
    except Exception as exc:
        print(f"\n(could not read final row counts: {exc})")

    report = {"started_at": started.isoformat(timespec="seconds"),
              "machine": machine, "phases": results,
              "total_seconds": round(total, 2), "database_counts": counts,
              "bundle": bundle,
              "arguments": {k: v for k, v in vars(args).items()}}
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport  {rp}")

    if bundle:
        print(f"\nNEXT, on the machine with the dashboard:\n"
              f"  python -m app.cli import {Path(bundle).name}\n"
              f"  make api   +   make dash")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
