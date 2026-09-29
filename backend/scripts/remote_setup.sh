#!/usr/bin/env bash
# Install NETRA's dependencies on whatever machine this is, and prove it worked.
#
#     bash backend/scripts/remote_setup.sh
#
# Deliberately provider-agnostic. The same script runs on a laptop, a
# university GPU box over SSH, a Colab or Kaggle notebook, or a plain VM. It
# does not assume a virtualenv, a package manager, or a GPU.
#
# THE ONE THING IT EXISTS FOR: getting ANPR onto the GPU when there is one.
# Vehicle detection follows the GPU by itself -- ultralytics asks torch and
# torch already knows. ANPR does not, because it runs on ONNX Runtime, and
# ONNX Runtime's CPU and GPU wheels are two DIFFERENT PACKAGES with the same
# import name. Install the wrong one and everything still runs, reports no
# error, and quietly does 31% of the work on the CPU. That failure is
# invisible from the outside, so this script checks rather than assumes.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
PIP="$PY -m pip"

echo "NETRA remote setup"
echo "  project  $ROOT"
echo "  python   $($PY -VV | head -1)"

# Some distributions mark the system Python as externally managed. Notebook
# environments are not virtualenvs and pip refuses without this flag.
BREAK=""
if $PY -c "import sysconfig,sys; sys.exit(0 if sysconfig.get_config_var('EXT_SUFFIX') else 0)" 2>/dev/null; then
  if $PIP install --dry-run pip 2>&1 | grep -q "externally-managed-environment"; then
    BREAK="--break-system-packages"
    echo "  note     system python is externally managed; using $BREAK"
  fi
fi

HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  HAS_GPU=1
  echo "  gpu      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  echo "  gpu      none detected (this will be a CPU run)"
fi

echo
echo "--- base dependencies ---------------------------------------------------"
# Notebook images already ship torch built against their own CUDA. Reinstalling
# it is a 2 GB download that usually makes things worse, so requirements.txt is
# installed WITHOUT upgrading anything that is already satisfied.
$PIP install $BREAK -q -r backend/requirements.txt

echo
echo "--- ANPR ----------------------------------------------------------------"
if [ "$HAS_GPU" = "1" ]; then
  # The two wheels conflict: both provide the `onnxruntime` import, and with
  # both installed you get the CPU one. Remove it first, deliberately.
  echo "installing the GPU build of ONNX Runtime"
  $PIP uninstall $BREAK -y -q onnxruntime 2>/dev/null || true
  $PIP install $BREAK -q "onnxruntime-gpu>=1.17"
else
  $PIP install $BREAK -q "onnxruntime>=1.17"
fi
$PIP install $BREAK -q "fast-alpr>=0.2"

echo
echo "--- verification --------------------------------------------------------"
# Checking, not assuming. Every line below is read off the installed libraries.
$PY - <<'PYCHECK'
import sys

ok = True

try:
    import torch
    dev = "cpu"
    if torch.cuda.is_available():
        dev = f"cuda ({torch.cuda.get_device_name(0)}, "\
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    print(f"  torch          {torch.__version__:<16} -> {dev}")
except Exception as exc:
    ok = False
    print(f"  torch          MISSING: {exc}")

try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    print(f"  onnxruntime    {ort.__version__:<16} -> {', '.join(provs)}")
    if "CUDAExecutionProvider" not in provs:
        print("                 (ANPR will run on the CPU)")
except Exception as exc:
    ok = False
    print(f"  onnxruntime    MISSING: {exc}")

try:
    import cv2
    print(f"  opencv         {cv2.__version__}")
except Exception as exc:
    ok = False
    print(f"  opencv         MISSING: {exc}")

for mod in ("ultralytics", "fast_alpr", "sqlalchemy", "yaml", "lap"):
    try:
        __import__(mod)
        print(f"  {mod:<14} ok")
    except Exception as exc:
        ok = False
        print(f"  {mod:<14} MISSING: {exc}")

# The plate-slot trap, checked at setup rather than discovered at 2 a.m.:
# a 9-slot OCR model silently truncates every 10-character Indian plate and
# still reports high confidence.
try:
    from fast_plate_ocr import LicensePlateRecognizer
    slots = LicensePlateRecognizer(hub_ocr_model="cct-s-v2-global-model")\
        .config.max_plate_slots
    print(f"  ocr slots      {slots} (need >= 10 for an Indian plate)")
    if slots < 10:
        ok = False
except Exception as exc:
    print(f"  ocr slots      not checked: {exc}")

sys.exit(0 if ok else 1)
PYCHECK

echo
echo "setup complete. next:"
echo "  $PY backend/scripts/remote_batch.py --workers 4 --max-frames 0"
