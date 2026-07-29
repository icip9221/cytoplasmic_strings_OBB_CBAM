#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="3ca0b4fc373c01522da1a6ec25710516ae21beb2"
EXPECTED_PARAMS="21165368"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/ultralytics-cbam"
MODEL_YAML="$PROJECT_ROOT/configs/yolo11m-obb-1cbam.yaml"

if [ ! -d "$RUNTIME_DIR/.git" ]; then
  echo "Missing reconstructed Ultralytics checkout: $RUNTIME_DIR" >&2
  echo "Run: bash scripts/setup_ultralytics.sh" >&2
  exit 1
fi

actual_commit="$(git -C "$RUNTIME_DIR" rev-parse HEAD)"
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "Unexpected Ultralytics commit: $actual_commit" >&2
  exit 1
fi

if ! git -C "$RUNTIME_DIR" apply --reverse --check "$PROJECT_ROOT/patches/ultralytics_cs_obb.patch" >/dev/null 2>&1; then
  echo "The CS-OBB Ultralytics patch is not applied cleanly." >&2
  exit 1
fi

export PROJECT_ROOT RUNTIME_DIR MODEL_YAML EXPECTED_PARAMS
export PYTHONPATH="$RUNTIME_DIR:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["PROJECT_ROOT"]).resolve()
runtime = Path(os.environ["RUNTIME_DIR"]).resolve()
model_yaml = Path(os.environ["MODEL_YAML"]).resolve()
expected_params = int(os.environ["EXPECTED_PARAMS"])

import ultralytics

imported = Path(ultralytics.__file__).resolve()
if not imported.is_relative_to(runtime):
    raise SystemExit(f"Expected local ultralytics import from {runtime}, got {imported}")

from ultralytics import YOLO
from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.nn.modules import CBAM
from ultralytics.nn import tasks

if CBAM.__module__ != "ultralytics.nn.modules.cbam":
    raise SystemExit(f"Unexpected CBAM implementation: {CBAM.__module__}")

parse_source = inspect.getsource(tasks.parse_model)
if "elif m is CBAM" not in parse_source or "args = [c1]" not in parse_source:
    raise SystemExit("CBAM parser registration is missing or not channel-preserving.")

model = YOLO(str(model_yaml), task="obb")
params = sum(parameter.numel() for parameter in model.model.parameters())
if params != expected_params:
    raise SystemExit(f"Unexpected parameter count: {params} != {expected_params}")

layers = list(model.model.model)
cbam_layers = [(index, layer) for index, layer in enumerate(layers) if layer.__class__.__name__ == "CBAM"]
if len(cbam_layers) != 1 or cbam_layers[0][0] != 11:
    raise SystemExit(f"Expected exactly one CBAM layer at index 11, got {[i for i, _ in cbam_layers]}")

cbam = cbam_layers[0][1]
channels_in = cbam.channel_attention.fc.in_channels
channels_out = cbam.channel_attention.fc.out_channels
if channels_in != channels_out or channels_in != 512:
    raise SystemExit(f"Unexpected CBAM channels: in={channels_in}, out={channels_out}")

head = layers[-1]
if head.__class__.__name__ != "OBB" or list(head.f) != [17, 20, 23]:
    raise SystemExit(f"Unexpected OBB head: {head.__class__.__name__}, from={getattr(head, 'f', None)}")

validator = OBBValidator(args={"task": "obb"})
iouv = [round(float(value), 2) for value in validator.iouv.tolist()]
expected_iouv = [0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
if iouv != expected_iouv:
    raise SystemExit(f"Unexpected OBB IoU vector: {iouv}")

get_stats_source = inspect.getsource(OBBValidator.get_stats)
if "legacy_ap=not self.training" not in get_stats_source:
    raise SystemExit("Historical AP integration is not enabled for direct OBB evaluation.")

from algorithms.Blastocyst._yolo_obb_ops import extract_obb_eval_metrics, load_yolo

load_source = inspect.getsource(load_yolo) + inspect.getsource(__import__("algorithms.Blastocyst._yolo_obb_ops", fromlist=["_load_pretrained"])._load_pretrained)
if "weights_only=False" not in load_source or "_shift_layer_key" not in load_source:
    raise SystemExit("Project pretrained weight transfer guard is missing.")

class Box:
    all_ap = np.array([[0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]])
    p = np.array([0.81])
    r = np.array([0.82])

class DummyMetrics:
    box = Box()
    nt_per_image = np.array([692])
    nt_per_class = np.array([889])

summary = extract_obb_eval_metrics(DummyMetrics())
if summary["AP25"] != 25.0 or summary["AP50"] != 50.0:
    raise SystemExit(f"AP column extraction is wrong: {summary}")

missing = [name for name in ("onnx", "onnxruntime") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing ONNX dependency modules: {missing}")

print(f"Ultralytics import: {imported}")
print(f"CBAM layer: index=11, channels={channels_in}->{channels_out}")
print(f"OBB head inputs: {list(head.f)}")
print(f"Parameter count: {params}")
print(f"Validator IoU vector: {iouv}")
print(f"AP semantics: AP25={summary['AP25']}, AP50={summary['AP50']}, mAP50-95={summary['mAP50-95']}")
print("ONNX dependencies: onnx, onnxruntime")
PY
