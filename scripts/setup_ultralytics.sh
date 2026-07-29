#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="3ca0b4fc373c01522da1a6ec25710516ae21beb2"
UPSTREAM_URL="https://github.com/ultralytics/ultralytics.git"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/ultralytics-cbam"
PATCH_FILE="$PROJECT_ROOT/patches/ultralytics_cs_obb.patch"

if [ ! -f "$PATCH_FILE" ]; then
  echo "Missing Ultralytics patch: $PATCH_FILE" >&2
  exit 1
fi

if [ ! -d "$RUNTIME_DIR" ]; then
  git clone "$UPSTREAM_URL" "$RUNTIME_DIR"
fi

cd "$RUNTIME_DIR"

if [ ! -d .git ]; then
  echo "$RUNTIME_DIR exists but is not a Git checkout." >&2
  exit 1
fi

git fetch --tags origin
git checkout "$PINNED_COMMIT"

if git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Ultralytics CS-OBB patch is already applied."
else
  if [ -n "$(git status --porcelain)" ]; then
    echo "Ultralytics checkout has local changes before patching." >&2
    git status --short >&2
    exit 1
  fi
  git apply --check "$PATCH_FILE"
  git apply "$PATCH_FILE"
fi

export PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from pathlib import Path
import inspect
import ultralytics

runtime = Path("ultralytics").resolve().parents[0]
imported = Path(ultralytics.__file__).resolve()
if runtime not in imported.parents:
    raise SystemExit(f"Expected local Ultralytics import under {runtime}, got {imported}")

from ultralytics.nn.modules import CBAM
from ultralytics.models.yolo.obb import OBBValidator

if CBAM.__module__ != "ultralytics.nn.modules.cbam":
    raise SystemExit(f"Unexpected CBAM module: {CBAM.__module__}")

validator = OBBValidator(args={"task": "obb"})
iouv = [round(float(x), 2) for x in validator.iouv.tolist()]
if iouv != [0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    raise SystemExit(f"Unexpected OBB IoU vector: {iouv}")

source = inspect.getsource(OBBValidator.get_stats)
if "legacy_ap=not self.training" not in source:
    raise SystemExit("Historical direct-evaluation AP integration is not active.")

print(f"Ultralytics import OK: {imported}")
print(f"CBAM import OK: {CBAM}")
print(f"OBB IoU vector OK: {iouv}")
PY

python -m pip install -e "$RUNTIME_DIR"

if [ -d "$PROJECT_ROOT/.git" ]; then
  {
    grep -qxF "/default.yaml" "$PROJECT_ROOT/.git/info/exclude" || echo "/default.yaml"
    grep -qxF "/ultralytics-cbam/" "$PROJECT_ROOT/.git/info/exclude" || echo "/ultralytics-cbam/"
  } >> "$PROJECT_ROOT/.git/info/exclude"
fi

echo "Ultralytics CS-OBB runtime is ready at $RUNTIME_DIR"
