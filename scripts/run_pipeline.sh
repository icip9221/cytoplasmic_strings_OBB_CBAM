#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <pipeline_config> [runtime_config] [runtime_key] [extra main.py args...]"
  echo "Example:"
  echo "  $0 configs/yolo_obb_train.yaml"
  echo "  $0 configs/yolo_obb_train.yaml default.yaml"
  exit 1
fi

PIPELINE_CONFIG="$1"
RUNTIME_CONFIG="default.yaml"
RUNTIME_KEY=""

shift 1

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  if [ -f "$1" ] || [[ "$1" == *.yaml ]] || [[ "$1" == *.yml ]] || [ "$1" = "default" ]; then
    if [ "$1" = "default" ]; then
      RUNTIME_CONFIG="default.yaml"
    else
      RUNTIME_CONFIG="$1"
    fi
    shift 1
  fi
fi

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  RUNTIME_KEY="$1"
  shift 1
fi

CMD=(python main.py --pipeline_config "$PIPELINE_CONFIG" --runtime_config "$RUNTIME_CONFIG")
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/ultralytics-cbam${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "$RUNTIME_KEY" ]; then
  CMD+=(--runtime_key "$RUNTIME_KEY")
fi
CMD+=("$@")

"${CMD[@]}"
