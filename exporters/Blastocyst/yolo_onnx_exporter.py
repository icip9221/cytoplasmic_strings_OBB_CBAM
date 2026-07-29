from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def export_blastocyst_yolo_to_onnx(
    source_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    imgsz: int = 800,
    opset: int = 12,
    simplify: bool = True,
    device: int | str | None = 0,
    task: str = "obb",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a YOLO checkpoint to ONNX with Ultralytics."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required to export YOLO-OBB checkpoints to ONNX.") from exc

    source_checkpoint = Path(source_checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {source_checkpoint}")

    model = YOLO(str(source_checkpoint), task=task)
    exported = model.export(format="onnx", imgsz=int(imgsz), opset=int(opset), simplify=bool(simplify), device=device)
    exported_path = Path(exported) if exported else source_checkpoint.with_suffix(".onnx")
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce an ONNX file: {exported_path}")

    destination = output_dir / f"{source_checkpoint.stem}.onnx"
    if exported_path.resolve() != destination.resolve():
        shutil.copy2(exported_path, destination)
    return destination
