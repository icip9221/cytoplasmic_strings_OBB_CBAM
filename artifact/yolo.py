from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class YoloObbArtifactManifest:
    """Minimal manifest for a YOLO OBB artifact."""

    weights_path: Path | None = None
    export_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class YoloOBBTrainArtifact:
    """Ultralytics YOLO-OBB training output."""

    model_name: str
    dataset_variant: str
    data_yaml_path: Path
    run_dir: Path
    best_pt_path: Path
    last_pt_path: Path
    train_metrics: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class YoloONNXExportArtifact:
    """YOLO ONNX export output."""

    source_pt_path: Path
    onnx_path: Path
    export_dir: Path
    opset: int
    simplify: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class YoloOBBExperimentArtifact:
    """Combined YOLO-OBB training, export, and evaluation artifact bundle."""

    train_artifact: YoloOBBTrainArtifact | None = None
    onnx_export_artifact: YoloONNXExportArtifact | None = None
    pt_eval_metrics: dict[str, Any] | None = None
    onnx_eval_metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
