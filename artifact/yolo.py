from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(slots=True)
class YoloTrainArtifact:
    """Ultralytics YOLO detection training output."""

    model_name: str
    task: str
    dataset: str
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
class YoloExperimentArtifact:
    """Combined YOLO training, export, and evaluation artifact bundle."""

    train_artifact: YoloTrainArtifact | None = None
    onnx_export_artifact: YoloONNXExportArtifact | None = None
    pt_eval_metrics: dict[str, Any] | None = None
    onnx_eval_metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
