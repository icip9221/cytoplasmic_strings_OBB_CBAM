"""Ultralytics YOLO-OBB helpers for Blastocyst experiments."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from artifact.yolo import YoloOBBExperimentArtifact, YoloOBBTrainArtifact, YoloONNXExportArtifact
from exporters.Blastocyst.yolo_onnx_exporter import export_blastocyst_yolo_to_onnx


def validate_yolo_runtime_config(cfg: dict[str, Any]) -> None:
    data_yaml = Path(cfg["dataset"]["data_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found for selected dataset_variant: {data_yaml}")
    if cfg["train"].get("enabled", False):
        for key in ("architecture", "pretrained"):
            value = cfg["model"].get(key)
            if not value:
                raise ValueError(f"model.{key}_ref must resolve when train.enabled=true.")
            is_downloadable_weight = key == "pretrained" and Path(value).parent == Path(".")
            if not Path(value).exists() and not is_downloadable_weight:
                raise FileNotFoundError(f"YOLO model {key} not found: {value}")


def run_yolo_obb_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(cfg["dataset"]["project_root"]) / cfg["experiment"]["name"]
    train_artifact = None
    export_artifact = None
    pt_eval_metrics = None
    onnx_eval_metrics = None
    best_pt = Path(cfg.get("best_pt_path") or run_dir / "weights" / "best.pt")

    if cfg["train"].get("enabled", False):
        train_artifact = train_yolo_obb(cfg)
        best_pt = train_artifact.best_pt_path
    elif not best_pt.exists():
        raise FileNotFoundError(f"Training disabled but best.pt was not found: {best_pt}")

    if cfg.get("pt_eval", {}).get("enabled", False):
        pt_eval_metrics = eval_yolo_obb_model(
            cfg=cfg,
            model_path=best_pt,
            eval_key="pt_eval",
            run_dir=run_dir,
            name="pt_eval",
        )

    if train_artifact is not None:
        if not best_pt.exists():
            raise FileNotFoundError(f"ONNX export requires a trained best.pt, but it was not found: {best_pt}")
        export_artifact = export_yolo_onnx(cfg, best_pt)

    if cfg.get("onnx_eval", {}).get("enabled", False):
        onnx_path = best_pt.with_suffix(".onnx")
        if not onnx_path.exists():
            if export_artifact and export_artifact.onnx_path.exists():
                onnx_path = export_artifact.onnx_path
            else:
                raise FileNotFoundError("onnx_eval.enabled=true but exported ONNX was not found.")
        onnx_eval_metrics = eval_yolo_obb_model(
            cfg=cfg,
            model_path=onnx_path,
            eval_key="onnx_eval",
            run_dir=run_dir,
            name="onnx_eval",
        )

    if cfg["predict"].get("enabled", False):
        predict_model = export_artifact.onnx_path if export_artifact else best_pt
        predict_with_yolo(cfg, predict_model)

    artifact = YoloOBBExperimentArtifact(
        train_artifact=train_artifact,
        onnx_export_artifact=export_artifact,
        pt_eval_metrics=pt_eval_metrics,
        onnx_eval_metrics=onnx_eval_metrics,
        metadata={"config": cfg},
    )
    report_path = write_experiment_report(run_dir, artifact, cfg)
    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "best_pt_path": str(best_pt),
        "best_onnx_path": str(export_artifact.onnx_path) if export_artifact else None,
        "pt_eval_metrics": pt_eval_metrics,
        "onnx_eval_metrics": onnx_eval_metrics,
    }


def train_yolo_obb(cfg: dict[str, Any]) -> YoloOBBTrainArtifact:
    architecture = cfg["model"]["architecture"]
    pretrained = cfg["model"]["pretrained"]
    model = load_yolo(
        architecture,
        task="obb",
        pretrained=pretrained,
        shift_from=cfg["model"].get("shift_from"),
    )
    train_args = dict(cfg["train"])
    train_args.pop("enabled", None)
    train_args.update(
        {
            "data": cfg["dataset"]["data_yaml"],
            "project": str(Path(cfg["dataset"]["project_root"]).resolve()),
            "name": cfg["experiment"]["name"],
        }
    )
    result = model.train(**train_args)
    run_dir = Path(getattr(result, "save_dir", Path(cfg["dataset"]["project_root"]) / cfg["experiment"]["name"]))
    return YoloOBBTrainArtifact(
        model_name=Path(architecture).name,
        dataset_variant=cfg["experiment"]["dataset_variant"],
        data_yaml_path=Path(cfg["dataset"]["data_yaml"]),
        run_dir=run_dir,
        best_pt_path=run_dir / "weights" / "best.pt",
        last_pt_path=run_dir / "weights" / "last.pt",
        train_metrics=metrics_dict(result),
        val_metrics=metrics_dict(getattr(result, "validator", None)),
        metadata={
            "experiment_name": cfg["experiment"]["name"],
            "pretrained": str(pretrained),
            "transfer": model._blastocyst_transfer,
        },
    )


def eval_yolo_obb_model(
    cfg: dict[str, Any],
    model_path: Path,
    eval_key: str,
    run_dir: Path,
    name: str,
) -> dict[str, Any]:
    """Evaluate a PyTorch or ONNX OBB model with AP25 plus the standard AP50:95 grid."""
    eval_cfg = dict(cfg[eval_key])
    eval_cfg.pop("enabled", None)
    eval_cfg.update(
        {
            "data": cfg["dataset"]["data_yaml"],
            "project": str(run_dir),
            "name": name,
            "exist_ok": True,
        }
    )
    model = load_yolo(model_path, task="obb")
    import ultralytics
    from ultralytics.models.yolo.obb import OBBValidator
    from ultralytics.utils import YAML

    data_cfg = YAML.load(eval_cfg["data"])
    split_list = data_cfg.get(eval_cfg["split"])
    validator = OBBValidator(args=eval_cfg)
    print(f"{eval_key} model: {model_path.resolve()}")
    print(f"Ultralytics import: {Path(ultralytics.__file__).resolve()}")
    print(
        f"Loaded model: {type(model.model).__module__}.{type(model.model).__name__}, "
        f"parameters={sum(parameter.numel() for parameter in model.model.parameters())}"
    )
    print(f"Dataset YAML: {Path(eval_cfg['data']).resolve()}")
    print(f"{eval_cfg['split']} list: {Path(split_list).resolve() if split_list else split_list}")
    print(f"model.val arguments: {json.dumps(json_safe(eval_cfg), ensure_ascii=False, sort_keys=True)}")
    print(f"Validator IoU vector: {validator.iouv.tolist()}")

    metrics = model.val(**eval_cfg)
    result = extract_obb_eval_metrics(metrics)
    report = {
        "config": cfg,
        "model": str(model_path.resolve()),
        **result,
    }
    report_path = run_dir / name / "metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "report_path": str(report_path)}


def export_yolo_onnx(cfg: dict[str, Any], best_pt: Path) -> YoloONNXExportArtifact:
    export_cfg = cfg["onnx_export"]
    onnx_path = export_blastocyst_yolo_to_onnx(
        source_checkpoint=best_pt,
        output_dir=export_cfg["output_dir"],
        imgsz=int(export_cfg.get("imgsz", cfg["train"].get("imgsz", 512))),
        opset=int(export_cfg.get("opset", 17)),
        simplify=bool(export_cfg.get("simplify", True)),
        device=export_cfg.get("device", cfg["train"].get("device", 0)),
    )
    return YoloONNXExportArtifact(
        source_pt_path=best_pt,
        onnx_path=onnx_path,
        export_dir=Path(export_cfg["output_dir"]),
        opset=int(export_cfg.get("opset", 17)),
        simplify=bool(export_cfg.get("simplify", True)),
        metadata={"imgsz": int(export_cfg.get("imgsz", cfg["train"].get("imgsz", 512)))},
    )


def extract_obb_eval_metrics(metrics: Any) -> dict[str, Any]:
    all_ap = metrics.box.all_ap
    if getattr(all_ap, "ndim", 0) != 2 or all_ap.shape[1] != 11:
        raise RuntimeError(f"Expected an AP matrix with 11 IoU columns, got shape {getattr(all_ap, 'shape', None)}.")
    return {
        "images": int(metrics.nt_per_image.sum()),
        "instances": int(metrics.nt_per_class.sum()),
        "AP25": float(all_ap[:, 0].mean() * 100),
        "AP50": float(all_ap[:, 1].mean() * 100),
        "mAP50-95": float(all_ap[:, 1:].mean() * 100),
        "precision": float(metrics.box.p[0] * 100),
        "recall": float(metrics.box.r[0] * 100),
    }


def predict_with_yolo(cfg: dict[str, Any], model_path: Path) -> None:
    source = cfg["predict"].get("source")
    if not source:
        raise ValueError("predict.enabled=true requires predict.source.")
    model = load_yolo(model_path, task="obb")
    model.predict(
        source=source,
        project=cfg["predict"]["output_dir"],
        name=cfg["experiment"]["name"],
        imgsz=int(cfg["predict"].get("imgsz", cfg["train"].get("imgsz", 512))),
        device=cfg["predict"].get("device", cfg["train"].get("device", 0)),
    )


def load_yolo(
    model_ref: str | Path,
    task: str = "obb",
    pretrained: str | Path | None = None,
    shift_from: int | None = None,
) -> Any:
    try:
        local_ul = _local_ultralytics_root()
        if not local_ul.is_dir():
            raise ImportError(f"Local ultralytics-cbam package not found: {local_ul}")
        if str(local_ul) not in sys.path:
            sys.path.insert(0, str(local_ul))
        import ultralytics
        imported_from = Path(ultralytics.__file__).resolve()
        if not imported_from.is_relative_to(local_ul):
            raise ImportError(
                f"Expected local ultralytics-cbam from {local_ul}, but imported {imported_from}. "
                "Restart the process with the local package first on PYTHONPATH."
            )
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required for the YOLO-OBB pipeline.") from exc

    model = YOLO(str(model_ref), task=task)
    if pretrained:
        model._blastocyst_transfer = _load_pretrained(model, Path(pretrained), shift_from)
    return model


def _load_pretrained(model: Any, weights_path: Path, shift_from: int | None) -> dict[str, Any]:
    """Deserialize a trusted Ultralytics checkpoint and prove compatible tensors were applied."""
    import torch
    from ultralytics.utils.downloads import attempt_download_asset

    if not weights_path.is_file():
        if weights_path.parent != Path("."):
            raise FileNotFoundError(f"Pretrained YOLO checkpoint not found: {weights_path}")
        weights_path = Path(attempt_download_asset(str(weights_path)))
    if not weights_path.is_file():
        raise FileNotFoundError(f"Ultralytics did not download the pretrained checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected an Ultralytics checkpoint dictionary: {weights_path}")
    source = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
    if source is None:
        raise KeyError(f"Checkpoint has neither 'ema' nor 'model': {weights_path}")
    source_state = source.float().state_dict() if hasattr(source, "state_dict") else source
    if not isinstance(source_state, dict):
        raise TypeError(f"Checkpoint model does not expose a state_dict: {weights_path}")

    target = model.model
    target_state = target.state_dict()
    transferred: dict[str, Any] = {}
    for source_key, tensor in source_state.items():
        target_key = _shift_layer_key(source_key, shift_from)
        if target_key in target_state and target_state[target_key].shape == tensor.shape:
            transferred[target_key] = tensor.float()
    if not transferred:
        raise RuntimeError(f"No compatible tensors found in pretrained checkpoint: {weights_path}")

    target.load_state_dict(transferred, strict=False)
    if not all(torch.equal(target.state_dict()[key].cpu(), value.cpu()) for key, value in transferred.items()):
        raise RuntimeError("Pretrained tensor verification failed after load_state_dict().")
    # Model.train() only hands its populated model to the trainer when checkpoint metadata is present.
    model.ckpt = checkpoint
    model.ckpt_path = str(weights_path.resolve())
    model.overrides["pretrained"] = model.ckpt_path
    report = {
        "checkpoint": str(weights_path.resolve()),
        "tensors": len(transferred),
        "target_tensors": len(target_state),
        "parameters": int(sum(value.numel() for value in transferred.values())),
        "shift_from": shift_from,
    }
    print(
        f"Applied {report['tensors']}/{report['target_tensors']} pretrained tensors "
        f"({report['parameters']} parameters) from {weights_path}"
    )
    return report


def _shift_layer_key(key: str, shift_from: int | None) -> str:
    if shift_from is None or not key.startswith("model."):
        return key
    parts = key.split(".", 2)
    if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) < shift_from:
        return key
    return f"model.{int(parts[1]) + 1}.{parts[2]}"


def _local_ultralytics_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "ultralytics-cbam").resolve()


def metrics_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    output: dict[str, Any] = {}
    raw = getattr(result, "results_dict", None)
    if isinstance(raw, dict):
        output.update(json_safe(raw))
    for attr in ("box", "obb"):
        metrics = getattr(result, attr, None)
        if metrics is not None:
            maps = getattr(metrics, "maps", None)
            output[attr] = json_safe(maps if maps is not None else getattr(metrics, "results_dict", {}))
    return output


def write_experiment_report(run_dir: Path, artifact: YoloOBBExperimentArtifact, cfg: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = artifact_to_json(artifact)
    report["config_snapshot"] = cfg
    path = run_dir / "yolo_obb_experiment_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def artifact_to_json(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        return {key: artifact_to_json(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): artifact_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [artifact_to_json(item) for item in value]
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
