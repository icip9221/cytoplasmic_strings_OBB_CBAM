from __future__ import annotations

from pathlib import Path
from typing import Any

from abs_interface import AbsAlgorithmProcessHandler
from registry.registers import Blastocyst_handler_register

from ._yolo_obb_ops import run_yolo_obb_experiment, validate_yolo_runtime_config


@Blastocyst_handler_register.register("BlastocystYoloOBBHandler")
class BlastocystYoloOBBHandler(AbsAlgorithmProcessHandler):
    """Resolve and run Ultralytics YOLO-OBB training, export, and evaluation stages."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        raw_config = config or {}
        self.config = raw_config.get("config", raw_config)

    def _pre_process(self, data: Any) -> dict[str, Any]:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("BlastocystYoloOBBHandler expects runtime data to be a dictionary.")
        if data.get("pipeline_error"):
            return data
        try:
            cfg = self._runtime_config(data)
            validate_yolo_runtime_config(cfg)
        except Exception as exc:
            data["pipeline_error"] = str(exc)
            return data
        data["_blastocyst_yolo_obb_config"] = cfg
        return data

    def _main_process(self, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("pipeline_error"):
            return data
        result = run_yolo_obb_experiment(data["_blastocyst_yolo_obb_config"])
        data["yolo_obb_run_dir"] = result["run_dir"]
        data["yolo_obb_report_path"] = result["report_path"]
        data.setdefault("summary", {}).update(
            {
                "yolo_obb_run_dir": result["run_dir"],
                "yolo_obb_report_path": result["report_path"],
                "yolo_obb_best_pt_path": result["best_pt_path"],
                "yolo_obb_best_onnx_path": result["best_onnx_path"],
                "yolo_obb_pt_eval_metrics": result["pt_eval_metrics"],
                "yolo_obb_onnx_eval_metrics": result["onnx_eval_metrics"],
            }
        )
        return data

    def _post_process(self, data: dict[str, Any]) -> dict[str, Any]:
        data.pop("_blastocyst_yolo_obb_config", None)
        return data

    def _runtime_config(self, data: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(self.config)
        experiment = dict(self.config.get("experiment", {}))
        model = dict(self.config.get("model", {}))
        train = dict(self.config.get("train", {}))
        pt_eval = dict(self.config.get("pt_eval", {}))
        onnx_export = dict(self.config.get("onnx_export", {}))
        onnx_eval = dict(self.config.get("onnx_eval", {}))
        predict = dict(self.config.get("predict", {}))

        variant = str(experiment.get("dataset_variant", "mix"))
        runtime = data.get("pipeline") or data.get("paths") or {}
        yolo_runtime = dict(runtime.get("yolo_obb", {}))
        datasets = dict(yolo_runtime.get("datasets", {}))
        dataset_cfg = dict(datasets.get(variant, {}))
        if not dataset_cfg:
            raise KeyError(f"dataset_variant {variant!r} not found in runtime pipeline.yolo_obb.datasets.")
        missing = sorted({"data_yaml", "project_root"}.difference(dataset_cfg))
        if missing:
            raise KeyError(f"dataset_variant {variant!r} is missing runtime keys: {missing}")
        inference = dict(yolo_runtime.get("inference", {}))

        model_arch = model.get("architecture") or model.get("name")
        model_weight = model.get("weight")
        experiment["name"] = experiment.get("name") or f"{Path(str(model_arch or 'yolo_obb')).stem}_{variant}"
        experiment["dataset_variant"] = variant
        if model_arch:
            model["architecture"] = str(Path(model_arch).resolve())
        if model_weight:
            weight_path = Path(model_weight)
            model["pretrained"] = (
                str(weight_path.resolve())
                if weight_path.exists() or weight_path.parent != Path(".")
                else str(model_weight)
            )

        run_dir = Path(dataset_cfg["project_root"]) / experiment["name"]
        export_output_dir = onnx_export.get("output_dir") or str(run_dir / "weights")
        inference_output_dir = (
            predict.get("output_dir")
            or inference.get("output_dir")
            or "outputs/yolo_obb/inference"
        )

        onnx_export["output_dir"] = str(Path(export_output_dir))
        predict["output_dir"] = str(Path(inference_output_dir))
        cfg["experiment"] = experiment
        cfg["model"] = model
        cfg["dataset"] = {
            "data_yaml": str(Path(dataset_cfg["data_yaml"]).resolve()),
            "project_root": str(Path(dataset_cfg["project_root"]).resolve()),
        }
        cfg["train"] = train
        cfg["pt_eval"] = pt_eval
        cfg["onnx_export"] = onnx_export
        cfg["onnx_eval"] = onnx_eval
        cfg["predict"] = predict
        return cfg
