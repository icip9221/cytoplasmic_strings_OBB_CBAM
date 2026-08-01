from __future__ import annotations

from pathlib import Path
from typing import Any

from abs_interface import AbsAlgorithmProcessHandler
from registry.registers import Blastocyst_handler_register

from ._yolo_detection_ops import run_yolo_experiment, validate_yolo_config


@Blastocyst_handler_register.register("YoloDetectionHandler")
class YoloDetectionHandler(AbsAlgorithmProcessHandler):
    """Resolve and orchestrate shared Ultralytics HBB and OBB stages."""

    def _pre_process(self, data: Any) -> dict[str, Any]:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("YoloDetectionHandler expects runtime data to be a dictionary.")
        if data.get("pipeline_error"):
            return data
        try:
            cfg = self._runtime_config(data)
            validate_yolo_config(cfg)
        except Exception as exc:
            data["pipeline_error"] = str(exc)
            return data
        data["_yolo_detection_config"] = cfg
        return data

    def _main_process(self, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("pipeline_error"):
            return data
        result = run_yolo_experiment(data["_yolo_detection_config"])
        data["yolo_run_dir"] = result["run_dir"]
        data["yolo_report_path"] = result["report_path"]
        data.setdefault("summary", {}).update(
            {
                "yolo_task": result["task"],
                "yolo_run_dir": result["run_dir"],
                "yolo_report_path": result["report_path"],
                "yolo_best_pt_path": result["best_pt_path"],
                "yolo_best_onnx_path": result["best_onnx_path"],
                "yolo_pt_eval_metrics": result["pt_eval_metrics"],
                "yolo_onnx_eval_metrics": result["onnx_eval_metrics"],
            }
        )
        return data

    def _post_process(self, data: dict[str, Any]) -> dict[str, Any]:
        data.pop("_yolo_detection_config", None)
        return data

    def _runtime_config(self, data: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(self.config)
        task = str(self.config["task"])
        experiment = dict(self.config.get("experiment", {}))
        model = dict(self.config.get("model", {}))
        train = dict(self.config.get("train", {}))
        pt_eval = dict(self.config.get("pt_eval", {}))
        onnx_export = dict(self.config.get("onnx_export", {}))
        onnx_eval = dict(self.config.get("onnx_eval", {}))
        predict = dict(self.config.get("predict", {}))

        dataset = str(experiment.get("dataset", task))
        runtime = data.get("pipeline") or data.get("paths") or {}
        yolo_runtime = dict(runtime.get("yolo_detection", {}))
        datasets = dict(yolo_runtime.get("datasets", {}))
        dataset_cfg = dict(datasets.get(dataset, {}))
        if not dataset_cfg:
            raise KeyError(f"Dataset {dataset!r} not found in runtime pipeline.yolo_detection.datasets.")
        missing = sorted({"data_yaml", "project_root"}.difference(dataset_cfg))
        if missing:
            raise KeyError(f"Dataset {dataset!r} is missing runtime keys: {missing}")
        inference = dict(yolo_runtime.get("inference", {}))

        model_arch = model.get("architecture") or model.get("name")
        model_weight = model.get("weight")
        experiment["name"] = experiment.get("name") or f"{Path(str(model_arch or 'yolo')).stem}_{dataset}"
        experiment["dataset"] = dataset
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
            or str(Path(dataset_cfg["project_root"]) / "inference")
        )

        onnx_export["output_dir"] = str(Path(export_output_dir))
        predict["output_dir"] = str(Path(inference_output_dir))
        cfg["task"] = task
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

    def handle(self, data: Any) -> Any:
        if self.config.get("task") not in {"detect", "obb"}:
            return self._handle_next(data)
        return self._handle_next(self.process(data))
