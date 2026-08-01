from typing import Any
from importlib import import_module

from abs_interface import AbsAlgorithmBuilder, AbstractProcess
from registry.registers import Blastocyst_handler_register, pipeline_builder_register


class BlastocystAlgorithm(AbstractProcess):
    """Pipeline product that executes a chain of Blastocyst handlers."""

    def __init__(self) -> None:
        self.main_processor: Any | None = None

    def _pre_process(self, data: Any) -> Any:
        return data

    def _main_process(self, data: Any) -> Any:
        if self.main_processor is None:
            return data
        return self.main_processor.handle(data)

    def _post_process(self, data: Any) -> Any:
        return data


class _BasePipelineBuilder(AbsAlgorithmBuilder):
    """Base builder that wires handler chains from YAML."""

    HANDLER_REGISTERS = []

    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}
        self.product = BlastocystAlgorithm()

    def build(self, cfg: dict[str, Any]) -> BlastocystAlgorithm:
        self._cfg = cfg
        self.product = BlastocystAlgorithm()
        pipeline_cfg = self._cfg.get("pipeline", {})

        first_handler = None
        previous_handler = None
        for handler_name, handler_cfg in pipeline_cfg.items():
            handler_cls = self._get_handler_cls(handler_name)
            handler = handler_cls(handler_cfg)
            if first_handler is None:
                first_handler = handler
            if previous_handler is not None:
                previous_handler.set_next(handler)
            previous_handler = handler

        self.product.main_processor = first_handler
        return self.product

    def _get_handler_cls(self, handler_name: str) -> Any:
        for handler_register in self.HANDLER_REGISTERS:
            try:
                return handler_register.get(handler_name)
            except KeyError:
                continue
        raise KeyError(f"{handler_name!r} is not registered in this pipeline builder.")


@pipeline_builder_register.register("Blastocyst")
class BlastocystBuilder(_BasePipelineBuilder):
    """Build Blastocyst-domain pipelines from YAML config."""

    HANDLER_REGISTERS = [Blastocyst_handler_register]

    def build(self, cfg: dict[str, Any]) -> BlastocystAlgorithm:
        handler_modules = {
            "YoloDetectionHandler": "algorithms.Blastocyst.yolo_detection_handler",
        }
        for handler_name in cfg.get("pipeline", {}):
            if module := handler_modules.get(handler_name):
                import_module(module)

        return super().build(cfg)
