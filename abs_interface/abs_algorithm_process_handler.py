from typing import Any

from .abs_handler import AbstractHandler
from .abs_process import AbstractProcess


class AbsAlgorithmProcessHandler(AbstractHandler, AbstractProcess):
    """Base class for algorithm handlers that can also run as processes."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        raw_config = config or {}
        self.config = raw_config.get("config", raw_config)

    def handle(self, data: Any) -> Any:
        data = self.process(data)
        return self._handle_next(data)
