from typing import Any

from .abs_handler import AbstractHandler
from .abs_process import AbstractProcess


class AbsAlgorithmProcessHandler(AbstractHandler, AbstractProcess):
    """Base class for algorithm handlers that can also run as processes."""

    def handle(self, data: Any) -> Any:
        data = self.process(data)
        return self._handle_next(data)
