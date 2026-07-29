from abc import ABC, abstractmethod
from typing import Any


class AbstractProcess(ABC):
    """Template process: pre-process, main process, then post-process."""

    def process(self, data: Any) -> Any:
        data = self._pre_process(data)
        data = self._main_process(data)
        return self._post_process(data)

    @abstractmethod
    def _pre_process(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _main_process(self, data: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _post_process(self, data: Any) -> Any:
        raise NotImplementedError
