from abc import ABC, abstractmethod
from typing import Any


class AbstractHandler(ABC):
    """Chain-of-responsibility handler contract."""

    def __init__(self) -> None:
        self._next_handler: AbstractHandler | None = None

    def set_next(self, handler: "AbstractHandler") -> "AbstractHandler":
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, data: Any) -> Any:
        raise NotImplementedError

    def _handle_next(self, data: Any) -> Any:
        if self._next_handler is None:
            return data
        return self._next_handler.handle(data)
