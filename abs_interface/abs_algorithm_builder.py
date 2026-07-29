from abc import ABC, abstractmethod
from typing import Any


class AbsAlgorithmBuilder(ABC):
    """Builder contract for constructing a pipeline product from config."""

    @abstractmethod
    def build(self, cfg: dict[str, Any]) -> Any:
        raise NotImplementedError
