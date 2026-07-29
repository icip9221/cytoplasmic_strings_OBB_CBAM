from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """Small name-to-object registry used by pipeline builders."""

    def __init__(self, name: str):
        self.name = name
        self._objects: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            self._objects[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        try:
            return self._objects[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._objects)) or "<empty>"
            raise KeyError(
                f"{name!r} is not registered in {self.name}. Available: {available}"
            ) from exc


pipeline_builder_register = Registry("pipeline_builder")
Blastocyst_handler_register = Registry("blastocyst_handler")

__all__ = ["Registry", "pipeline_builder_register", "Blastocyst_handler_register"]
