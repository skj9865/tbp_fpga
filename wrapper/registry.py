from core.base_algorithm import BaseAlgorithm

_REGISTRY: dict[str, type[BaseAlgorithm]] = {}


def register_algorithm(cls: type[BaseAlgorithm]) -> type[BaseAlgorithm]:
    """Class decorator that registers an algorithm by its name()."""
    instance = cls.__new__(cls)
    key = instance.name()
    if key in _REGISTRY:
        raise ValueError(f"Algorithm '{key}' is already registered")
    _REGISTRY[key] = cls
    return cls


class AlgorithmRegistry:
    @staticmethod
    def get(name: str) -> type[BaseAlgorithm]:
        if name not in _REGISTRY:
            available = ", ".join(_REGISTRY.keys()) or "(none)"
            raise KeyError(f"Algorithm '{name}' not found. Available: {available}")
        return _REGISTRY[name]

    @staticmethod
    def list_all() -> list[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def is_registered(name: str) -> bool:
        return name in _REGISTRY
