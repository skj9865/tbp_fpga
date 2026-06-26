from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """All algorithms must implement this interface."""

    @abstractmethod
    def name(self) -> str:
        """Return the algorithm's short identifier (e.g. 'ff', 'monty')."""

    @abstractmethod
    def configure(self, config: dict) -> None:
        """Apply framework-level and algorithm-specific configuration."""

    @abstractmethod
    def train(self, **kwargs) -> dict:
        """Run training. Return a dict with at least 'accuracy' or 'loss'."""

    @abstractmethod
    def evaluate(self, **kwargs) -> dict:
        """Run evaluation. Return a dict with at least 'accuracy'."""

    @abstractmethod
    def get_supported_datasets(self) -> list:
        """Return list of supported dataset names (e.g. ['cifar10', 'mnist'])."""
