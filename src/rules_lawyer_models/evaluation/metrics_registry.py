from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_protocol import MetricProtocol


class MetricsRegistry:
    """Registry of available metrics with factory methods."""

    _metrics: dict[str, type[MetricProtocol]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[MetricProtocol]], type[MetricProtocol]]:
        """Decorator to register a metric class."""

        def decorator(metric_cls: type[MetricProtocol]) -> type[MetricProtocol]:
            cls._metrics[name] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> MetricProtocol:
        """Create a metric instance by name."""
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric: {name}. Available: {available}")
        return cls._metrics[name](**kwargs)

    @classmethod
    def create_all(cls, names: list[str], **kwargs: Any) -> list[MetricProtocol]:
        """Create multiple metric instances."""
        return [cls.create(name, **kwargs) for name in names]

    @classmethod
    def available(cls) -> list[str]:
        """List available metric names."""
        return list(cls._metrics.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a metric is registered."""
        return name in cls._metrics
