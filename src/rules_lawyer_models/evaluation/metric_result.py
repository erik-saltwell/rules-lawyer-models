from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Immutable result from a metric computation."""

    name: str
    value: float
    step: int
    epoch: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "epoch": self.epoch,
            **self.metadata,
        }


@dataclass(slots=True)
class MetricsSnapshot:
    """Collection of metrics at a point in time."""

    step: int
    epoch: float | None = None
    results: dict[str, MetricResult] = field(default_factory=dict)

    def add(self, result: MetricResult) -> None:
        self.results[result.name] = result

    def get(self, name: str) -> MetricResult | None:
        return self.results.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "metrics": {k: v.value for k, v in self.results.items()},
        }
