from __future__ import annotations

from typing import Protocol

from datasets import Dataset

from .metric_result import MetricResult


class MetricProtocol(Protocol):
    def compute_metric(self, dataset: Dataset) -> MetricResult: ...
