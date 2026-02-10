from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from evaluation.metric_result import MetricResult


class MetricsReportingProtocol(Protocol):
    def Report(self, results: Iterable[MetricResult]) -> None: ...
