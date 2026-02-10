from __future__ import annotations

from collections.abc import Iterable

from .metric_reporting_protocol import MetricsReportingProtocol
from .metric_result import MetricResult


class MetricsReportingManager:
    """Registry of reporters that fans out results to all registered instances."""

    def __init__(self, reporters: Iterable[MetricsReportingProtocol] = ()) -> None:
        self._reporters: list[MetricsReportingProtocol] = list(reporters)

    def register(self, reporter: MetricsReportingProtocol) -> None:
        self._reporters.append(reporter)

    def report(self, results: Iterable[MetricResult]) -> None:
        materialized = list(results)
        for reporter in self._reporters:
            reporter.Report(materialized)

    def clear(self) -> None:
        self._reporters.clear()
