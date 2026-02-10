from __future__ import annotations

from collections.abc import Iterable

from rules_lawyer_models.utils.logging_protocol import LoggingProtocol

from .metric_result import MetricResult


class LoggerMetricsReporter:
    """Reports metric results as a table via a LoggingProtocol implementation (e.g. RichConsoleLogger)."""

    def __init__(self, logger: LoggingProtocol) -> None:
        self._logger = logger

    def Report(self, results: Iterable[MetricResult]) -> None:
        self._logger.report_table_message({r.metric_name: r.metric_result for r in results})
