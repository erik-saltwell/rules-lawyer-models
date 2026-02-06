from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot
    from rules_lawyer_models.reporting.reporter_protocol import ReporterProtocol


class ReportingDispatcher:
    """Dispatches metrics to multiple reporters (Composite pattern)."""

    def __init__(self, reporters: list[ReporterProtocol] | None = None):
        self._reporters: list[ReporterProtocol] = reporters or []

    def add_reporter(self, reporter: ReporterProtocol) -> None:
        self._reporters.append(reporter)

    def remove_reporter(self, reporter: ReporterProtocol) -> None:
        self._reporters.remove(reporter)

    @property
    def reporters(self) -> list[ReporterProtocol]:
        """Return the list of registered reporters."""
        return self._reporters.copy()

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        for reporter in self._reporters:
            reporter.initialize(run_name, config)

    def log_metrics(self, snapshot: MetricsSnapshot) -> None:
        for reporter in self._reporters:
            reporter.log_metrics(snapshot)

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        for reporter in self._reporters:
            reporter.log_artifact(name, path, artifact_type)

    def finalize(self) -> None:
        for reporter in self._reporters:
            reporter.finalize()
