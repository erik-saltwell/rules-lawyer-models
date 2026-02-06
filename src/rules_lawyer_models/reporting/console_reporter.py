from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot
    from rules_lawyer_models.utils import LoggingProtocol


class ConsoleReporter:
    """Reports metrics to console via LoggingProtocol."""

    def __init__(self, logger: LoggingProtocol):
        self._logger = logger
        self._run_name: str = ""

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        self._run_name = run_name
        self._logger.report_message(f"Starting run: {run_name}")

    def log_metrics(self, snapshot: MetricsSnapshot) -> None:
        metrics_str = ", ".join(f"{k}={v.value:.4f}" for k, v in snapshot.results.items())
        step_info = f"Step {snapshot.step}"
        if snapshot.epoch is not None:
            step_info += f" (Epoch {snapshot.epoch:.2f})"

        self._logger.report_message(f"{step_info}: {metrics_str}")

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        self._logger.report_message(f"Saved {artifact_type}: {name} -> {path}")

    def finalize(self) -> None:
        self._logger.report_message(f"Run {self._run_name} complete")
