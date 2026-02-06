from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot


@runtime_checkable
class ReporterProtocol(Protocol):
    """Protocol for all metric reporters."""

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        """Initialize the reporter for a training run."""
        ...

    def log_metrics(self, snapshot: MetricsSnapshot) -> None:
        """Log a metrics snapshot."""
        ...

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        """Log an artifact (e.g., model checkpoint)."""
        ...

    def finalize(self) -> None:
        """Clean up and finalize reporting."""
        ...
