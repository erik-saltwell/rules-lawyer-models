from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot


class DiskReporter:
    """Persists metrics to local disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self._output_dir = Path(output_dir)
        self._run_dir: Path | None = None
        self._metrics_file: Path | None = None
        self._history: list[dict[str, Any]] = []

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        self._run_dir = self._output_dir / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_file = self._run_dir / "metrics.json"
        self._history = []

        # Save config
        config_file = self._run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _write_metrics(self) -> None:
        if self._metrics_file:
            with open(self._metrics_file, "w") as f:
                json.dump(self._history, f, indent=2)

    def log_metrics(self, snapshot: MetricsSnapshot) -> None:
        self._history.append(snapshot.to_dict())
        self._write_metrics()

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        # Record artifact reference in metrics
        if self._history:
            self._history[-1].setdefault("artifacts", []).append(
                {
                    "name": name,
                    "path": path,
                    "type": artifact_type,
                }
            )
            self._write_metrics()

    def finalize(self) -> None:
        self._write_metrics()

    @property
    def run_dir(self) -> Path | None:
        """Return the current run directory, if initialized."""
        return self._run_dir
