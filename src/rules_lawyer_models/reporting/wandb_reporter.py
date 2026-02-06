from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot


class WandbReporter:
    """Reports metrics to Weights & Biases."""

    def __init__(self, project: str, entity: str | None = None):
        self._project = project
        self._entity = entity
        self._run: Any = None  # wandb.Run, but we avoid importing at module level

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        import wandb

        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            config=config,
        )

    def log_metrics(self, snapshot: MetricsSnapshot) -> None:
        if self._run is None:
            return

        metrics: dict[str, Any] = {r.name: r.value for r in snapshot.results.values()}
        metrics["step"] = snapshot.step
        if snapshot.epoch is not None:
            metrics["epoch"] = snapshot.epoch

        self._run.log(metrics, step=snapshot.step)

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        if self._run is None:
            return

        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finalize(self) -> None:
        if self._run:
            self._run.finish()
            self._run = None
