# Metrics Collection and Reporting System - Technical Design

## Overview

This document describes the architecture for a metrics collection, evaluation, and reporting system for the `rules-lawyer-models` training pipeline. The design emphasizes flexibility, extensibility, and clean separation of concerns.

## Requirements Summary

1. **Architectural flexibility** for collecting multiple metrics using appropriate design patterns
2. **Dual storage**: local persistence and W&B (Weights & Biases) reporting
3. **Best model tracking**: feed metrics back to trainer for checkpoint selection
4. **Reporting abstraction**: unified interface for disk, console, and W&B outputs
5. **Evaluation abstraction**: pluggable metric computation in the evaluation directory

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                             │
│                                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │   Trainer   │───▶│ MetricsCollector │───▶│   ReportingDispatcher   │ │
│  │ (SFTTrainer)│    │   (Observer Hub) │    │    (Strategy Pattern)   │ │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘ │
│         │                    │                          │               │
│         │                    │                          ▼               │
│         │                    │              ┌─────────────────────────┐ │
│         │                    │              │       Reporters         │ │
│         │                    │              │  ┌───────┐ ┌──────────┐ │ │
│         │                    │              │  │Console│ │   Disk   │ │ │
│         │                    │              │  └───────┘ └──────────┘ │ │
│         │                    │              │  ┌──────────┐           │ │
│         │                    │              │  │  W&B     │           │ │
│         │                    │              │  └──────────┘           │ │
│         │                    │              └─────────────────────────┘ │
│         │                    │                                          │
│         │                    ▼                                          │
│         │         ┌──────────────────┐                                  │
│         │         │  MetricObservers │                                  │
│         │         │  (Evaluation)    │                                  │
│         │         │  ┌────────────┐  │                                  │
│         │         │  │   Loss     │  │                                  │
│         │         │  ├────────────┤  │                                  │
│         │         │  │ Perplexity │  │                                  │
│         │         │  ├────────────┤  │                                  │
│         │         │  │  Accuracy  │  │                                  │
│         │         │  ├────────────┤  │                                  │
│         │         │  │  Custom... │  │                                  │
│         │         │  └────────────┘  │                                  │
│         │         └──────────────────┘                                  │
│         │                    │                                          │
│         │                    ▼                                          │
│         │         ┌──────────────────┐                                  │
│         └────────▶│ BestModelTracker │                                  │
│                   │  (Checkpoint Sel)│                                  │
│                   └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns Used

### 1. Observer Pattern (Evaluation Directory)
Used for metric computation. Each metric is an **observer** that receives evaluation data and computes its specific metric. This allows:
- Adding new metrics without modifying existing code
- Running multiple metrics in parallel
- Clean separation of metric logic

### 2. Strategy Pattern (Reporting Directory)
Used for reporting destinations. Each reporter implements a common interface but with different output strategies (console, disk, W&B). This allows:
- Swapping reporters at runtime
- Using multiple reporters simultaneously
- Easy testing with mock reporters

### 3. Composite Pattern (MetricsCollector)
The MetricsCollector acts as a hub that aggregates metrics from multiple observers and dispatches to multiple reporters.

---

## Directory Structure

```
src/rules_lawyer_models/
├── evaluation/
│   ├── __init__.py
│   ├── metric_protocol.py          # Base protocol for all metrics
│   ├── metric_result.py            # Data classes for metric results
│   ├── metrics_registry.py         # Registry for available metrics
│   ├── loss_metric.py              # Loss computation
│   ├── perplexity_metric.py        # Perplexity computation
│   └── accuracy_metric.py          # Accuracy computation
│
├── reporting/
│   ├── __init__.py
│   ├── reporter_protocol.py        # Base protocol for reporters
│   ├── reporting_dispatcher.py     # Dispatches to multiple reporters
│   ├── console_reporter.py         # Console/terminal output
│   ├── disk_reporter.py            # Local file storage (JSON/CSV)
│   └── wandb_reporter.py           # Weights & Biases integration
│
└── training/
    ├── ...
    ├── metrics_collector.py        # Central hub for metrics
    ├── best_model_tracker.py       # Tracks best checkpoint
    └── training_pipeline.py        # Updated pipeline
```

---

## Evaluation Module Design

### MetricProtocol (evaluation/metric_protocol.py)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rules_lawyer_models.evaluation.metric_result import MetricResult


@runtime_checkable
class MetricProtocol(Protocol):
    """Protocol for all metric observers."""

    @property
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        ...

    def compute(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_data: list[dict],
        predictions: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MetricResult:
        """Compute the metric value."""
        ...

    def reset(self) -> None:
        """Reset any accumulated state."""
        ...
```

### MetricResult (evaluation/metric_result.py)

```python
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
    epoch: float | None
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
```

### MetricsRegistry (evaluation/metrics_registry.py)

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_protocol import MetricProtocol


class MetricsRegistry:
    """Registry of available metrics with factory methods."""

    _metrics: dict[str, type[MetricProtocol]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a metric class."""
        def decorator(metric_cls: type[MetricProtocol]):
            cls._metrics[name] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> MetricProtocol:
        """Create a metric instance by name."""
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}. Available: {list(cls._metrics.keys())}")
        return cls._metrics[name](**kwargs)

    @classmethod
    def create_all(cls, names: list[str], **kwargs) -> list[MetricProtocol]:
        """Create multiple metric instances."""
        return [cls.create(name, **kwargs) for name in names]

    @classmethod
    def available(cls) -> list[str]:
        """List available metric names."""
        return list(cls._metrics.keys())
```

### Example Metric Implementation (evaluation/loss_metric.py)

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rules_lawyer_models.evaluation.metric_result import MetricResult
from rules_lawyer_models.evaluation.metrics_registry import MetricsRegistry


@MetricsRegistry.register("loss")
class LossMetric:
    """Computes average loss on evaluation data."""

    def __init__(self):
        self._accumulated_loss: float = 0.0
        self._num_samples: int = 0

    @property
    def name(self) -> str:
        return "eval_loss"

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_data: list[dict],
        predictions: Tensor | None = None,
        labels: Tensor | None = None,
        step: int = 0,
        epoch: float | None = None,
    ) -> MetricResult:
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_data:
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return MetricResult(
            name=self.name,
            value=avg_loss,
            step=step,
            epoch=epoch,
        )

    def reset(self) -> None:
        self._accumulated_loss = 0.0
        self._num_samples = 0
```

---

## Reporting Module Design

### ReporterProtocol (reporting/reporter_protocol.py)

```python
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
```

### ReportingDispatcher (reporting/reporting_dispatcher.py)

```python
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
```

### ConsoleReporter (reporting/console_reporter.py)

```python
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
        metrics_str = ", ".join(
            f"{k}={v.value:.4f}" for k, v in snapshot.results.items()
        )
        step_info = f"Step {snapshot.step}"
        if snapshot.epoch is not None:
            step_info += f" (Epoch {snapshot.epoch:.2f})"

        self._logger.report_message(f"{step_info}: {metrics_str}")

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        self._logger.report_message(f"Saved {artifact_type}: {name} -> {path}")

    def finalize(self) -> None:
        self._logger.report_message(f"Run {self._run_name} complete")
```

### DiskReporter (reporting/disk_reporter.py)

```python
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
        self._metrics_file: Path | None = None
        self._history: list[dict[str, Any]] = []

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        run_dir = self._output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_file = run_dir / "metrics.json"
        self._history = []

        # Save config
        config_file = run_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metrics(self, snapshot: MetricsSnapshot) -> None:
        self._history.append(snapshot.to_dict())
        self._write_metrics()

    def log_artifact(self, name: str, path: str, artifact_type: str = "model") -> None:
        # Record artifact reference in metrics
        if self._history:
            self._history[-1].setdefault("artifacts", []).append({
                "name": name,
                "path": path,
                "type": artifact_type,
            })
            self._write_metrics()

    def finalize(self) -> None:
        self._write_metrics()

    def _write_metrics(self) -> None:
        if self._metrics_file:
            with open(self._metrics_file, "w") as f:
                json.dump(self._history, f, indent=2)
```

### WandbReporter (reporting/wandb_reporter.py)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot


class WandbReporter:
    """Reports metrics to Weights & Biases."""

    def __init__(self, project: str, entity: str | None = None):
        self._project = project
        self._entity = entity
        self._run = None

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

        metrics = {r.name: r.value for r in snapshot.results.values()}
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
```

---

## Training Module Additions

### BestModelTracker (training/best_model_tracker.py)

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    step: int
    epoch: float | None
    metric_name: str
    metric_value: float
    path: Path


class BestModelTracker:
    """Tracks the best model checkpoint based on a target metric."""

    def __init__(
        self,
        metric_name: str,
        higher_is_better: bool = False,
        checkpoint_dir: str | Path = "checkpoints",
    ):
        self._metric_name = metric_name
        self._higher_is_better = higher_is_better
        self._checkpoint_dir = Path(checkpoint_dir)
        self._best: CheckpointInfo | None = None
        self._all_checkpoints: list[CheckpointInfo] = []

    @property
    def best_checkpoint(self) -> CheckpointInfo | None:
        return self._best

    @property
    def best_metric_value(self) -> float | None:
        return self._best.metric_value if self._best else None

    def update(self, snapshot: MetricsSnapshot, checkpoint_path: Path) -> bool:
        """
        Update tracker with new metrics. Returns True if this is the new best.
        """
        result = snapshot.get(self._metric_name)
        if result is None:
            return False

        info = CheckpointInfo(
            step=snapshot.step,
            epoch=snapshot.epoch,
            metric_name=self._metric_name,
            metric_value=result.value,
            path=checkpoint_path,
        )
        self._all_checkpoints.append(info)

        is_better = self._is_better(result.value)
        if is_better:
            self._best = info

        return is_better

    def _is_better(self, value: float) -> bool:
        if self._best is None:
            return True

        if self._higher_is_better:
            return value > self._best.metric_value
        return value < self._best.metric_value

    def get_best_checkpoint_path(self) -> Path | None:
        """Return path to best checkpoint for loading."""
        return self._best.path if self._best else None
```

### MetricsCollector (training/metrics_collector.py)

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from rules_lawyer_models.evaluation.metric_protocol import MetricProtocol
    from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot
    from rules_lawyer_models.reporting.reporting_dispatcher import ReportingDispatcher
    from rules_lawyer_models.training.best_model_tracker import BestModelTracker

from rules_lawyer_models.evaluation.metric_result import MetricsSnapshot as MetricsSnapshotClass


class MetricsCollector:
    """
    Central hub for metric collection, aggregation, and reporting.

    Coordinates between:
    - Metric observers (compute metrics)
    - Reporting dispatcher (output metrics)
    - Best model tracker (checkpoint selection)
    """

    def __init__(
        self,
        metrics: list[MetricProtocol],
        dispatcher: ReportingDispatcher,
        best_model_tracker: BestModelTracker | None = None,
    ):
        self._metrics = metrics
        self._dispatcher = dispatcher
        self._tracker = best_model_tracker
        self._history: list[MetricsSnapshot] = []

    def initialize(self, run_name: str, config: dict[str, Any]) -> None:
        """Initialize collectors and reporters for a new run."""
        self._dispatcher.initialize(run_name, config)
        self._history.clear()
        for metric in self._metrics:
            metric.reset()

    def collect(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_data: list[dict],
        step: int,
        epoch: float | None = None,
        predictions: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MetricsSnapshot:
        """
        Collect all metrics and dispatch to reporters.
        Returns the snapshot for further processing.
        """
        snapshot = MetricsSnapshotClass(step=step, epoch=epoch)

        for metric in self._metrics:
            result = metric.compute(
                model=model,
                tokenizer=tokenizer,
                eval_data=eval_data,
                predictions=predictions,
                labels=labels,
                step=step,
                epoch=epoch,
            )
            snapshot.add(result)

        self._history.append(snapshot)
        self._dispatcher.log_metrics(snapshot)

        return snapshot

    def update_best_model(self, snapshot: MetricsSnapshot, checkpoint_path: str) -> bool:
        """
        Update best model tracker. Returns True if this is the new best.
        """
        if self._tracker is None:
            return False

        from pathlib import Path
        return self._tracker.update(snapshot, Path(checkpoint_path))

    def get_best_checkpoint_path(self) -> str | None:
        """Get path to the best checkpoint for loading."""
        if self._tracker is None:
            return None
        path = self._tracker.get_best_checkpoint_path()
        return str(path) if path else None

    def finalize(self) -> None:
        """Finalize all reporters."""
        self._dispatcher.finalize()

    @property
    def history(self) -> list[MetricsSnapshot]:
        return self._history.copy()
```

---

## Updated Training Pipeline Integration

### MetricsConfig (training/metrics_config.py)

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and reporting."""

    # Metrics to compute
    metrics: list[str] = field(default_factory=lambda: ["loss"])

    # Best model tracking
    track_best_model: bool = True
    best_model_metric: str = "eval_loss"
    higher_is_better: bool = False

    # Reporting
    report_to_console: bool = True
    report_to_disk: bool = True
    report_to_wandb: bool = False

    # W&B settings
    wandb_project: str = "rules-lawyer"
    wandb_entity: str | None = None

    # Evaluation frequency
    eval_steps: int = 100
```

### Updated run_pipeline (training/training_pipeline.py)

```python
def run_pipeline(
    run_configuration: RunConfiguration,
    factory_settings: SettingsForTrainingOptionsFactory,
    ctxt: RunContext,
    metrics_config: MetricsConfig | None = None,
) -> None:
    # ... existing model setup code ...

    # Setup metrics collection
    if metrics_config is None:
        metrics_config = MetricsConfig()

    metrics_collector = create_metrics_collector(
        metrics_config=metrics_config,
        output_dir=run_configuration.output_dir,
        logger=ctxt.logger,
    )

    run_name = f"{ctxt.model_name.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config_dict = {
        **run_configuration.to_dict(),
        **training_options.to_dict(),
    }
    metrics_collector.initialize(run_name, config_dict)

    # Custom callback for metrics collection
    class MetricsCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            snapshot = metrics_collector.collect(
                model=model,
                tokenizer=tokenizer,
                eval_data=eval_dataset,
                step=state.global_step,
                epoch=state.epoch,
            )

            # Save checkpoint and update best model tracker
            checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
            is_best = metrics_collector.update_best_model(snapshot, checkpoint_path)

            if is_best:
                ctxt.logger.report_message(
                    f"New best model at step {state.global_step}"
                )

    trainer = SFTTrainer(
        # ... existing args ...
        callbacks=[MetricsCallback()],
    )

    _ = trainer.train()

    # Load best model at end of training
    best_path = metrics_collector.get_best_checkpoint_path()
    if best_path:
        ctxt.logger.report_message(f"Loading best model from {best_path}")
        # Load best checkpoint logic here

    metrics_collector.finalize()
```

### Factory Function (training/metrics_factory.py)

```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rules_lawyer_models.utils import LoggingProtocol

from rules_lawyer_models.evaluation.metrics_registry import MetricsRegistry
from rules_lawyer_models.reporting.console_reporter import ConsoleReporter
from rules_lawyer_models.reporting.disk_reporter import DiskReporter
from rules_lawyer_models.reporting.reporting_dispatcher import ReportingDispatcher
from rules_lawyer_models.reporting.wandb_reporter import WandbReporter
from rules_lawyer_models.training.best_model_tracker import BestModelTracker
from rules_lawyer_models.training.metrics_collector import MetricsCollector
from rules_lawyer_models.training.metrics_config import MetricsConfig


def create_metrics_collector(
    metrics_config: MetricsConfig,
    output_dir: str | Path,
    logger: LoggingProtocol,
) -> MetricsCollector:
    """Factory function to create a fully configured MetricsCollector."""

    # Create metrics
    metrics = MetricsRegistry.create_all(metrics_config.metrics)

    # Create reporters
    reporters = []
    if metrics_config.report_to_console:
        reporters.append(ConsoleReporter(logger))
    if metrics_config.report_to_disk:
        reporters.append(DiskReporter(output_dir))
    if metrics_config.report_to_wandb:
        reporters.append(WandbReporter(
            project=metrics_config.wandb_project,
            entity=metrics_config.wandb_entity,
        ))

    dispatcher = ReportingDispatcher(reporters)

    # Create best model tracker
    tracker = None
    if metrics_config.track_best_model:
        tracker = BestModelTracker(
            metric_name=metrics_config.best_model_metric,
            higher_is_better=metrics_config.higher_is_better,
            checkpoint_dir=Path(output_dir) / "checkpoints",
        )

    return MetricsCollector(
        metrics=metrics,
        dispatcher=dispatcher,
        best_model_tracker=tracker,
    )
```

---

## Sequence Diagram: Metric Collection Flow

```
┌─────────┐  ┌──────────────────┐  ┌────────────────┐  ┌────────────────────┐  ┌──────────────┐
│ Trainer │  │ MetricsCollector │  │ MetricObserver │  │ReportingDispatcher │  │BestModelTracker│
└────┬────┘  └────────┬─────────┘  └───────┬────────┘  └─────────┬──────────┘  └───────┬──────┘
     │                │                    │                     │                      │
     │  on_evaluate() │                    │                     │                      │
     │───────────────▶│                    │                     │                      │
     │                │                    │                     │                      │
     │                │  compute()         │                     │                      │
     │                │───────────────────▶│                     │                      │
     │                │                    │                     │                      │
     │                │  MetricResult      │                     │                      │
     │                │◀───────────────────│                     │                      │
     │                │                    │                     │                      │
     │                │  log_metrics(snapshot)                   │                      │
     │                │──────────────────────────────────────────▶                      │
     │                │                    │                     │                      │
     │                │                    │                     │  (to each reporter)  │
     │                │                    │                     │                      │
     │                │  update(snapshot, checkpoint_path)       │                      │
     │                │─────────────────────────────────────────────────────────────────▶
     │                │                    │                     │                      │
     │                │  is_best: bool     │                     │                      │
     │                │◀─────────────────────────────────────────────────────────────────
     │                │                    │                     │                      │
     │   snapshot     │                    │                     │                      │
     │◀───────────────│                    │                     │                      │
     │                │                    │                     │                      │
```

---

## Implementation Order

### Phase 1: Core Abstractions
1. `evaluation/metric_result.py` - Data classes
2. `evaluation/metric_protocol.py` - Metric protocol
3. `reporting/reporter_protocol.py` - Reporter protocol

### Phase 2: Reporting Infrastructure
4. `reporting/console_reporter.py` - Console output
5. `reporting/disk_reporter.py` - Local persistence
6. `reporting/wandb_reporter.py` - W&B integration
7. `reporting/reporting_dispatcher.py` - Dispatcher

### Phase 3: Evaluation Metrics
8. `evaluation/metrics_registry.py` - Registry
9. `evaluation/loss_metric.py` - Loss metric
10. `evaluation/perplexity_metric.py` - Perplexity metric

### Phase 4: Training Integration
11. `training/best_model_tracker.py` - Checkpoint tracking
12. `training/metrics_collector.py` - Central hub
13. `training/metrics_config.py` - Configuration
14. `training/metrics_factory.py` - Factory function
15. Update `training/training_pipeline.py` - Integration

### Phase 5: Additional Metrics (as needed)
16. `evaluation/accuracy_metric.py`
17. Custom domain-specific metrics

---

## Testing Strategy

### Unit Tests
- Each metric in isolation with mock model/tokenizer
- Each reporter in isolation with mock data
- BestModelTracker state transitions
- MetricsRegistry registration and creation

### Integration Tests
- Full flow: MetricsCollector -> Metrics -> Reporters
- Disk reporter file output verification
- W&B reporter with mock wandb module

### End-to-End Tests
- Short training run with metrics collection
- Best model selection and loading

---

## Future Considerations

1. **Async Reporting**: For W&B, consider async uploads to avoid blocking training
2. **Metric Caching**: Cache expensive metric computations across evaluation calls
3. **Custom Aggregations**: Support for aggregating metrics across multiple eval batches
4. **Alerting**: Add alerting reporter for metric thresholds (Slack, email)
5. **Visualization**: Local HTML report generation with training curves
