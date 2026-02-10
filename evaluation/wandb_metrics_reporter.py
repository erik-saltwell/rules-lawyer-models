from __future__ import annotations

from collections.abc import Iterable

import wandb
from evaluation.metric_result import MetricResult


class WandbMetricsReporter:
    """Reports metric results to Weights & Biases."""

    def Report(self, results: Iterable[MetricResult]) -> None:
        wandb.log({r.metric_name: r.metric_result for r in results})
