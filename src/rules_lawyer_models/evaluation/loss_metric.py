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

    def __init__(self) -> None:
        self._accumulated_loss: float = 0.0
        self._num_batches: int = 0

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
        step: int,
        epoch: float | None = None,
        predictions: Tensor | None = None,
        labels: Tensor | None = None,
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
        self._num_batches = 0
