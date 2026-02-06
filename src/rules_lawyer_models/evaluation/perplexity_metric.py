from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rules_lawyer_models.evaluation.metric_result import MetricResult
from rules_lawyer_models.evaluation.metrics_registry import MetricsRegistry


@MetricsRegistry.register("perplexity")
class PerplexityMetric:
    """Computes perplexity on evaluation data.

    Perplexity = exp(average cross-entropy loss)
    Lower perplexity indicates better model performance.
    """

    def __init__(self) -> None:
        self._total_loss: float = 0.0
        self._total_tokens: int = 0

    @property
    def name(self) -> str:
        return "eval_perplexity"

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
        total_tokens = 0

        with torch.no_grad():
            for batch in eval_data:
                outputs = model(**batch)
                loss = outputs.loss

                # Count non-padding tokens if attention_mask is available
                if "attention_mask" in batch:
                    num_tokens = batch["attention_mask"].sum().item()
                else:
                    # Fallback: count all tokens in labels
                    if "labels" in batch:
                        num_tokens = (batch["labels"] != -100).sum().item()
                    else:
                        num_tokens = batch["input_ids"].numel()

                # Accumulate weighted loss
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute average loss and perplexity
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)

        return MetricResult(
            name=self.name,
            value=perplexity,
            step=step,
            epoch=epoch,
            metadata={"avg_loss": avg_loss, "total_tokens": total_tokens},
        )

    def reset(self) -> None:
        self._total_loss = 0.0
        self._total_tokens = 0
