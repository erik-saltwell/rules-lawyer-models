from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rules_lawyer_models.evaluation.metric_result import MetricResult
from rules_lawyer_models.evaluation.metrics_registry import MetricsRegistry


@MetricsRegistry.register("false_positive_rate")
class FalsePositiveRateMetric:
    """Computes false positive rate (FPR) on evaluation data.

    FPR = FP / (FP + TN) = FP / Actual Negatives

    For binary classification where positive_class_id is the "positive" label.
    A false positive occurs when we predict positive_class_id but the true label
    is something else.
    """

    def __init__(self, positive_class_id: int = 1, ignore_index: int = -100) -> None:
        """
        Args:
            positive_class_id: The class ID considered "positive" for FP calculation.
            ignore_index: Label value to ignore (typically -100 for masked tokens).
        """
        self._positive_class_id = positive_class_id
        self._ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "false_positive_rate"

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

        total_false_positives = 0
        total_actual_negatives = 0

        with torch.no_grad():
            for batch in eval_data:
                outputs = model(**batch)
                logits = outputs.logits  # Shape: (batch, seq_len, vocab_size) or (batch, num_classes)

                batch_labels = batch.get("labels")
                if batch_labels is None:
                    continue

                # Get predicted classes (argmax of logits)
                predicted = torch.argmax(logits, dim=-1)

                # Create mask for valid (non-ignored) positions
                valid_mask = batch_labels != self._ignore_index

                # Get valid predictions and labels
                valid_preds = predicted[valid_mask]
                valid_labels = batch_labels[valid_mask]

                # Actual negatives: where true label is NOT the positive class
                actual_negatives_mask = valid_labels != self._positive_class_id

                # False positives: predicted positive but actual is negative
                false_positives = ((valid_preds == self._positive_class_id) & actual_negatives_mask).sum().item()

                total_false_positives += false_positives
                total_actual_negatives += actual_negatives_mask.sum().item()

        # FPR = FP / (FP + TN) = FP / Actual Negatives
        if total_actual_negatives > 0:
            fpr = (total_false_positives / total_actual_negatives) * 100.0
        else:
            fpr = 0.0

        return MetricResult(
            name=self.name,
            value=fpr,
            step=step,
            epoch=epoch,
            metadata={
                "false_positives": total_false_positives,
                "actual_negatives": total_actual_negatives,
                "positive_class_id": self._positive_class_id,
            },
        )

    def reset(self) -> None:
        pass
