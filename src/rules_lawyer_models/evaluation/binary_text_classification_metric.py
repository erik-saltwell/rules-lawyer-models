from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, TypeAlias

import torch

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rules_lawyer_models.evaluation.metric_result import MetricResult


class BinaryClassificationResult(IntEnum):
    TRUE_POSITIVE = auto()
    FALSE_POSITIVE = auto()
    TRUE_NEGATIVE = auto()
    FALSE_NEGATIVE = auto()


@dataclass
class BinaryClassificationData:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def add_result(self, result: BinaryClassificationResult) -> None:
        if result == BinaryClassificationResult.TRUE_POSITIVE:
            self.true_positives += 1
        elif result == BinaryClassificationResult.FALSE_POSITIVE:
            self.false_positives += 1
        elif result == BinaryClassificationResult.TRUE_NEGATIVE:
            self.true_negatives += 1
        elif result == BinaryClassificationResult.FALSE_NEGATIVE:
            self.false_negatives += 1

    @property
    def total(self) -> int:
        return self.true_positives + self.false_positives + self.true_negatives + self.false_negatives


@dataclass(frozen=True)
class UnclassifiedTextData:
    predicted_text: str
    actual_text: str


BinaryClassifier: TypeAlias = Callable[[UnclassifiedTextData], BinaryClassificationResult]
BinaryClassificationAggregator: TypeAlias = Callable[[BinaryClassificationData, int, float | None], MetricResult]


@dataclass
class BinaryTextClassificationMetric:
    """Sample-level text classification metric.

    Compares decoded predicted text against decoded label text for each sample,
    classifies the result (TP/FP/TN/FN), and aggregates into a final metric.
    """

    metric_name: str
    classifier: BinaryClassifier
    aggregator: BinaryClassificationAggregator
    higher_better: bool = False
    ignore_index: int = -100

    @property
    def name(self) -> str:
        return self.metric_name

    @property
    def higher_is_better(self) -> bool:
        return self.higher_better

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
        """Compute the metric by comparing decoded text for each sample."""
        model.eval()
        classification_data: BinaryClassificationData = BinaryClassificationData()

        with torch.no_grad():
            for batch in eval_data:
                outputs = model(**batch)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                batch_labels = batch.get("labels")
                if batch_labels is None:
                    continue

                # Get predicted token IDs (argmax of logits)
                predicted_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

                batch_size = predicted_ids.shape[0]

                # Process each sample in the batch
                for sample_idx in range(batch_size):
                    sample_preds = predicted_ids[sample_idx]  # [seq_len]
                    sample_labels = batch_labels[sample_idx]  # [seq_len]

                    # Create mask for valid (non-ignored) positions
                    valid_mask = sample_labels != self.ignore_index

                    # Extract only valid token IDs
                    valid_pred_ids = sample_preds[valid_mask]
                    valid_label_ids = sample_labels[valid_mask]

                    # Decode to text (skip special tokens)
                    predicted_text = tokenizer.decode(valid_pred_ids, skip_special_tokens=True).strip()
                    actual_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True).strip()

                    # Classify this sample
                    unclassified = UnclassifiedTextData(
                        predicted_text=predicted_text,
                        actual_text=actual_text,
                    )
                    result = self.classifier(unclassified)
                    classification_data.add_result(result)

        # Aggregate results into final metric
        return self.aggregator(classification_data, step, epoch)

    def reset(self) -> None:
        pass
