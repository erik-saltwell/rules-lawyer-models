from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from datasets import Dataset

from .binary_classification import (
    BinaryClassificationResult,
    compute_classification_metric,
)
from .metric_result import MetricResult


@dataclass(frozen=True)
class BinaryClassificationMetric:
    """A MetricProtocol implementation backed by compute_classification_metric."""

    inputs_column_name: str
    ground_truth_column_name: str
    predictions_column_name: str
    positive_class: str
    clean_prediction: Callable[[str], str]
    aggregate_predictions: Callable[[dict[BinaryClassificationResult, int]], MetricResult]

    def compute_metric(self, dataset: Dataset) -> MetricResult:
        return compute_classification_metric(
            dataset,
            self.inputs_column_name,
            self.ground_truth_column_name,
            self.predictions_column_name,
            self.positive_class,
            self.clean_prediction,
            self.aggregate_predictions,
        )
