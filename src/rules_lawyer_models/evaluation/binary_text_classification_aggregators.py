"""
Various BinaryClassificationAggregator functions that can be used to compute classification metrics like
F1, Precision and Recall.  Used with BinaryTextClassificationMetric.  These functions all have the signature:

def compute_metric(classification_count: BinaryClassificationData, steps: int, epoch: float)->MetricResult
"""

from __future__ import annotations

from rules_lawyer_models.evaluation.binary_text_classification_metric import BinaryClassificationData
from rules_lawyer_models.evaluation.metric_result import MetricResult


def _build_metadata(data: BinaryClassificationData) -> dict:
    return {
        "true_positives": data.true_positives,
        "false_positives": data.false_positives,
        "true_negatives": data.true_negatives,
        "false_negatives": data.false_negatives,
        "total": data.total,
    }


def compute_precision(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """Precision = TP / (TP + FP). Of all positive predictions, how many were correct."""
    denominator = data.true_positives + data.false_positives
    value = (data.true_positives / denominator * 100.0) if denominator > 0 else 0.0
    return MetricResult(name="precision", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))


def compute_recall(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """Recall = TP / (TP + FN). Of all actual positives, how many did we find."""
    denominator = data.true_positives + data.false_negatives
    value = (data.true_positives / denominator * 100.0) if denominator > 0 else 0.0
    return MetricResult(name="recall", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))


def compute_f1(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """F1 = 2 * (Precision * Recall) / (Precision + Recall). Harmonic mean of precision and recall."""
    tp = data.true_positives
    fp = data.false_positives
    fn = data.false_negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    value = (2.0 * precision * recall / (precision + recall) * 100.0) if (precision + recall) > 0 else 0.0
    return MetricResult(name="f1", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))


def compute_accuracy(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """Accuracy = (TP + TN) / Total. Overall rate of correct predictions."""
    total = data.total
    value = ((data.true_positives + data.true_negatives) / total * 100.0) if total > 0 else 0.0
    return MetricResult(name="accuracy", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))


def compute_false_positive_rate(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """FPR = FP / (FP + TN). Of all actual negatives, how many did we incorrectly predict as positive."""
    denominator = data.false_positives + data.true_negatives
    value = (data.false_positives / denominator * 100.0) if denominator > 0 else 0.0
    return MetricResult(name="false_positive_rate", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))


def compute_false_negative_rate(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """FNR = FN / (FN + TP). Of all actual positives, how many did we miss."""
    denominator = data.false_negatives + data.true_positives
    value = (data.false_negatives / denominator * 100.0) if denominator > 0 else 0.0
    return MetricResult(name="false_negative_rate", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))


def compute_specificity(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """Specificity = TN / (TN + FP). Of all actual negatives, how many did we correctly identify."""
    denominator = data.true_negatives + data.false_positives
    value = (data.true_negatives / denominator * 100.0) if denominator > 0 else 0.0
    return MetricResult(name="specificity", value=value, step=step, epoch=epoch, metadata=_build_metadata(data))
