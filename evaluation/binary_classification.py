from collections.abc import Callable
from enum import IntEnum, auto
from typing import Any, cast

from datasets import Dataset

from evaluation.metric_result import MetricResult


class BinaryClassificationResult(IntEnum):
    TRUE_POSITIVE = auto()
    FALSE_POSITIVE = auto()
    TRUE_NEGATIVE = auto()
    FALSE_NEGATIVE = auto()


def compute_classification_metric(
    self,
    dataset: Dataset,
    inputs_column_name: str,
    ground_truth_column_name: str,
    predictions_column_name: str,
    positive_class: str,
    clean_prediction: Callable[[str], str],
    aggregate_predictions: Callable[[dict[BinaryClassificationResult, int]], MetricResult],
) -> MetricResult:
    classifications: dict[BinaryClassificationResult, int] = self._collect_classifications(
        dataset,
        inputs_column_name,
        ground_truth_column_name,
        predictions_column_name,
        positive_class,
        clean_prediction,
    )
    result: MetricResult = aggregate_predictions(classifications)
    return result


def _collect_classifications(
    self,
    dataset: Dataset,
    inputs_column_name: str,
    ground_truth_column_name: str,
    predictions_column_name: str,
    positive_class: str,
    clean_prediction: Callable[[str], str],
) -> dict[BinaryClassificationResult, int]:
    classifications: dict[BinaryClassificationResult, int] = {
        BinaryClassificationResult.TRUE_POSITIVE: 0,
        BinaryClassificationResult.FALSE_POSITIVE: 0,
        BinaryClassificationResult.TRUE_NEGATIVE: 0,
        BinaryClassificationResult.FALSE_NEGATIVE: 0,
    }
    for r in dataset:
        row = cast(dict[str, Any], r)
        cleaned_prediction: str = clean_prediction(row[predictions_column_name])
        is_positive_class: bool = cleaned_prediction == positive_class
        is_match: bool = cleaned_prediction == predictions_column_name
        if is_positive_class:  # TRUE_POSITIVE or FALSE_POSITIVE
            if is_match:
                classifications[BinaryClassificationResult.TRUE_POSITIVE] = (
                    classifications[BinaryClassificationResult.TRUE_POSITIVE] + 1
                )
            else:
                classifications[BinaryClassificationResult.FALSE_POSITIVE] = (
                    classifications[BinaryClassificationResult.FALSE_POSITIVE] + 1
                )
        else:
            if is_match:
                classifications[BinaryClassificationResult.TRUE_NEGATIVE] = (
                    classifications[BinaryClassificationResult.TRUE_NEGATIVE] + 1
                )
            else:
                classifications[BinaryClassificationResult.FALSE_NEGATIVE] = (
                    classifications[BinaryClassificationResult.FALSE_NEGATIVE] + 1
                )
    return classifications


_C = dict[BinaryClassificationResult, int]


def _unpack(c: _C) -> tuple[int, int, int, int]:
    return (
        c[BinaryClassificationResult.TRUE_POSITIVE],
        c[BinaryClassificationResult.FALSE_POSITIVE],
        c[BinaryClassificationResult.TRUE_NEGATIVE],
        c[BinaryClassificationResult.FALSE_NEGATIVE],
    )


def accuracy(c: _C) -> MetricResult:
    tp, fp, tn, fn = _unpack(c)
    total = tp + fp + tn + fn
    return MetricResult("accuracy", (tp + tn) / total if total else 0.0)


def precision(c: _C) -> MetricResult:
    tp, fp, _tn, _fn = _unpack(c)
    denom = tp + fp
    return MetricResult("precision", tp / denom if denom else 0.0)


def recall(c: _C) -> MetricResult:
    tp, _fp, _tn, fn = _unpack(c)
    denom = tp + fn
    return MetricResult("recall", tp / denom if denom else 0.0)


def f1_score(c: _C) -> MetricResult:
    p = precision(c).metric_result
    r = recall(c).metric_result
    denom = p + r
    return MetricResult("f1", (2 * p * r) / denom if denom else 0.0)


def specificity(c: _C) -> MetricResult:
    _tp, fp, tn, _fn = _unpack(c)
    denom = tn + fp
    return MetricResult("specificity", tn / denom if denom else 0.0)


def false_positive_rate(c: _C) -> MetricResult:
    _tp, fp, tn, _fn = _unpack(c)
    denom = fp + tn
    return MetricResult("false_positive_rate", fp / denom if denom else 0.0)


def matthews_correlation_coefficient(c: _C) -> MetricResult:
    tp, fp, tn, fn = _unpack(c)
    numer = tp * tn - fp * fn
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return MetricResult("mcc", numer / (denom_sq**0.5) if denom_sq else 0.0)
