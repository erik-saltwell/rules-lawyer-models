from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rules_lawyer_models.evaluation import (
    BinaryClassificationResult,
    accuracy,
    compute_classification_metric,
    f1_score,
    false_positive_rate,
    matthews_correlation_coefficient,
    precision,
    recall,
    specificity,
)

TP = BinaryClassificationResult.TRUE_POSITIVE
FP = BinaryClassificationResult.FALSE_POSITIVE
TN = BinaryClassificationResult.TRUE_NEGATIVE
FN = BinaryClassificationResult.FALSE_NEGATIVE


def _make_counts(tp: int, fp: int, tn: int, fn: int) -> dict[BinaryClassificationResult, int]:
    return {TP: tp, FP: fp, TN: tn, FN: fn}


# ── fixtures ─────────────────────────────────────────────────────

STANDARD = _make_counts(tp=50, fp=10, tn=80, fn=20)
PERFECT = _make_counts(tp=50, fp=0, tn=50, fn=0)
ALL_WRONG = _make_counts(tp=0, fp=50, tn=0, fn=50)
EMPTY = _make_counts(tp=0, fp=0, tn=0, fn=0)


# ── accuracy ─────────────────────────────────────────────────────


class TestAccuracy:
    def test_standard(self) -> None:
        result = accuracy(STANDARD)
        assert result.metric_name == "accuracy"
        assert result.metric_result == pytest.approx(130 / 160)

    def test_perfect(self) -> None:
        assert accuracy(PERFECT).metric_result == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        assert accuracy(ALL_WRONG).metric_result == pytest.approx(0.0)

    def test_empty(self) -> None:
        assert accuracy(EMPTY).metric_result == 0.0


# ── precision ────────────────────────────────────────────────────


class TestPrecision:
    def test_standard(self) -> None:
        result = precision(STANDARD)
        assert result.metric_name == "precision"
        assert result.metric_result == pytest.approx(50 / 60)

    def test_perfect(self) -> None:
        assert precision(PERFECT).metric_result == pytest.approx(1.0)

    def test_no_positive_predictions(self) -> None:
        c = _make_counts(tp=0, fp=0, tn=80, fn=20)
        assert precision(c).metric_result == 0.0


# ── recall ───────────────────────────────────────────────────────


class TestRecall:
    def test_standard(self) -> None:
        result = recall(STANDARD)
        assert result.metric_name == "recall"
        assert result.metric_result == pytest.approx(50 / 70)

    def test_perfect(self) -> None:
        assert recall(PERFECT).metric_result == pytest.approx(1.0)

    def test_no_actual_positives(self) -> None:
        c = _make_counts(tp=0, fp=10, tn=80, fn=0)
        assert recall(c).metric_result == 0.0


# ── f1 ───────────────────────────────────────────────────────────


class TestF1:
    def test_standard(self) -> None:
        p = 50 / 60
        r = 50 / 70
        expected = (2 * p * r) / (p + r)
        result = f1_score(STANDARD)
        assert result.metric_name == "f1"
        assert result.metric_result == pytest.approx(expected)

    def test_perfect(self) -> None:
        assert f1_score(PERFECT).metric_result == pytest.approx(1.0)

    def test_zero_precision_and_recall(self) -> None:
        c = _make_counts(tp=0, fp=0, tn=80, fn=0)
        assert f1_score(c).metric_result == 0.0


# ── specificity ──────────────────────────────────────────────────


class TestSpecificity:
    def test_standard(self) -> None:
        result = specificity(STANDARD)
        assert result.metric_name == "specificity"
        assert result.metric_result == pytest.approx(80 / 90)

    def test_no_actual_negatives(self) -> None:
        c = _make_counts(tp=50, fp=0, tn=0, fn=20)
        assert specificity(c).metric_result == 0.0


# ── false positive rate ──────────────────────────────────────────


class TestFalsePositiveRate:
    def test_standard(self) -> None:
        result = false_positive_rate(STANDARD)
        assert result.metric_name == "false_positive_rate"
        assert result.metric_result == pytest.approx(10 / 90)

    def test_complement_of_specificity(self) -> None:
        fpr = false_positive_rate(STANDARD).metric_result
        spec = specificity(STANDARD).metric_result
        assert fpr + spec == pytest.approx(1.0)


# ── MCC ──────────────────────────────────────────────────────────


class TestMCC:
    def test_standard(self) -> None:
        numer = 50 * 80 - 10 * 20
        denom_sq = 60 * 70 * 90 * 100
        expected = numer / (denom_sq**0.5)
        result = matthews_correlation_coefficient(STANDARD)
        assert result.metric_name == "mcc"
        assert result.metric_result == pytest.approx(expected)

    def test_perfect(self) -> None:
        assert matthews_correlation_coefficient(PERFECT).metric_result == pytest.approx(1.0)

    def test_empty(self) -> None:
        assert matthews_correlation_coefficient(EMPTY).metric_result == 0.0


# ── compute_classification_metric (end-to-end) ──────────────────


class TestComputeClassificationMetric:
    @patch("rules_lawyer_models.evaluation.binary_classification._collect_classifications")
    def test_delegates_to_collect_and_aggregate(self, mock_collect: MagicMock) -> None:
        """Verify compute_classification_metric wires _collect_classifications to the aggregators."""
        mock_collect.return_value = STANDARD
        dataset = MagicMock()

        results = compute_classification_metric(
            dataset,
            "input",
            "ground_truth",
            "predictions",
            "positive",
            [accuracy],
        )

        mock_collect.assert_called_once_with(dataset, "input", "ground_truth", "predictions", "positive")
        assert len(results) == 1
        assert results[0].metric_name == "accuracy"
        assert results[0].metric_result == pytest.approx(130 / 160)

    @patch("rules_lawyer_models.evaluation.binary_classification._collect_classifications")
    def test_with_multiple_aggregators(self, mock_collect: MagicMock) -> None:
        counts = _make_counts(tp=40, fp=10, tn=30, fn=20)
        mock_collect.return_value = counts

        results = compute_classification_metric(
            MagicMock(),
            "input",
            "ground_truth",
            "predictions",
            "positive",
            [accuracy, f1_score],
        )

        assert len(results) == 2
        assert results[0].metric_name == "accuracy"
        assert results[0].metric_result == pytest.approx(70 / 100)

        p = 40 / 50
        r = 40 / 60
        assert results[1].metric_name == "f1"
        assert results[1].metric_result == pytest.approx((2 * p * r) / (p + r))
