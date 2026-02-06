from __future__ import annotations

from rules_lawyer_models.evaluation import (
    BinaryClassificationData,
    compute_accuracy,
    compute_f1,
    compute_false_negative_rate,
    compute_false_positive_rate,
    compute_precision,
    compute_recall,
    compute_specificity,
)

# Reusable fixtures

STEP = 100
EPOCH = 1.5


def _make_data(tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0) -> BinaryClassificationData:
    return BinaryClassificationData(true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn)


# --- Tests for compute_precision ---


class TestComputePrecision:
    def test_perfect_precision(self) -> None:
        result = compute_precision(_make_data(tp=10, fp=0), STEP, EPOCH)
        assert result.name == "precision"
        assert result.value == 100.0
        assert result.step == STEP
        assert result.epoch == EPOCH

    def test_fifty_percent(self) -> None:
        result = compute_precision(_make_data(tp=5, fp=5), STEP, EPOCH)
        assert result.value == 50.0

    def test_zero_with_only_false_positives(self) -> None:
        result = compute_precision(_make_data(tp=0, fp=10), STEP, EPOCH)
        assert result.value == 0.0

    def test_no_positive_predictions_returns_zero(self) -> None:
        result = compute_precision(_make_data(tp=0, fp=0, tn=10, fn=5), STEP, EPOCH)
        assert result.value == 0.0

    def test_metadata_includes_counts(self) -> None:
        result = compute_precision(_make_data(tp=3, fp=2, tn=4, fn=1), STEP, EPOCH)
        assert result.metadata["true_positives"] == 3
        assert result.metadata["false_positives"] == 2
        assert result.metadata["true_negatives"] == 4
        assert result.metadata["false_negatives"] == 1
        assert result.metadata["total"] == 10


# --- Tests for compute_recall ---


class TestComputeRecall:
    def test_perfect_recall(self) -> None:
        result = compute_recall(_make_data(tp=10, fn=0), STEP, EPOCH)
        assert result.name == "recall"
        assert result.value == 100.0

    def test_fifty_percent(self) -> None:
        result = compute_recall(_make_data(tp=5, fn=5), STEP, EPOCH)
        assert result.value == 50.0

    def test_zero_with_only_false_negatives(self) -> None:
        result = compute_recall(_make_data(tp=0, fn=10), STEP, EPOCH)
        assert result.value == 0.0

    def test_no_actual_positives_returns_zero(self) -> None:
        result = compute_recall(_make_data(tp=0, fp=5, tn=10, fn=0), STEP, EPOCH)
        assert result.value == 0.0


# --- Tests for compute_f1 ---


class TestComputeF1:
    def test_perfect_f1(self) -> None:
        result = compute_f1(_make_data(tp=10, fp=0, fn=0), STEP, EPOCH)
        assert result.name == "f1"
        assert result.value == 100.0

    def test_balanced_errors(self) -> None:
        # TP=5, FP=5, FN=5 -> precision=50%, recall=50% -> F1=50%
        result = compute_f1(_make_data(tp=5, fp=5, fn=5), STEP, EPOCH)
        assert abs(result.value - 50.0) < 0.01

    def test_high_precision_low_recall(self) -> None:
        # TP=1, FP=0, FN=9 -> precision=100%, recall=10% -> F1=18.18%
        result = compute_f1(_make_data(tp=1, fp=0, fn=9), STEP, EPOCH)
        assert abs(result.value - 18.18) < 0.1

    def test_no_true_positives_returns_zero(self) -> None:
        result = compute_f1(_make_data(tp=0, fp=5, fn=5), STEP, EPOCH)
        assert result.value == 0.0

    def test_no_predictions_or_positives_returns_zero(self) -> None:
        result = compute_f1(_make_data(tp=0, fp=0, tn=10, fn=0), STEP, EPOCH)
        assert result.value == 0.0


# --- Tests for compute_accuracy ---


class TestComputeAccuracy:
    def test_perfect_accuracy(self) -> None:
        result = compute_accuracy(_make_data(tp=5, tn=5), STEP, EPOCH)
        assert result.name == "accuracy"
        assert result.value == 100.0

    def test_fifty_percent(self) -> None:
        result = compute_accuracy(_make_data(tp=3, fp=3, tn=2, fn=2), STEP, EPOCH)
        assert result.value == 50.0

    def test_zero_accuracy(self) -> None:
        result = compute_accuracy(_make_data(tp=0, fp=5, tn=0, fn=5), STEP, EPOCH)
        assert result.value == 0.0

    def test_empty_data_returns_zero(self) -> None:
        result = compute_accuracy(_make_data(), STEP, EPOCH)
        assert result.value == 0.0


# --- Tests for compute_false_positive_rate ---


class TestComputeFalsePositiveRate:
    def test_no_false_positives(self) -> None:
        result = compute_false_positive_rate(_make_data(fp=0, tn=10), STEP, EPOCH)
        assert result.name == "false_positive_rate"
        assert result.value == 0.0

    def test_all_false_positives(self) -> None:
        result = compute_false_positive_rate(_make_data(fp=10, tn=0), STEP, EPOCH)
        assert result.value == 100.0

    def test_fifty_percent(self) -> None:
        result = compute_false_positive_rate(_make_data(fp=5, tn=5), STEP, EPOCH)
        assert result.value == 50.0

    def test_no_actual_negatives_returns_zero(self) -> None:
        result = compute_false_positive_rate(_make_data(tp=10, fp=0, tn=0, fn=5), STEP, EPOCH)
        assert result.value == 0.0


# --- Tests for compute_false_negative_rate ---


class TestComputeFalseNegativeRate:
    def test_no_false_negatives(self) -> None:
        result = compute_false_negative_rate(_make_data(tp=10, fn=0), STEP, EPOCH)
        assert result.name == "false_negative_rate"
        assert result.value == 0.0

    def test_all_false_negatives(self) -> None:
        result = compute_false_negative_rate(_make_data(tp=0, fn=10), STEP, EPOCH)
        assert result.value == 100.0

    def test_fifty_percent(self) -> None:
        result = compute_false_negative_rate(_make_data(tp=5, fn=5), STEP, EPOCH)
        assert result.value == 50.0

    def test_no_actual_positives_returns_zero(self) -> None:
        result = compute_false_negative_rate(_make_data(tp=0, fp=5, tn=10, fn=0), STEP, EPOCH)
        assert result.value == 0.0


# --- Tests for compute_specificity ---


class TestComputeSpecificity:
    def test_perfect_specificity(self) -> None:
        result = compute_specificity(_make_data(tn=10, fp=0), STEP, EPOCH)
        assert result.name == "specificity"
        assert result.value == 100.0

    def test_zero_specificity(self) -> None:
        result = compute_specificity(_make_data(tn=0, fp=10), STEP, EPOCH)
        assert result.value == 0.0

    def test_fifty_percent(self) -> None:
        result = compute_specificity(_make_data(tn=5, fp=5), STEP, EPOCH)
        assert result.value == 50.0

    def test_no_actual_negatives_returns_zero(self) -> None:
        result = compute_specificity(_make_data(tp=10, fp=0, tn=0, fn=5), STEP, EPOCH)
        assert result.value == 0.0


# --- Cross-metric consistency tests ---


class TestCrossMetricConsistency:
    """Verify mathematical relationships between metrics hold."""

    def test_specificity_is_complement_of_fpr(self) -> None:
        data = _make_data(tp=8, fp=3, tn=7, fn=2)
        fpr = compute_false_positive_rate(data, STEP, EPOCH)
        spec = compute_specificity(data, STEP, EPOCH)
        assert abs(fpr.value + spec.value - 100.0) < 0.01

    def test_recall_is_complement_of_fnr(self) -> None:
        data = _make_data(tp=8, fp=3, tn=7, fn=2)
        recall = compute_recall(data, STEP, EPOCH)
        fnr = compute_false_negative_rate(data, STEP, EPOCH)
        assert abs(recall.value + fnr.value - 100.0) < 0.01

    def test_f1_between_precision_and_recall(self) -> None:
        """F1 (harmonic mean) should be <= arithmetic mean of precision and recall."""
        data = _make_data(tp=6, fp=4, tn=5, fn=3)
        p = compute_precision(data, STEP, EPOCH).value
        r = compute_recall(data, STEP, EPOCH).value
        f1 = compute_f1(data, STEP, EPOCH).value
        arithmetic_mean = (p + r) / 2
        assert f1 <= arithmetic_mean + 0.01

    def test_epoch_none_propagates(self) -> None:
        data = _make_data(tp=5, fp=2, tn=3, fn=1)
        for fn in [
            compute_precision,
            compute_recall,
            compute_f1,
            compute_accuracy,
            compute_false_positive_rate,
            compute_false_negative_rate,
            compute_specificity,
        ]:
            result = fn(data, STEP, None)
            assert result.epoch is None
