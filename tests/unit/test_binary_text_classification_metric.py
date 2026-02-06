from __future__ import annotations

from unittest.mock import MagicMock

import torch

from rules_lawyer_models.evaluation import (
    BinaryClassificationData,
    BinaryClassificationResult,
    BinaryTextClassificationMetric,
    MetricResult,
    UnclassifiedTextData,
)

# --- Helper functions for tests ---


def precision_classifier(data: UnclassifiedTextData) -> BinaryClassificationResult:
    """Classifier for question detection: 'question' is the positive class."""
    predicted_positive = data.predicted_text.lower() == "question"
    actual_positive = data.actual_text.lower() == "question"

    if predicted_positive and actual_positive:
        return BinaryClassificationResult.TRUE_POSITIVE
    elif predicted_positive and not actual_positive:
        return BinaryClassificationResult.FALSE_POSITIVE
    elif not predicted_positive and not actual_positive:
        return BinaryClassificationResult.TRUE_NEGATIVE
    else:
        return BinaryClassificationResult.FALSE_NEGATIVE


def precision_aggregator(data: BinaryClassificationData, step: int, epoch: float | None) -> MetricResult:
    """Aggregator that computes precision as a percentage."""
    denominator = data.true_positives + data.false_positives
    precision = (data.true_positives / denominator * 100.0) if denominator > 0 else 0.0
    return MetricResult(
        name="precision",
        value=precision,
        step=step,
        epoch=epoch,
        metadata={
            "true_positives": data.true_positives,
            "false_positives": data.false_positives,
            "true_negatives": data.true_negatives,
            "false_negatives": data.false_negatives,
        },
    )


# --- Tests for BinaryClassificationData ---


class TestBinaryClassificationData:
    def test_initial_values_are_zero(self) -> None:
        data = BinaryClassificationData()
        assert data.true_positives == 0
        assert data.false_positives == 0
        assert data.true_negatives == 0
        assert data.false_negatives == 0
        assert data.total == 0

    def test_add_true_positive(self) -> None:
        data = BinaryClassificationData()
        data.add_result(BinaryClassificationResult.TRUE_POSITIVE)
        assert data.true_positives == 1
        assert data.total == 1

    def test_add_false_positive(self) -> None:
        data = BinaryClassificationData()
        data.add_result(BinaryClassificationResult.FALSE_POSITIVE)
        assert data.false_positives == 1
        assert data.total == 1

    def test_add_true_negative(self) -> None:
        data = BinaryClassificationData()
        data.add_result(BinaryClassificationResult.TRUE_NEGATIVE)
        assert data.true_negatives == 1
        assert data.total == 1

    def test_add_false_negative(self) -> None:
        data = BinaryClassificationData()
        data.add_result(BinaryClassificationResult.FALSE_NEGATIVE)
        assert data.false_negatives == 1
        assert data.total == 1

    def test_add_multiple_results(self) -> None:
        data = BinaryClassificationData()
        data.add_result(BinaryClassificationResult.TRUE_POSITIVE)
        data.add_result(BinaryClassificationResult.TRUE_POSITIVE)
        data.add_result(BinaryClassificationResult.FALSE_POSITIVE)
        data.add_result(BinaryClassificationResult.TRUE_NEGATIVE)
        data.add_result(BinaryClassificationResult.FALSE_NEGATIVE)

        assert data.true_positives == 2
        assert data.false_positives == 1
        assert data.true_negatives == 1
        assert data.false_negatives == 1
        assert data.total == 5


# --- Tests for precision_classifier ---


class TestPrecisionClassifier:
    def test_true_positive(self) -> None:
        data = UnclassifiedTextData(predicted_text="question", actual_text="question")
        assert precision_classifier(data) == BinaryClassificationResult.TRUE_POSITIVE

    def test_false_positive(self) -> None:
        data = UnclassifiedTextData(predicted_text="question", actual_text="other")
        assert precision_classifier(data) == BinaryClassificationResult.FALSE_POSITIVE

    def test_true_negative(self) -> None:
        data = UnclassifiedTextData(predicted_text="other", actual_text="other")
        assert precision_classifier(data) == BinaryClassificationResult.TRUE_NEGATIVE

    def test_false_negative(self) -> None:
        data = UnclassifiedTextData(predicted_text="other", actual_text="question")
        assert precision_classifier(data) == BinaryClassificationResult.FALSE_NEGATIVE

    def test_case_insensitive(self) -> None:
        data = UnclassifiedTextData(predicted_text="QUESTION", actual_text="Question")
        assert precision_classifier(data) == BinaryClassificationResult.TRUE_POSITIVE


# --- Tests for precision_aggregator ---


class TestPrecisionAggregator:
    def test_perfect_precision(self) -> None:
        data = BinaryClassificationData(true_positives=10, false_positives=0, true_negatives=5, false_negatives=2)
        result = precision_aggregator(data, step=100, epoch=1.5)

        assert result.name == "precision"
        assert result.value == 100.0
        assert result.step == 100
        assert result.epoch == 1.5

    def test_fifty_percent_precision(self) -> None:
        data = BinaryClassificationData(true_positives=5, false_positives=5, true_negatives=10, false_negatives=0)
        result = precision_aggregator(data, step=50, epoch=None)

        assert result.value == 50.0
        assert result.epoch is None

    def test_zero_precision_with_false_positives(self) -> None:
        data = BinaryClassificationData(true_positives=0, false_positives=10, true_negatives=5, false_negatives=5)
        result = precision_aggregator(data, step=10, epoch=0.5)

        assert result.value == 0.0

    def test_no_positive_predictions(self) -> None:
        """When no positive predictions made, precision is 0 (avoid division by zero)."""
        data = BinaryClassificationData(true_positives=0, false_positives=0, true_negatives=10, false_negatives=5)
        result = precision_aggregator(data, step=10, epoch=1.0)

        assert result.value == 0.0

    def test_metadata_included(self) -> None:
        data = BinaryClassificationData(true_positives=3, false_positives=2, true_negatives=4, false_negatives=1)
        result = precision_aggregator(data, step=100, epoch=2.0)

        assert result.metadata["true_positives"] == 3
        assert result.metadata["false_positives"] == 2
        assert result.metadata["true_negatives"] == 4
        assert result.metadata["false_negatives"] == 1


# --- Tests for BinaryTextClassificationMetric ---


class TestBinaryTextClassificationMetric:
    def test_name_property(self) -> None:
        metric = BinaryTextClassificationMetric(
            metric_name="precision",
            classifier=precision_classifier,
            aggregator=precision_aggregator,
            higher_better=True,
        )
        assert metric.name == "precision"

    def test_higher_is_better_property(self) -> None:
        metric = BinaryTextClassificationMetric(
            metric_name="precision",
            classifier=precision_classifier,
            aggregator=precision_aggregator,
            higher_better=True,
        )
        assert metric.higher_is_better is True

    def test_compute_with_mock_model(self) -> None:
        """Test full compute flow with mocked model and tokenizer."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        # Create mock outputs with logits
        # Batch of 2 samples, seq_len=5, vocab_size=100
        # We'll set up logits so argmax gives us specific token IDs
        mock_logits = torch.zeros(2, 5, 100)
        # Sample 0: tokens [10, 20, 30, 40, 50] predicted
        mock_logits[0, 0, 10] = 10.0
        mock_logits[0, 1, 20] = 10.0
        mock_logits[0, 2, 30] = 10.0
        mock_logits[0, 3, 40] = 10.0
        mock_logits[0, 4, 50] = 10.0
        # Sample 1: tokens [11, 21, 31, 41, 51] predicted
        mock_logits[1, 0, 11] = 10.0
        mock_logits[1, 1, 21] = 10.0
        mock_logits[1, 2, 31] = 10.0
        mock_logits[1, 3, 41] = 10.0
        mock_logits[1, 4, 51] = 10.0

        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        # Create mock tokenizer
        mock_tokenizer = MagicMock()

        # Set up decode to return specific strings based on input
        def mock_decode(token_ids, skip_special_tokens=True):
            # Convert tensor to list for comparison
            ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            if ids == [30, 40, 50]:
                return "question"
            elif ids == [31, 41, 51]:
                return "other"
            elif ids == [30, 40]:  # label for sample 0
                return "question"
            elif ids == [31, 41]:  # label for sample 1
                return "question"
            return "unknown"

        mock_tokenizer.decode = mock_decode

        # Create eval_data batch
        # Labels: first 2 positions are -100 (masked), rest are real tokens
        labels = torch.tensor(
            [
                [-100, -100, 30, 40, 50],  # Sample 0: valid tokens [30, 40, 50] -> "question"
                [-100, -100, 31, 41, 51],  # Sample 1: valid tokens [31, 41, 51] -> will decode differently
            ]
        )

        eval_data = [{"input_ids": torch.zeros(2, 5), "labels": labels}]

        # Create metric
        metric = BinaryTextClassificationMetric(
            metric_name="precision",
            classifier=precision_classifier,
            aggregator=precision_aggregator,
            higher_better=True,
        )

        # Run compute
        result = metric.compute(
            model=mock_model,
            tokenizer=mock_tokenizer,
            eval_data=eval_data,
            step=100,
            epoch=1.0,
        )

        # Verify model was put in eval mode
        mock_model.eval.assert_called_once()

        # Verify result
        assert result.name == "precision"
        assert result.step == 100
        assert result.epoch == 1.0

    def test_compute_skips_batch_without_labels(self) -> None:
        """Test that batches without labels are skipped."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        mock_logits = torch.zeros(2, 5, 100)
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()

        # Batch without labels
        eval_data = [{"input_ids": torch.zeros(2, 5)}]

        metric = BinaryTextClassificationMetric(
            metric_name="precision",
            classifier=precision_classifier,
            aggregator=precision_aggregator,
        )

        result = metric.compute(
            model=mock_model,
            tokenizer=mock_tokenizer,
            eval_data=eval_data,
            step=50,
            epoch=None,
        )

        # Should return result with 0 samples processed
        assert result.value == 0.0  # No positive predictions = 0 precision

    def test_reset(self) -> None:
        """Test that reset doesn't raise errors."""
        metric = BinaryTextClassificationMetric(
            metric_name="precision",
            classifier=precision_classifier,
            aggregator=precision_aggregator,
        )
        # Should not raise
        metric.reset()


# --- Integration-style tests ---


class TestPrecisionMetricIntegration:
    """Tests that verify the full pipeline works correctly."""

    def test_all_correct_predictions(self) -> None:
        """When all predictions are correct, precision should be 100%."""
        data = BinaryClassificationData()

        # Simulate: predicted "question" when actual was "question" (TP)
        samples = [
            UnclassifiedTextData("question", "question"),
            UnclassifiedTextData("question", "question"),
            UnclassifiedTextData("other", "other"),
            UnclassifiedTextData("other", "other"),
        ]

        for sample in samples:
            result = precision_classifier(sample)
            data.add_result(result)

        metric_result = precision_aggregator(data, step=0, epoch=None)

        # 2 TP, 0 FP -> precision = 100%
        assert metric_result.value == 100.0

    def test_mixed_predictions(self) -> None:
        """Test with a mix of correct and incorrect predictions."""
        data = BinaryClassificationData()

        samples = [
            UnclassifiedTextData("question", "question"),  # TP
            UnclassifiedTextData("question", "other"),  # FP
            UnclassifiedTextData("question", "question"),  # TP
            UnclassifiedTextData("other", "other"),  # TN
            UnclassifiedTextData("other", "question"),  # FN
        ]

        for sample in samples:
            result = precision_classifier(sample)
            data.add_result(result)

        metric_result = precision_aggregator(data, step=0, epoch=None)

        # 2 TP, 1 FP -> precision = 2/3 = 66.67%
        assert abs(metric_result.value - 66.666666) < 0.01
