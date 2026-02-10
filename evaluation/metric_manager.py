from __future__ import annotations

from collections.abc import Callable, Iterable

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from evaluation.metric_protocol import MetricProtocol
from evaluation.metric_result import MetricResult
from evaluation.model_generator import ModelGenerator
from rules_lawyer_models.data.dataset_helper import add_predictions_column
from rules_lawyer_models.utils.text_fragments import FragmentID


class MetricsManager:
    """Registry of MetricProtocol instances with a method to compute all of them."""

    def __init__(self, metrics: Iterable[MetricProtocol] = ()) -> None:
        self._metrics: list[MetricProtocol] = list(metrics)

    def register(self, metric: MetricProtocol) -> None:
        self._metrics.append(metric)

    def clear(self) -> None:
        self._metrics.clear()

    def calculate_metrics(
        self,
        dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt_id: FragmentID,
        content_column_name: str,
        predictions_column_name: str,
        clean_prediction: Callable[[str], str],
        max_new_tokens: int = 128,
    ) -> list[MetricResult]:
        """Generate predictions on a dataset and compute every registered metric.

        Args:
            dataset: The evaluation dataset.
            model: The trained model (e.g. from the training pipeline).
            tokenizer: The tokenizer paired with the model.
            system_prompt_id: Fragment ID for the system prompt used during generation.
            content_column_name: Column containing user input text.
            predictions_column_name: Name for the new predictions column.
            clean_prediction: Function applied to each raw model output.
            max_new_tokens: Maximum tokens to generate per row.

        Returns:
            A list of MetricResult, one per registered metric.
        """
        generator = ModelGenerator(model, tokenizer, clean_prediction, max_new_tokens)
        dataset_with_predictions = add_predictions_column(
            dataset,
            system_prompt_id,
            content_column_name,
            predictions_column_name,
            generator,
        )
        return [metric.compute_metric(dataset_with_predictions) for metric in self._metrics]
