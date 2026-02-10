from __future__ import annotations

from collections.abc import Callable

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from evaluation.metric_protocol import MetricProtocol
from evaluation.metric_result import MetricResult
from evaluation.model_generator import ModelGenerator
from rules_lawyer_models.data.dataset_helper import add_predictions_column
from rules_lawyer_models.utils.text_fragments import FragmentID


def calculate_metrics(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    metrics: list[MetricProtocol],
    system_prompt_id: FragmentID,
    content_column_name: str,
    predictions_column_name: str,
    clean_prediction: Callable[[str], str],
    max_new_tokens: int = 128,
) -> list[MetricResult]:
    """Generate predictions on a dataset and compute every supplied metric.

    Args:
        dataset: The evaluation dataset.
        model: The trained model (e.g. from the training pipeline).
        tokenizer: The tokenizer paired with the model.
        metrics: Metrics to compute against the predictions.
        system_prompt_id: Fragment ID for the system prompt used during generation.
        content_column_name: Column containing user input text.
        predictions_column_name: Name for the new predictions column.
        clean_prediction: Function applied to each raw model output.
        max_new_tokens: Maximum tokens to generate per row.

    Returns:
        A list of MetricResult, one per metric.
    """
    generator = ModelGenerator(model, tokenizer, clean_prediction, max_new_tokens)
    dataset_with_predictions = add_predictions_column(
        dataset,
        system_prompt_id,
        content_column_name,
        predictions_column_name,
        generator,
    )
    return [metric.compute_metric(dataset_with_predictions) for metric in metrics]
