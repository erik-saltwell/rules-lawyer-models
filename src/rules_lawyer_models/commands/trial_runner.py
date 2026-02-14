"""Runs a single training trial: load model, train, evaluate, report metrics."""

from __future__ import annotations

from collections.abc import Callable

from datasets import DatasetDict

from rules_lawyer_models.core import RunContext
from rules_lawyer_models.evaluation import (
    BinaryClassificationMetric,
    LoggerMetricsReporter,
    MetricResult,
    MetricsManager,
    MetricsReportingManager,
    WandbMetricsReporter,
    accuracy,
    f1_score,
    precision,
)
from rules_lawyer_models.training import (
    TrainingMetaOptions,
    TrainingRunConfiguration,
    create_trainer,
    load_base_model,
    run_training,
)
from rules_lawyer_models.utils import flush_gpu_memory


def run_single_trial(
    meta_options: TrainingMetaOptions,
    run_configuration: TrainingRunConfiguration,
    dataset_dict: DatasetDict,
    ctxt: RunContext,
    content_column_name: str,
    str_label_column_name: str,
    predictions_column_name: str,
    positive_class: str,
    clean_prediction_fn: Callable[[str], str],
    report_to_wandb: bool = True,
    aggregate_predictions: list[Callable] | None = None,
) -> list[MetricResult]:
    """Execute one full train+eval trial. Returns metric results.

    The caller is responsible for the wandb run lifecycle (context manager or sweep agent).
    """
    if aggregate_predictions is None:
        aggregate_predictions = [accuracy, precision, f1_score]

    training_options = meta_options.to_training_options()
    model = None
    trainer = None
    tokenizer = None

    try:
        model, tokenizer = load_base_model(run_configuration, training_options, run_configuration.base_model_name)

        trainer = create_trainer(
            model,
            tokenizer,
            run_configuration,
            training_options,
            report_to_wandb,
            dataset_dict["train"],
            dataset_dict["validation"],
        )
        run_training(model, tokenizer, trainer, run_configuration, training_options)

        metrics_manager = MetricsManager(
            [
                BinaryClassificationMetric(
                    inputs_column_name=content_column_name,
                    ground_truth_column_name=str_label_column_name,
                    predictions_column_name=predictions_column_name,
                    positive_class=positive_class,
                    aggregate_predictions=aggregate_predictions,
                ),
            ],
            logger=ctxt.logger,
        )

        results = metrics_manager.calculate_metrics(
            dataset_dict["test"],
            model,
            tokenizer,
            run_configuration.system_prompt_id,
            content_column_name,
            predictions_column_name,
            clean_prediction_fn,
            max_new_tokens=10,
        )

        reporting_manager = MetricsReportingManager(
            [
                LoggerMetricsReporter(ctxt.logger),
                WandbMetricsReporter(),
            ]
        )
        reporting_manager.report(results)

        return results
    finally:
        del trainer
        del model
        del tokenizer
        flush_gpu_memory()
