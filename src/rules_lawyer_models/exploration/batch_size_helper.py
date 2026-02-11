from dataclasses import replace
from enum import Enum

import torch
from datasets import Dataset

from rules_lawyer_models.core.run_context import RunContext
from rules_lawyer_models.training import (
    EvalationSettings,
    SettingsForTrainingOptionsFactory,
    StepSize,
    TrainingLength,
    TrainingOptions,
    TrainingRunConfiguration,
    create_trainer,
    load_base_model,
    run_training,
)
from rules_lawyer_models.utils import flush_gpu_memory


class BatchSizeStrategy(Enum):
    INCREMENT_BY_1 = "increment_by_1"
    INCREMENT_BY_2 = "increment_by_2"
    INCREMENT_BY_5 = "increment_by_5"
    DOUBLE = "double"


def _next_batch_size(current: int, strategy: BatchSizeStrategy) -> int:
    match strategy:
        case BatchSizeStrategy.INCREMENT_BY_1:
            return current + 1
        case BatchSizeStrategy.INCREMENT_BY_2:
            return current + 2
        case BatchSizeStrategy.INCREMENT_BY_5:
            return current + 5
        case BatchSizeStrategy.DOUBLE:
            return current * 2


def find_max_batch_size(
    run_configuration: TrainingRunConfiguration,
    max_sequence_length: int,
    ctxt: RunContext,
    train_dataset: Dataset,
    strategy: BatchSizeStrategy = BatchSizeStrategy.INCREMENT_BY_1,
    starting_batch_size: int = 1,
    max_batch_size: int | None = None,
) -> int:
    factory_settings = SettingsForTrainingOptionsFactory.get_simple_default()
    training_options: TrainingOptions = factory_settings.to_training_options()
    model, tokenizer = load_base_model(run_configuration, training_options, run_configuration.base_model_name)

    last_good = 0
    batch_size = starting_batch_size
    trainer = None
    base_configuration = replace(
        run_configuration,
        evaluation_settings=EvalationSettings.from_disabled(),
        training_length=TrainingLength(use_steps=True, max_steps=1),
        step_size=StepSize(
            max_sequence_length=max_sequence_length,
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=1,
        ),
    )
    while True:
        probe_config = replace(
            base_configuration,
            step_size=StepSize(
                max_sequence_length=max_sequence_length,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=1,
            ),
        )

        try:
            trainer = create_trainer(model, tokenizer, probe_config, training_options, False, train_dataset)
            ctxt.logger.report_message(f"Probing Batch Size: {batch_size}")
            if not probe_config.step_size.per_device_batch_size == batch_size:
                raise ValueError(
                    f"Expected Batch Size: {batch_size}, got {probe_config.step_size.per_device_batch_size}"
                )
            run_training(model, tokenizer, trainer, run_configuration, training_options)
            last_good = batch_size
            batch_size = _next_batch_size(batch_size, strategy)
            if max_batch_size is not None and batch_size > max_batch_size:
                ctxt.logger.report_message(
                    f"Reached max batch size cap ({max_batch_size}). Largest successful: {last_good}"
                )
                del model, tokenizer
                flush_gpu_memory()
                return last_good
            del trainer
            flush_gpu_memory()
        except torch.cuda.OutOfMemoryError:
            ctxt.logger.report_message(f"Largest succesfull batch size: {last_good}")
            del model, tokenizer
            flush_gpu_memory()
            return last_good
