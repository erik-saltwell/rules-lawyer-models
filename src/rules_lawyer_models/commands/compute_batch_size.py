import logging
from dataclasses import dataclass, replace

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

from rules_lawyer_models.core.run_context import RunContext
from rules_lawyer_models.data import add_string_label_column, add_training_column, make_stress_split
from rules_lawyer_models.serialization.token_serializer import load_tokenizer_from_hf
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
from rules_lawyer_models.utils import BaseModelName, flush_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class ComputeBatchSizeCommand:
    run_configuration: TrainingRunConfiguration
    samples_count: int
    content_column_name: str
    labels_column_name: str
    max_sequence_length: int

    def execute(self, ctxt: RunContext) -> None:
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        stress_dataset = self.create_stress_set(self.run_configuration.base_model_name)
        self.run_configuration = replace(self.run_configuration, train_dataset=stress_dataset)
        _best_batch_size: int = self.find_max_batch_size(stress_dataset, ctxt)
        # ctxt.logger.report_message(f"Max batch size: {best_batch_size}")

    def create_stress_set(self, base_model_name: BaseModelName) -> Dataset:
        dataset: Dataset = self.run_configuration.train_dataset
        tokenizer: PreTrainedTokenizerBase = load_tokenizer_from_hf(base_model_name)
        string_labels_column_name = "str_" + self.labels_column_name
        training_column_name = self.run_configuration.training_column_name
        dataset = add_string_label_column(dataset, self.labels_column_name, string_labels_column_name)

        stress_dataset = add_training_column(
            base_model_name,
            dataset,
            self.content_column_name,
            string_labels_column_name,
            training_column_name,
            self.run_configuration.get_system_prompt(),
            tokenizer,
        )
        stress_dataset = make_stress_split(stress_dataset, self.samples_count, training_column_name, tokenizer)

        del tokenizer
        flush_gpu_memory()
        return stress_dataset

    def find_max_batch_size(self, dataset: Dataset, ctxt: RunContext) -> int:
        run_configuration: TrainingRunConfiguration = self.run_configuration
        factory_settings = SettingsForTrainingOptionsFactory.get_simple_default()
        training_options: TrainingOptions = factory_settings.to_training_options()
        model, tokenizer = load_base_model(run_configuration, training_options, self.run_configuration.base_model_name)

        last_good = 0
        batch_size = 1
        trainer = None
        base_configuration = replace(
            run_configuration,
            evaluation_settings=EvalationSettings.from_disabled(),
            training_length=TrainingLength(use_steps=True, max_steps=1),
            step_size=StepSize(
                max_sequence_length=self.max_sequence_length,
                per_device_batch_size=batch_size,
                gradient_accumulation_steps=1,
            ),
        )
        while True:
            probe_config = replace(
                base_configuration,
                step_size=StepSize(
                    max_sequence_length=self.max_sequence_length,
                    per_device_batch_size=batch_size,
                    gradient_accumulation_steps=1,
                ),
            )

            try:
                trainer = create_trainer(model, tokenizer, probe_config, training_options, False)
                ctxt.logger.report_message(f"Probing Batch Size: {batch_size}")
                if not probe_config.step_size.per_device_batch_size == batch_size:
                    raise ValueError(
                        f"Expected Batch Size: {batch_size}, got {probe_config.step_size.per_device_batch_size}"
                    )
                run_training(model, tokenizer, trainer, self.run_configuration, training_options)
                last_good = batch_size
                batch_size += 1
                del trainer
                flush_gpu_memory()
            except torch.cuda.OutOfMemoryError:
                ctxt.logger.report_message(f"Largest succesfull batch size: {last_good}")
                del model, tokenizer
                flush_gpu_memory()
                return last_good
