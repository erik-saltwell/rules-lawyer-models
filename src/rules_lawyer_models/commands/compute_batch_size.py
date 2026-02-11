import logging
from dataclasses import dataclass

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

from rules_lawyer_models.core.run_context import RunContext
from rules_lawyer_models.data import add_string_label_column, add_training_column, make_stress_split
from rules_lawyer_models.exploration import find_max_batch_size
from rules_lawyer_models.serialization.token_serializer import load_tokenizer_from_hf
from rules_lawyer_models.training import TrainingRunConfiguration
from rules_lawyer_models.utils import BaseModelName, flush_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class ComputeBatchSizeCommand:
    run_configuration: TrainingRunConfiguration
    train_dataset: Dataset
    samples_count: int
    content_column_name: str
    labels_column_name: str
    max_sequence_length: int

    def execute(self, ctxt: RunContext) -> None:
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        stress_dataset = self.create_stress_set(self.run_configuration.base_model_name)
        _best_batch_size: int = find_max_batch_size(
            self.run_configuration, self.max_sequence_length, ctxt, train_dataset=stress_dataset
        )

    def create_stress_set(self, base_model_name: BaseModelName) -> Dataset:
        dataset: Dataset = self.train_dataset
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
