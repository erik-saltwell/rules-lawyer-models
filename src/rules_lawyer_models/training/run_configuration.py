from dataclasses import dataclass

from datasets import Dataset

from rules_lawyer_models.utils.model_name import BaseModelName


@dataclass
class TrainingLength:
    use_steps: bool = False
    max_steps: int = -1
    max_epochs: float = 3


@dataclass
class RunConfiguration:
    train_dataset: Dataset
    eval_dataset: Dataset
    model_name: BaseModelName
    training_length: TrainingLength
    max_sequence_lenth: int = 1024
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    dtype: str | None = None
    load_in_4bit: bool = True
    packing: bool = False
    dataset_text_column_name: str = "text"
    seed: int = 3412
    output_dir: str = "outputs"
    report_to: str = "none"
    logging_steps: int = 50
    train_on_outputs_only = False

    def compute_step_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
