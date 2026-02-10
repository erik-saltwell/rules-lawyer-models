from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from datasets import Dataset
from trl.trainer.sft_config import SFTConfig

from rules_lawyer_models.serialization.dataset_serializer import load_dataset_from_hf
from rules_lawyer_models.utils.dataset_name import DatasetName
from rules_lawyer_models.utils.model_name import BaseModelName, TargetModelName
from rules_lawyer_models.utils.text_fragments import FragmentID, get_fragment

from .training_options import TrainingOptions

STEPS: str = "steps"
NO: str = "no"


@dataclass
class TrainingLength:
    use_steps: bool = False
    max_steps: int = -1
    max_epochs: float = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalationSettings:
    evaluation_enabled: bool
    evaluation_steps: int
    evaluation_dataset: Dataset | None

    @property
    def evaluation_strategy(self) -> str:
        return STEPS if self.evaluation_enabled else NO

    @property
    def save_strategy(self) -> str:
        return STEPS if self.evaluation_enabled else NO

    @property
    def load_best_model_at_end(self) -> bool:
        return self.evaluation_enabled

    @property
    def metric_for_best_model(self) -> str | None:
        return "eval_loss" if self.evaluation_enabled else None

    @property
    def greater_is_better(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "evaluation_dataset"}

    @classmethod
    def from_disabled(cls) -> EvalationSettings:
        return EvalationSettings(False, -1, None)

    @classmethod
    def from_enabled(cls, evaluation_dataset: Dataset, evaluation_steps: int) -> EvalationSettings:
        return EvalationSettings(True, evaluation_steps, evaluation_dataset)


@dataclass
class StepSize:
    max_sequence_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int

    @property
    def total_step_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunConfiguration:
    train_dataset: Dataset
    base_model_name: BaseModelName
    target_model_name: TargetModelName
    training_column_name: str
    system_prompt_id: FragmentID

    step_size: StepSize
    training_length: TrainingLength
    evaluation_settings: EvalationSettings

    train_on_outputs_only: bool = False
    dtype: str | None = None
    load_in_4bit: bool = True
    packing: bool = False
    seed: int = 3412

    def get_system_prompt(self) -> str:
        return get_fragment(self.system_prompt_id)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "train_dataset"}
        d["step_size"] = self.step_size.to_dict()
        d["training_length"] = self.training_length.to_dict()
        d["evaluation_settings"] = self.evaluation_settings.to_dict()
        return d

    def create_sft_config(self, training_options: TrainingOptions, report_to_wandb: bool) -> SFTConfig:
        return SFTConfig(
            dataset_text_field=self.training_column_name,
            max_length=self.step_size.max_sequence_length,
            packing=self.packing,
            per_device_train_batch_size=self.step_size.per_device_batch_size,
            gradient_accumulation_steps=self.step_size.gradient_accumulation_steps,
            warmup_ratio=training_options.warmup_ratio,
            max_steps=self.training_length.max_steps,
            num_train_epochs=self.training_length.max_epochs,
            learning_rate=training_options.learning_rate,
            optim=training_options.optim,
            weight_decay=training_options.weight_decay,
            lr_scheduler_type=training_options.lr_schedular_type,
            output_dir=self.target_model_name,
            seed=self.seed,
            logging_steps=10,
            save_steps=self.evaluation_settings.evaluation_steps,
            save_strategy=self.evaluation_settings.save_strategy,
            eval_strategy=self.evaluation_settings.evaluation_strategy,
            eval_steps=self.evaluation_settings.evaluation_steps,
            fp16_full_eval=True,  # makes eval cheaper on VRAM
            report_to="wandb" if report_to_wandb else "none",  # enable logging to W&B
            # run_name=run_name if report_to_wandb else None,  # name of the W&B run (optional)
            load_best_model_at_end=self.evaluation_settings.load_best_model_at_end,
            metric_for_best_model=self.evaluation_settings.metric_for_best_model,
            greater_is_better=self.evaluation_settings.greater_is_better,
        )

    @classmethod
    def construct_base(
        cls,
        dataset_name: DatasetName,
        split_name: str,
        training_column_name: str,
        base_model: BaseModelName,
        target_model: TargetModelName,
        length: TrainingLength,
        prompt_id: FragmentID,
    ) -> TrainingRunConfiguration:
        return TrainingRunConfiguration(
            train_dataset=load_dataset_from_hf(dataset_name)[split_name],
            training_column_name=training_column_name,
            base_model_name=base_model,
            target_model_name=target_model,
            training_length=length,
            system_prompt_id=prompt_id,
            evaluation_settings=EvalationSettings.from_disabled(),
            step_size=StepSize(1024, 2, 8),
        )
