from rules_lawyer_models.utils.model_data import BaseModelName

from unsloth import FastLanguageModel  # isort: skip
from typing import cast

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import TrainOutput
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from rules_lawyer_models.utils import get_model_data

from .training_options import TrainingOptions
from .training_run_configuration import TrainingRunConfiguration


def load_base_model(
    run_configuration: TrainingRunConfiguration,
    training_options: TrainingOptions,
    base_model_name: BaseModelName,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=run_configuration.step_size.max_sequence_length,
        dtype=run_configuration.dtype,
        load_in_4bit=run_configuration.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    loftq_config: dict[str, int] | None = None
    if training_options.use_loftq:
        loftq_config = {"loftq_bits": training_options.loftq_bits, "loftq_iter": training_options.loftq_iter}

    model = FastLanguageModel.get_peft_model(
        model,
        training_options.r,
        target_modules=training_options.target_modules,
        lora_alpha=training_options.lora_alpha,
        lora_dropout=training_options.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=run_configuration.seed,
        use_rslora=training_options.use_rslora,
        loftq_config=loftq_config,
    )

    return (model, tokenizer)


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    run_configuration: TrainingRunConfiguration,
    training_options: TrainingOptions,
    report_to_wandb: bool,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
) -> SFTTrainer:
    config: SFTConfig = run_configuration.create_sft_config(training_options, report_to_wandb)
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
    )
    if run_configuration.train_on_outputs_only:
        model_data = get_model_data(run_configuration.base_model_name)
        trainer = train_on_responses_only(
            trainer=trainer,
            instruction_part=model_data.instruction_seperator,
            response_part=model_data.response_seperator,
        )
    return trainer


def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    trainer: SFTTrainer,
    run_configuration: TrainingRunConfiguration,
    training_options: TrainingOptions,
) -> TrainOutput:
    training_output = cast(TrainOutput, trainer.train())
    return training_output  # pyright: ignore
