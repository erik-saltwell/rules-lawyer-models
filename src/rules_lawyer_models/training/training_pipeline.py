from unsloth import FastLanguageModel  # isort: skip
from typing import cast

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import TrainOutput
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from rules_lawyer_models.training.training_options import TrainingOptions
from rules_lawyer_models.training.training_run_configuration import TrainingRunConfiguration
from rules_lawyer_models.utils import get_model_data


def load_base_model(
    run_configuration: TrainingRunConfiguration,
    training_options: TrainingOptions,
    base_model_name: str,
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
) -> SFTTrainer:
    config: SFTConfig = run_configuration.create_sft_config(training_options, report_to_wandb)
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=run_configuration.train_dataset,
        eval_dataset=run_configuration.evaluation_settings.evaluation_dataset,
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


def run_training(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, trainer: SFTTrainer) -> TrainOutput:
    training_output = trainer.train()
    return cast(TrainOutput, training_output)


# def run_pipeline(
#     run_configuration: TrainingRunConfiguration,
#     factory_settings: SettingsForTrainingOptionsFactory,
#     ctxt: RunContext,
# ) -> None:
#     training_options: TrainingOptions = create_training_options(factory_settings)

#     os.environ["WANDB_LOG_MODEL"] = "checkpoint"
#     os.environ["WANDB_PROJECT"] = run_configuration.target_model_name

#     wandb.login()
#     config: dict[str, Any] = {
#         "training_options": training_options.to_dict(),
#         "run_configuration": run_configuration.to_dict(),
#     }

#     with wandb.init(project=run_configuration.target_model_name, config=config) as _run:
#         model: PreTrainedModel
#         tokenizer: PreTrainedTokenizerBase
#         trainer: SFTTrainer
#         model, tokenizer = load_base_model(run_configuration, training_options, run_configuration.base_model_name)
#         trainer = create_trainer(
#             model, tokenizer, run_configuration, training_options, True, run_configuration.target_model_name
#         )
#         run_training(model, tokenizer, trainer)
