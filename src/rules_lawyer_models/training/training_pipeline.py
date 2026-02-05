from unsloth import FastLanguageModel  # isort: skip
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from rules_lawyer_models.core.run_context import RunContext
from rules_lawyer_models.training.run_configuration import RunConfiguration
from rules_lawyer_models.training.training_options import TrainingOptions
from rules_lawyer_models.training.training_options_factory import (
    SettingsForTrainingOptionsFactory,
    create_training_options,
)


def run_pipeline(
    factory_settings: SettingsForTrainingOptionsFactory,
    run_configuration: RunConfiguration,
    ctxt: RunContext,
) -> None:
    training_options: TrainingOptions = create_training_options(factory_settings)

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ctxt.base_model_name,
        max_seq_length=run_configuration.max_sequence_lenth,
        dtype=run_configuration.dtype,
        load_in_4bit=run_configuration.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    loftq_config = None
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=run_configuration.train_dataset,
        eval_dataset=run_configuration.eval_dataset,
        args=SFTConfig(
            dataset_text_field=run_configuration.dataset_text_column_name,
            max_length=run_configuration.max_sequence_lenth,
            packing=run_configuration.packing,
            per_device_train_batch_size=run_configuration.per_device_train_batch_size,
            gradient_accumulation_steps=run_configuration.gradient_accumulation_steps,
            warmup_ratio=training_options.warmup_ratio,
            max_steps=-1
            if not run_configuration.training_length.use_steps
            else run_configuration.training_length.max_steps,
            num_train_epochs=RunConfiguration.training_length.max_epochs
            if not run_configuration.training_length.use_steps
            else -1,
            learning_rate=training_options.learning_rate,
            logging_steps=run_configuration.logging_steps,
            optim=training_options.optim,
            weight_decay=training_options.weight_decay,
            lr_scheduler_type=training_options.lr_schedular_type,
            output_dir=run_configuration.output_dir,
            report_to=run_configuration.report_to,
            save_steps=run_configuration.logging_steps,
            seed=run_configuration.seed,
        ),
    )

    _ = trainer.train()
