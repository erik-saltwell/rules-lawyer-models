from unsloth import FastLanguageModel  # isort: skip

from datasets import Dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from rules_lawyer_models.core.run_context import RunContext


def perform_test_run(ctxt: RunContext, training_dataset: Dataset) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Can select any from the below:
        # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
        # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
        # And also all Instruct versions and Math. Coding verisons!
        model_name=ctxt.base_model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=training_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=1024,
            packing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_ratio=0.05,
            max_steps=60,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            save_steps=50,
        ),
    )

    _ = trainer.train()
